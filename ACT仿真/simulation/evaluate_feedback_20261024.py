"""CUDA ACT evaluation with 30 Hz actions and explicit visual-query modes.

chunk100 queries on frames 0/100/200/300/400. ensemble1 queries fresh RGB/state
on every action frame and combines covering predictions oldest first using
exp(-0.01*i). These rates describe simulation time, not wall-clock real time.
No teacher arrays, object coordinates, or task labels are policy inputs.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cpu_budget import CPU_BUDGET

import mujoco
import numpy as np
from PIL import Image

from analyze_position_generalization import longest_run
from evaluate_act_ensemble import SparseTemporalEnsembler, finger_contacts
from joint_mapping import dataset_state_to_sim, model_action_to_sim, mapping_metadata
from preview import load_model
from run_checkpoint_comparison import SOURCE, create_scene
from run_position_generalization import model_hashes, normalized_scene_tree
from run_task import brick_metrics, check_physics, digest, save_json
from task_dynamics import ServoController, initialize, read_public_state, native_to_public

ACTION_HZ, CHUNK_SIZE, DURATION_S = 30, 100, 15
QUERY_EVERY = {'chunk100': 100, 'ensemble1': 1}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--position-plan', required=True, type=Path)
    parser.add_argument('--split', required=True, choices=('baseline', 'validation', 'test'))
    parser.add_argument('--mode', required=True, choices=tuple(QUERY_EVERY))
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    return parser.parse_args(argv)


def select_positions(plan, split):
    points = []
    for point in plan['positions']:
        if point['split'] != split:
            continue
        if not re.fullmatch(r'[A-Za-z0-9_+\-]+', point['id']):
            raise ValueError('Invalid point ID')
        xy = np.asarray(point['offset_xy_m'], dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all() or (np.abs(xy) > .0400000001).any():
            raise ValueError('Expected finite initial XY offsets within +/-40 mm')
        points.append({'id': point['id'], 'split': split, 'offset_xy_m': xy.tolist()})
    if not points or len({point['id'] for point in points}) != len(points):
        raise ValueError('Expected nonempty positions with unique IDs')
    return points


def query_frames(mode, total_frames=ACTION_HZ * DURATION_S):
    return np.arange(0, total_frames, QUERY_EVERY[mode], dtype=np.int64)


def weighted_action(chunks, starts, frame):
    """Independent reconstruction for auditing; no ensembler object is used."""
    active = np.flatnonzero((starts <= frame) & (frame < starts + CHUNK_SIZE))
    if not len(active):
        raise ValueError('No prediction covers the requested frame')
    rows = frame - starts[active]
    weights = np.exp(-.01 * np.arange(len(active), dtype=np.float64))
    weights /= weights.sum()
    action = np.sum(chunks[active, rows].astype(np.float64) * weights[:, None], axis=0)
    return action, active, rows, weights


def audit_arrays(arrays, result, mode):
    """Reconstruct all actions/provenance and pre-action observation states."""
    frames = len(arrays['target_native'])
    starts, chunks = arrays['query_frame_index'], arrays['action_chunks']
    expected = query_frames(mode, frames)
    if (not np.array_equal(starts, expected) or chunks.shape != (len(expected), CHUNK_SIZE, 7)
            or not np.array_equal(arrays['frame_index'], np.arange(frames))):
        raise ValueError('Mode/query/chunk timeline differs from recorded execution')
    for frame in range(frames):
        action, active, rows, weights = weighted_action(chunks, starts, frame)
        count = int(arrays['active_prediction_count'][frame])
        if (count != len(active)
                or not np.array_equal(arrays['ensemble_query_indices'][frame, :count], active)
                or not np.array_equal(arrays['ensemble_chunk_rows'][frame, :count], rows)
                or not np.array_equal(arrays['ensemble_weights'][frame, :count], weights)
                or (arrays['ensemble_query_indices'][frame, count:] != -1).any()
                or (arrays['ensemble_chunk_rows'][frame, count:] != -1).any()
                or (arrays['ensemble_weights'][frame, count:] != 0).any()
                or not np.array_equal(action, arrays['ensembled_public_action'][frame])
                or not np.array_equal(model_action_to_sim(action), arrays['target_native'][frame])):
            raise ValueError('Applied action or provenance differs from the raw predictions')
    if len(result['inference']) != len(expected):
        raise ValueError('Neural-query records differ from the actual query schedule')
    for index, frame in enumerate(starts):
        observed = (np.asarray(result['initial_public_state']) if frame == 0
                    else arrays['public_state'][frame - 1])
        if (not np.array_equal(arrays['chunk_observation_state'][index], observed)
                or result['inference'][index]['frame_index'] != int(frame)
                or result['inference'][index]['request_index'] != index + 1):
            raise ValueError('Query did not use the actual state before its action frame')
    dt = result['evaluation']['physics_dt_s']
    if (np.max(np.abs(arrays['query_actual_sim_time'] - starts / ACTION_HZ)) > dt + 1e-8
            or np.max(np.abs(arrays['sim_time'] - (np.arange(frames) + 1) / ACTION_HZ)) > dt + 1e-8):
        raise ValueError('Recorded simulation times differ from the 30Hz action schedule')
    return {'all_applied_targets_exact_raw_prediction_conversion': True,
        'all_weighted_actions_and_provenance_exact': True,
        'all_query_states_match_actual_pre_action_state': True,
        'query_count': len(expected), 'action_frames': frames}


def run_feedback(scene, metadata, output, model_dir, initial, frozen, mode):
    from policy_process_cuda_20261024 import CudaPolicyProcess

    output.mkdir(parents=True, exist_ok=False)
    model, data = load_model(scene), None
    data = mujoco.MjData(model)
    initialize(model, data, initial)
    controller = ServoController(model)
    for _ in range(500):
        controller.step(data, initial)
    initial_public = read_public_state(model, data)
    if np.max(np.abs(initial_public - np.asarray(frozen['successful_reference_initial_public_state']))) > 1e-7:
        raise ValueError('Initial settled robot pose changed')
    initial_position = data.body('target_brick').xpos.copy()
    renderer = mujoco.Renderer(model, height=480, width=640)
    options = mujoco.MjvOption()
    options.geomgroup[3], options.sitegroup[:] = 0, 0
    names = ('frame_index', 'public_state', 'target_native', 'brick_position', 'qpos', 'qvel',
        'sim_time', 'action_chunks', 'chunk_observation_state', 'query_frame_index',
        'query_actual_sim_time', 'ensembled_public_action', 'active_prediction_count',
        'ensemble_query_indices', 'ensemble_chunk_rows', 'ensemble_weights', 'finger_contacts')
    saved = {name: [] for name in names}
    ensemble, inference, penetration = SparseTemporalEnsembler(), [], Counter()
    worker, status, error_text, lift_max = None, 'completed', None, 0.
    start_wall, sim_start = time.perf_counter(), data.time
    try:
        worker = CudaPolicyProcess(model_dir, output)
        save_json(output / 'model_identity.json', worker.identity)
        if not str(worker.identity['model_identity']['device']).startswith('cuda'):
            raise ValueError('CUDA forward is required; a CPU fallback is not allowed')
        for frame in range(ACTION_HZ * DURATION_S):
            if frame % QUERY_EVERY[mode] == 0:
                observations = {}
                for role, camera in [('wrist', 'wrist_rgb'), ('top', 'overhead_rgb')]:
                    options.geomgroup[4] = int(role != 'wrist')
                    renderer.update_scene(data, camera=camera, scene_option=options)
                    observations[role] = renderer.render().copy()
                    if frame == 0:
                        Image.fromarray(observations[role]).save(output / f'initial_{role}.png')
                public = read_public_state(model, data)
                chunk, timing = worker.predict(public, observations['wrist'], observations['top'], len(inference))
                ensemble.add(frame, chunk)
                saved['action_chunks'].append(chunk)
                saved['chunk_observation_state'].append(public)
                saved['query_frame_index'].append(frame)
                saved['query_actual_sim_time'].append(data.time - sim_start)
                timing.update({'frame_index': frame, 'observation_time_s': frame / ACTION_HZ,
                    'actual_observation_sim_time_s': float(data.time - sim_start)})
                inference.append(timing)
            public_action, sources = ensemble.action(frame)
            target = model_action_to_sim(public_action)
            end_time = sim_start + (frame + 1) / ACTION_HZ
            while data.time < end_time - 1e-10:
                previous_time = data.time
                controller.step(data, target)
                check_physics(data, previous_time)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise FloatingPointError('Non-finite simulation state')
            mujoco.mj_forward(model, data)
            position = data.body('target_brick').xpos.copy()
            lift_max = max(lift_max, float(position[2] - initial_position[2]))
            for name, value in {'frame_index': frame, 'public_state': read_public_state(model, data),
                'target_native': target, 'brick_position': position, 'qpos': data.qpos.copy(),
                'qvel': data.qvel.copy(), 'sim_time': data.time - sim_start,
                'ensembled_public_action': public_action, 'active_prediction_count': len(sources),
                'finger_contacts': finger_contacts(model, data)}.items():
                saved[name].append(value)
            indices = np.full(CHUNK_SIZE, -1, dtype=np.int64)
            rows, weights = indices.copy(), np.zeros(CHUNK_SIZE, dtype=np.float64)
            for index, source in enumerate(sources):
                indices[index], rows[index], weights[index] = (
                    source['query_index'], source['chunk_row'], source['normalized_weight'])
            saved['ensemble_query_indices'].append(indices)
            saved['ensemble_chunk_rows'].append(rows)
            saved['ensemble_weights'].append(weights)
            for contact in data.contact[:data.ncon]:
                if contact.dist < -.001:
                    geom_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or str(g)
                             for g in (contact.geom1, contact.geom2)]
                    penetration[' / '.join(geom_names)] += 1
    except Exception as error:
        status, error_text = 'failed', f'{type(error).__name__}: {error}'
    finally:
        if worker:
            try:
                worker.close()
            except Exception as error:
                status, error_text = 'failed', f'{error_text}; cleanup: {error}'
        renderer.close()
        np.savez_compressed(output / 'rollout_arrays.npz', **{name: np.asarray(value) for name, value in saved.items()})
        result = {'status': status, 'error': error_text, 'model_directory': str(model_dir),
            'model_sha256': digest(model_dir / 'model.safetensors'), 'mode': mode,
            'elapsed_wall_s': time.perf_counter() - start_wall,
            'simulated_duration_s': float(data.time - sim_start),
            'policy_actions_applied': len(saved['target_native']),
            'actual_inference_queries': len(inference), 'actual_neural_forward_calls': len(inference),
            'inference': inference, 'initial_public_state': initial_public.tolist(),
            'physics_steps': int(round(data.time / model.opt.timestep)),
            'metrics': brick_metrics(model, data, initial_position, metadata, lift_max),
            'mujoco_warning_counts': [int(w.number) for w in data.warning],
            'penetrations_over_1mm_frame_pair_counts': penetration.most_common(12),
            'mapping': mapping_metadata(), 'simulation_controller': controller.metadata(),
            'object_pose_assigned_during_rollout': False, 'grasp_weld_or_object_attach_used': False,
            'hardware_io': False, 'training_steps': 0, 'video_encoded': False,
            'policy_inputs': ['state', 'wrist', 'top'], 'object_truth_passed_to_policy': False,
            'teacher_arrays_read': False, 'teacher_actions_read': False, 'cpu_budget': CPU_BUDGET,
            'evaluation': {'mode': mode, 'action_hz': ACTION_HZ, 'policy_hz': ACTION_HZ,
                'vision_query_hz_simulated_time': ACTION_HZ / QUERY_EVERY[mode],
                'query_every_action_frames': QUERY_EVERY[mode], 'chunk_size': CHUNK_SIZE,
                'executed_actions_per_chunk': 100 if mode == 'chunk100' else None,
                'temporal_ensemble_coeff': .01 if mode == 'ensemble1' else None,
                'weight_rule': 'Covering predictions oldest first; normalized exp(-.01*rank)',
                'actual_forward_device': 'cuda', 'wall_clock_realtime_claimed': False,
                'timing': '30Hz action grid in simulation time; physics advances in 1ms increments; synchronous fresh forward pauses simulation time.',
                'physics_dt_s': float(model.opt.timestep)}}
        save_json(output / 'result.json', result)
    return result


def audit_rollout(scene, directory, result, mode):
    with np.load(directory / 'rollout_arrays.npz', allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    audit = audit_arrays(arrays, result, mode)
    for query, record in enumerate(result['inference']):
        observation_path, chunk_path = directory / f'observation_{query:03d}.npz', directory / f'chunk_{query:03d}.npz'
        if digest(observation_path) != record['input_sha256'] or digest(chunk_path) != record['output_sha256']:
            raise ValueError('Worker artifact hash changed')
        with np.load(observation_path, allow_pickle=False) as observation:
            if (set(observation.files) != {'state', 'wrist', 'top'}
                    or not np.array_equal(observation['state'], arrays['chunk_observation_state'][query])
                    or any(observation[role].shape != (480, 640, 3) or observation[role].dtype != np.uint8
                           for role in ('wrist', 'top'))):
                raise ValueError('Unexpected policy input or state')
        with np.load(chunk_path, allow_pickle=False) as raw:
            if set(raw.files) != {'action'} or not np.array_equal(raw['action'], arrays['action_chunks'][query]):
                raise ValueError('Saved chunks differ from worker raw predictions')
    model = load_model(scene)
    data = mujoco.MjData(model)
    addresses = [int(model.joint(f'joint{i}').qposadr[0]) for i in range(1, 8)]
    contacts, positions = [], []
    for index, pose in enumerate(arrays['qpos']):
        data.qpos[:], data.qvel[:] = pose, arrays['qvel'][index]
        mujoco.mj_forward(model, data)
        if not np.array_equal(native_to_public(pose[addresses]), arrays['public_state'][index]):
            raise ValueError('Saved public state differs from physical qpos')
        contacts.append(finger_contacts(model, data))
        positions.append(data.body('target_brick').xpos.copy())
    if not np.array_equal(contacts, arrays['finger_contacts']) or not np.allclose(positions, arrays['brick_position'], atol=1e-12, rtol=0):
        raise ValueError('Recorded contact or object pose differs from the physical trace')
    bilateral = arrays['finger_contacts'].all(axis=1)
    lift = arrays['brick_position'][:, 2] - result['metrics']['initial_position_m'][2]
    confirmed = longest_run(bilateral & (lift >= .03)) >= 3
    recomputed = brick_metrics(model, data, np.asarray(result['metrics']['initial_position_m']), {}, float(max(0, lift.max())))
    if recomputed != result['metrics'] or (result['metrics']['success'] and not confirmed):
        raise ValueError('Score recomputation differs or success lacks a sustained elevated grasp')
    return {**audit, 'all_input_output_hashes_match_worker_records': True,
        'physical_score_exact_qpos_qvel_recomputation': True,
        'contact_trace_exact_recomputation': True,
        'left_finger_contact_frames': int(arrays['finger_contacts'][:, 0].sum()),
        'right_finger_contact_frames': int(arrays['finger_contacts'][:, 1].sum()),
        'bilateral_contact_frames': int(bilateral.sum()),
        'first_bilateral_frame': int(np.flatnonzero(bilateral)[0]) if bilateral.any() else None,
        'longest_elevated_bilateral_frames': longest_run(bilateral & (lift >= .03)),
        'physical_grasp_confirmed': confirmed, 'audit_new_ACT_calls': 0, 'audit_new_physics_steps': 0}


def evaluate(args):
    model_dir, position_path, output = args.model_dir.resolve(), args.position_plan.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError('Preserve prior runs; choose a fresh output directory')
    points = select_positions(json.loads(position_path.read_text(encoding='utf-8')), args.split)
    frozen_path = SOURCE / 'protocol.json'
    frozen = json.loads(frozen_path.read_text(encoding='utf-8'))
    if (digest(Path(frozen['camera_fit_path'])) != frozen['camera_fit_sha256']
            or digest(Path(frozen['successful_reference_scene'])) != frozen['successful_reference_scene_sha256']):
        raise ValueError('Camera or reference scene changed')
    before = model_hashes(model_dir)
    code = dict(frozen['code_files_sha256'])
    for name in ('simulation/evaluate_feedback_20261024.py', 'simulation/evaluate_act_ensemble.py',
                 'simulation/run_checkpoint_comparison.py', 'simulation/preview.py',
                 'simulation/analyze_position_generalization.py', 'simulation/policy_process_cuda_20261024.py'):
        code[name] = digest(ROOT / name)
    cuda_worker = ROOT / 'simulation/policy_worker_cuda_20261024.py'
    if cuda_worker.exists():
        code['simulation/policy_worker_cuda_20261024.py'] = digest(cuda_worker)
    sources = {str(path): digest(path) for path in (position_path, frozen_path,
        Path(frozen['camera_fit_path']), Path(frozen['successful_reference_scene']))}

    def check_frozen():
        if model_hashes(model_dir) != before:
            raise ValueError('Checkpoint changed')
        for name, expected in code.items():
            if digest(ROOT / name) != expected:
                raise ValueError(f'Execution code changed: {name}')
        for name, expected in sources.items():
            if digest(Path(name)) != expected:
                raise ValueError(f'Frozen source changed: {name}')

    check_frozen()
    output.mkdir(parents=True, exist_ok=False)
    plan = {'created_utc': datetime.now(timezone.utc).isoformat(), 'model_directory': str(model_dir),
        'model_files_sha256': before, 'position_plan': str(position_path), 'split': args.split,
        'positions': points, 'mode': args.mode, 'duration_s': DURATION_S, 'action_hz': ACTION_HZ,
        'query_every_action_frames': QUERY_EVERY[args.mode], 'chunk_size': CHUNK_SIZE,
        'expected_ACT_queries_per_trial': len(query_frames(args.mode)),
        'vision_query_hz_simulated_time': ACTION_HZ / QUERY_EVERY[args.mode],
        'wall_clock_realtime_claimed': False, 'source_files_sha256': sources,
        'code_files_sha256': code, 'fixed_initial_state_protocol': str(frozen_path),
        'teacher_arrays_read': False, 'teacher_actions_read': False,
        'policy_input_fields': ['state', 'wrist', 'top'],
        'object_truth_usage': 'Initial scene placement and physical scoring only; never a policy input',
        'training_steps_allowed': 0, 'hardware_commands_allowed': 0, 'cpu_budget': CPU_BUDGET}
    save_json(output / 'protocol.json', plan)
    report = {'status': 'running', 'protocol': plan, 'results': [], 'actual_ACT_queries': 0,
        'training_steps': 0, 'hardware_commands_sent': 0}
    save_json(output / 'evaluation.json', report)
    initial = dataset_state_to_sim(frozen['initial_public_joint_target_degrees'])
    fit = json.loads(Path(frozen['camera_fit_path']).read_text(encoding='utf-8'))
    reference = normalized_scene_tree(ET.parse(frozen['successful_reference_scene']).getroot())
    try:
        for point in points:
            check_frozen()
            directory = output / 'trials' / point['id']
            scene, metadata = create_scene(directory, point, frozen, fit, reference)
            result = run_feedback(scene, metadata, directory / 'custom', model_dir, initial, frozen, args.mode)
            report['actual_ACT_queries'] += result['actual_inference_queries']
            entry = {'id': point['id'], 'split': args.split, 'offset_xy_m': point['offset_xy_m'],
                'scene_sha256': digest(scene), 'result': result}
            report['results'].append(entry)
            save_json(output / 'evaluation.json', report)
            if (result['status'] != 'completed' or result['policy_actions_applied'] != 450
                    or result['actual_inference_queries'] != len(query_frames(args.mode))):
                raise RuntimeError(f'Incomplete feedback rollout: {result["error"]}')
            identity = json.loads((directory / 'custom/model_identity.json').read_text(encoding='utf-8'))
            if identity['model_identity']['files_sha256'] != before:
                raise ValueError('Worker loaded a different model or processors')
            entry['contacts'] = audit_rollout(scene, directory / 'custom', result, args.mode)
            save_json(output / 'evaluation.json', report)
            print(json.dumps({'point': point['id'], 'mode': args.mode,
                'success': result['metrics']['success'], 'lift_mm': result['metrics']['maximum_lift_m'] * 1000,
                'ACT_queries': result['actual_inference_queries'],
                'bilateral_frames': entry['contacts']['bilateral_contact_frames']}, ensure_ascii=False), flush=True)
        report['status'] = 'completed_frozen_feedback_ACT_evaluation'
    except Exception as error:
        report.update({'status': 'evaluation_error', 'error': f'{type(error).__name__}: {error}'})
        raise
    finally:
        report['completed_utc'] = datetime.now(timezone.utc).isoformat()
        report['model_files_unchanged'] = model_hashes(model_dir) == before
        report['completed_trials'] = len(report['results'])
        report['successes'] = sum(bool(entry['result']['metrics']['success']) for entry in report['results'])
        save_json(output / 'evaluation.json', report)
    check_frozen()


if __name__ == '__main__':
    evaluate(parse_args())
