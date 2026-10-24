"""Validation-only sparse ACT temporal ensemble: 3 Hz vision, 30 Hz actions.

Every ten action frames a real observation produces a fresh 100-frame chunk.
All predictions covering the current frame are averaged, oldest first with
weights exp(-0.01*i). This sparse variant is not the official every-frame ACT
temporal ensemble. It uses no teacher actions, training, or hardware commands.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cpu_budget import CPU_BUDGET

import numpy as np

CHUNK_SIZE, QUERY_EVERY, ACTION_HZ, COEFFICIENT = 100, 10, 30, .01
POSITION_PROTOCOL = ROOT / 'runs/mujoco_iteration_20261023/multisite_v1/teacher_v2/protocol.json'
DEFAULT_MODEL = ROOT / 'runs/mujoco_iteration_20261023/multisite_v1/adapt_v1/checkpoints/step_000600/pretrained_model'
ENSEMBLER_SOURCE = ROOT / '.venv/Lib/site-packages/lerobot/policies/act/modeling_act.py'


class SparseTemporalEnsembler:
    """Keep complete overlapping chunks, addressed by absolute action frame."""
    def __init__(self):
        self.chunks = []
        self.last_start = self.last_frame = -1
        self.next_query_index = 0

    def add(self, start_frame, chunk):
        if (not isinstance(start_frame, (int, np.integer)) or isinstance(start_frame, bool)
                or start_frame < 0 or start_frame <= self.last_start or start_frame < self.last_frame):
            raise ValueError('Chunk starts must be nonnegative, increasing action-frame integers')
        value = np.asarray(chunk)
        if (value.shape != (CHUNK_SIZE, 7) or value.dtype.kind not in 'fiu'
                or not np.isfinite(value).all()):
            raise ValueError('Expected a finite real 100-by-7 public action chunk')
        self.chunks.append((int(start_frame), self.next_query_index, value.astype(np.float64, copy=True)))
        self.next_query_index += 1
        self.last_start = int(start_frame)

    def action(self, frame):
        if (not isinstance(frame, (int, np.integer)) or isinstance(frame, bool)
                or frame < 0 or frame < self.last_frame):
            raise ValueError('Action frames must be nonnegative and cannot run backwards')
        self.last_frame = int(frame)
        self.chunks = [item for item in self.chunks if item[0] + CHUNK_SIZE > frame]
        active = [item for item in self.chunks if item[0] <= frame]
        if not active:
            raise ValueError('No ACT prediction covers this action frame')
        weights = np.exp(-COEFFICIENT * np.arange(len(active), dtype=np.float64))
        weights /= weights.sum()
        rows = np.stack([chunk[frame - start] for start, _, chunk in active])
        action = np.sum(rows * weights[:, None], axis=0)
        sources = [{'query_index': query, 'start_frame': start, 'chunk_row': int(frame - start),
                    'normalized_weight': float(weight)}
                   for (start, query, _), weight in zip(active, weights, strict=True)]
        return action, sources


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--teacher-protocol', type=Path, default=POSITION_PROTOCOL,
                        help='Fixed source of the four validation XY entries; no teacher arrays are read')
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.teacher_protocol.resolve() != POSITION_PROTOCOL.resolve():
        parser.error('Only the preregistered multisite_v1/teacher_v2/protocol.json is allowed')
    return args


def validation_positions(protocol):
    from evaluate_multisite_act import select_positions
    positions = select_positions(protocol, 'validation')
    if len(positions) != 4:
        raise ValueError('This experiment requires exactly the four predeclared validation positions')
    return positions


def finger_contacts(model, data):
    brick = model.body('target_brick').id
    left, right = (model.body(name).id for name in ('gripper_left', 'gripper_right'))
    fingers = set()
    for contact in data.contact[:data.ncon]:
        if contact.efc_address < 0 or contact.dist > contact.includemargin:
            continue
        first, second = (int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
        if first == brick:
            fingers.add(second)
        elif second == brick:
            fingers.add(first)
    return np.asarray([left in fingers, right in fingers], dtype=bool)


def run_sparse_ensemble(scene, metadata, output, model_dir, initial, protocol):
    """Preserve the frozen servo/physics loop and display-camera render calls."""
    import mujoco
    from PIL import Image
    from joint_mapping import mapping_metadata, model_action_to_sim
    from preview import load_model, overview_camera
    from run_task import PolicyProcess, brick_metrics, check_physics, digest, save_json
    from task_dynamics import ServoController, initialize, read_public_state

    output.mkdir(parents=True, exist_ok=False)
    model = load_model(scene)
    data = mujoco.MjData(model)
    initialize(model, data, initial)
    controller = ServoController(model)
    for _ in range(500):
        controller.step(data, initial)
    initial_public = read_public_state(model, data)
    if np.max(np.abs(initial_public - np.asarray(protocol['successful_reference_initial_public_state']))) > 1e-7:
        raise ValueError('Initial settled robot pose changed')
    initial_position = data.body('target_brick').xpos.copy()
    renderer = mujoco.Renderer(model, height=480, width=640)
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    options.sitegroup[:] = 0
    overview = overview_camera()
    overview.lookat[:] = [.23, 0, .14]
    overview.azimuth, overview.elevation, overview.distance = 125, -40, 1.12
    saved = {name: [] for name in ('frame_index', 'public_state', 'target_native', 'brick_position',
        'qpos', 'sim_time', 'action_chunks', 'chunk_observation_state', 'query_frame_index',
        'ensembled_public_action', 'active_prediction_count', 'ensemble_query_indices',
        'ensemble_chunk_rows', 'ensemble_weights', 'finger_contacts')}
    ensemble, inference, penetration = SparseTemporalEnsembler(), [], Counter()
    worker, status, error_text = None, 'completed', None
    lift_max = 0.
    start_wall, sim_start = time.perf_counter(), data.time
    try:
        worker = PolicyProcess(model_dir, output)
        save_json(output / 'model_identity.json', worker.identity)
        for frame in range(round(protocol['duration_s'] * ACTION_HZ)):
            if frame % QUERY_EVERY == 0:
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
                timing['frame_index'] = frame
                timing['observation_time_s'] = frame / ACTION_HZ
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
            saved['frame_index'].append(frame)
            saved['public_state'].append(read_public_state(model, data))
            saved['target_native'].append(target)
            saved['brick_position'].append(position)
            saved['qpos'].append(data.qpos.copy())
            saved['sim_time'].append(data.time - sim_start)
            saved['ensembled_public_action'].append(public_action)
            saved['active_prediction_count'].append(len(sources))
            saved['finger_contacts'].append(finger_contacts(model, data))
            query_indices = np.full(CHUNK_SIZE // QUERY_EVERY, -1, dtype=np.int64)
            chunk_rows = query_indices.copy()
            weights = np.zeros(CHUNK_SIZE // QUERY_EVERY, dtype=np.float64)
            for index, source in enumerate(sources):
                query_indices[index], chunk_rows[index], weights[index] = (
                    source['query_index'], source['chunk_row'], source['normalized_weight'])
            saved['ensemble_query_indices'].append(query_indices)
            saved['ensemble_chunk_rows'].append(chunk_rows)
            saved['ensemble_weights'].append(weights)
            for contact in data.contact[:data.ncon]:
                if contact.dist < -.001:
                    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or str(g)
                             for g in (contact.geom1, contact.geom2)]
                    penetration[' / '.join(names)] += 1
            if frame % 2 == 0:
                # Retain the old display render sequence. Additional policy
                # samples are necessary for this new feedback/ensemble variant.
                options.geomgroup[4] = 1
                renderer.update_scene(data, camera=overview, scene_option=options)
                renderer.render()
                for role, camera in [('top', 'overhead_rgb'), ('wrist', 'wrist_rgb')]:
                    options.geomgroup[4] = int(role != 'wrist')
                    renderer.update_scene(data, camera=camera, scene_option=options)
                    renderer.render()
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
            'model_sha256': digest(model_dir / 'model.safetensors'),
            'elapsed_wall_s': time.perf_counter() - start_wall,
            'simulated_duration_s': float(data.time - sim_start),
            'policy_actions_applied': len(saved['target_native']),
            'actual_inference_queries': len(inference), 'inference': inference,
            'physics_steps': int(round(data.time / model.opt.timestep)),
            'initial_public_state': initial_public.tolist(),
            'metrics': brick_metrics(model, data, initial_position, metadata, lift_max),
            'mujoco_warning_counts': [int(w.number) for w in data.warning],
            'penetrations_over_1mm_frame_pair_counts': penetration.most_common(12),
            'mapping': mapping_metadata(), 'simulation_controller': controller.metadata(),
            'object_pose_assigned_during_rollout': False, 'grasp_weld_or_object_attach_used': False,
            'hardware_io': False, 'training_steps': 0, 'video_encoded': False,
            'frozen_display_camera_render_sequence_retained': True,
            'additional_real_policy_camera_samples': True,
            'policy_inputs': ['state', 'wrist', 'top'], 'object_truth_passed_to_policy': False,
            'teacher_arrays_read': False, 'teacher_actions_read': False,
            'cpu_budget': CPU_BUDGET, 'evaluation': {'variant': 'sparse_temporal_ensemble',
                'action_hz': ACTION_HZ, 'vision_query_hz': ACTION_HZ / QUERY_EVERY,
                'query_every_action_frames': QUERY_EVERY, 'chunk_size': CHUNK_SIZE,
                'temporal_ensemble_coeff': COEFFICIENT,
                'weights': 'exp(-0.01*i), oldest covering prediction first, normalized at each action frame',
                'official_every_frame_ensemble': False, 'physics_dt_s': float(model.opt.timestep)}}
        save_json(output / 'result.json', result)
    return result


def audit_sparse_rollout(scene, directory, result):
    """Recompute every applied action from raw chunks; never call ACT or mj_step."""
    import mujoco
    from analyze_position_generalization import longest_run
    from joint_mapping import model_action_to_sim
    from preview import load_model
    from run_task import digest

    with np.load(directory / 'rollout_arrays.npz', allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    chunks, starts = arrays['action_chunks'], arrays['query_frame_index']
    if not np.array_equal(starts, np.arange(0, 450, QUERY_EVERY)):
        raise ValueError('Missing or misindexed real ACT queries')
    if chunks.shape != (45, CHUNK_SIZE, 7) or not np.array_equal(arrays['frame_index'], np.arange(450)):
        raise ValueError('Incomplete raw ACT chunks or action-frame trace')
    maximum_difference = 0.
    for frame in range(450):
        active = np.flatnonzero((starts <= frame) & (frame < starts + CHUNK_SIZE))
        rows = frame - starts[active]
        weights = np.exp(-COEFFICIENT * np.arange(len(active), dtype=np.float64))
        weights /= weights.sum()
        action = np.sum(chunks[active, rows].astype(np.float64) * weights[:, None], axis=0)
        maximum_difference = max(maximum_difference,
            float(np.max(np.abs(action - arrays['ensembled_public_action'][frame]))))
        count = int(arrays['active_prediction_count'][frame])
        if (count != len(active)
                or not np.array_equal(arrays['ensemble_query_indices'][frame, :count], active)
                or not np.array_equal(arrays['ensemble_chunk_rows'][frame, :count], rows)
                or not np.array_equal(arrays['ensemble_weights'][frame, :count], weights)
                or not np.array_equal(model_action_to_sim(action), arrays['target_native'][frame])):
            raise ValueError('Applied action or its recorded provenance differs from raw ACT predictions')
    if maximum_difference != 0.:
        raise ValueError('Recorded ensemble action differs from independent raw-chunk reconstruction')
    for query, inference in enumerate(result['inference']):
        if (inference['frame_index'] != int(starts[query])
                or digest(directory / f'observation_{query:03d}.npz') != inference['input_sha256']
                or digest(directory / f'chunk_{query:03d}.npz') != inference['output_sha256']):
            raise ValueError('Inference frame or artifact hash differs')
        with np.load(directory / f'chunk_{query:03d}.npz', allow_pickle=False) as raw:
            if not np.array_equal(raw['action'], chunks[query]):
                raise ValueError('Saved rollout chunk differs from the worker raw output')
        with np.load(directory / f'observation_{query:03d}.npz', allow_pickle=False) as observation:
            expected_state = (np.asarray(result['initial_public_state']) if starts[query] == 0
                              else arrays['public_state'][starts[query] - 1])
            if (set(observation.files) != {'state', 'wrist', 'top'}
                    or not np.array_equal(observation['state'], expected_state)
                    or not np.array_equal(observation['state'], arrays['chunk_observation_state'][query])):
                raise ValueError('ACT input state is not the actual pre-action simulator state')
    model = load_model(scene)
    data = mujoco.MjData(model)
    contacts = []
    for pose in arrays['qpos']:
        data.qpos[:] = pose
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        contacts.append(finger_contacts(model, data))
    if not np.array_equal(np.asarray(contacts), arrays['finger_contacts']):
        raise ValueError('Recorded finger contacts differ from qpos-based contact recomputation')
    bilateral = arrays['finger_contacts'].all(axis=1)
    lift = arrays['brick_position'][:, 2] - result['metrics']['initial_position_m'][2]
    elevated = bilateral & (lift >= .03)
    confirmed = longest_run(elevated) >= 3
    if result['metrics']['success'] and not confirmed:
        raise ValueError('Apparent success lacks sustained elevated bilateral finger contact')
    return {'all_applied_actions_exact_raw_chunk_weighted_averages': True,
        'maximum_ensemble_public_action_difference_degrees': maximum_difference,
        'all_applied_targets_exact_ensemble_conversion': True,
        'all_input_output_hashes_match_worker_records': True,
        'all_query_observations_match_actual_pre_action_states': True,
        'contact_trace_exact_qpos_recomputation': True,
        'left_finger_contact_frames': int(arrays['finger_contacts'][:, 0].sum()),
        'right_finger_contact_frames': int(arrays['finger_contacts'][:, 1].sum()),
        'bilateral_contact_frames': int(bilateral.sum()),
        'first_bilateral_frame': int(np.flatnonzero(bilateral)[0]) if bilateral.any() else None,
        'longest_elevated_bilateral_frames': longest_run(elevated),
        'physical_grasp_confirmed': confirmed,
        'audit_new_physics_steps': 0, 'audit_new_ACT_calls': 0}


def evaluate(args):
    from joint_mapping import dataset_state_to_sim
    from run_checkpoint_comparison import SOURCE, create_scene
    from run_position_generalization import model_hashes, normalized_scene_tree
    from run_task import digest, save_json

    model_dir, output = args.model_dir.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError('Preserve earlier results; choose a fresh output directory')
    if args.teacher_protocol.resolve() != POSITION_PROTOCOL.resolve():
        raise ValueError('Only the predeclared validation protocol is allowed')
    teacher = json.loads(POSITION_PROTOCOL.read_text(encoding='utf-8'))
    points = validation_positions(teacher)
    frozen_path = SOURCE / 'protocol.json'
    frozen = json.loads(frozen_path.read_text(encoding='utf-8'))
    if (frozen['duration_s'], frozen['policy_hz'], frozen['executed_actions_per_chunk']) != (15, 30, 100):
        raise ValueError('Unexpected frozen comparison settings')
    if (digest(Path(frozen['camera_fit_path'])) != frozen['camera_fit_sha256']
            or digest(Path(frozen['successful_reference_scene'])) != frozen['successful_reference_scene_sha256']):
        raise ValueError('Camera or scene differs from the frozen comparison')
    model_before = model_hashes(model_dir)
    code = dict(frozen['code_files_sha256'])
    for name in ('simulation/run_checkpoint_comparison.py', 'simulation/evaluate_multisite_act.py',
                 'simulation/evaluate_act_ensemble.py', 'simulation/analyze_position_generalization.py',
                 'simulation/preview.py'):
        code[name] = digest(ROOT / name)
    sources = {str(path): digest(path) for path in (POSITION_PROTOCOL, frozen_path,
        Path(frozen['camera_fit_path']), Path(frozen['successful_reference_scene']), ENSEMBLER_SOURCE)}

    def check_frozen():
        if model_hashes(model_dir) != model_before:
            raise ValueError('Checkpoint or saved processors changed')
        for filename, expected in code.items():
            if digest(ROOT / filename) != expected:
                raise ValueError(f'Evaluation code changed: {filename}')
        for filename, expected in sources.items():
            if digest(Path(filename)) != expected:
                raise ValueError(f'Protocol, scene, or ensemble definition source changed: {filename}')

    check_frozen()
    output.mkdir(parents=True, exist_ok=False)
    plan = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'variant': 'sparse_temporal_ensemble', 'model_directory': str(model_dir),
        'model_files_sha256': model_before, 'positions': points, 'split': 'validation',
        'position_protocol': str(POSITION_PROTOCOL), 'fixed_initial_state_protocol': str(frozen_path),
        'source_files_sha256': sources, 'code_files_sha256': code, 'duration_s': 15,
        'action_hz': ACTION_HZ, 'vision_query_hz': ACTION_HZ / QUERY_EVERY,
        'query_every_action_frames': QUERY_EVERY, 'chunk_size': CHUNK_SIZE,
        'temporal_ensemble_coeff': COEFFICIENT,
        'weight_rule': 'At frame t use every chunk starting s with s<=t<s+100, row t-s; rank covering chunks oldest first; normalize exp(-0.01*i)',
        'official_every_frame_ensemble': False,
        'changed_execution_strategy': 'Replace 100-frame nonoverlapping execution with 10-frame observation queries and overlapping-chunk averaging',
        'teacher_arrays_read': False, 'teacher_actions_read': False, 'test_episode_contents_opened': 0,
        'object_truth_usage': 'Initial XY placement and physical scoring only; never ACT input',
        'training_steps_allowed': 0, 'hardware_commands_allowed': 0, 'cpu_budget': CPU_BUDGET}
    save_json(output / 'protocol.json', plan)
    report = {'status': 'running', 'protocol': plan, 'results': [],
        'training_steps': 0, 'hardware_commands_sent': 0, 'actual_ACT_queries': 0}
    save_json(output / 'evaluation.json', report)
    initial = dataset_state_to_sim(frozen['initial_public_joint_target_degrees'])
    fit = json.loads(Path(frozen['camera_fit_path']).read_text(encoding='utf-8'))
    normalized = normalized_scene_tree(ET.parse(frozen['successful_reference_scene']).getroot())
    try:
        for point in points:
            check_frozen()
            directory = output / 'trials' / point['id']
            scene, metadata = create_scene(directory, point, frozen, fit, normalized)
            result = run_sparse_ensemble(scene, metadata, directory / 'custom', model_dir, initial, frozen)
            report['actual_ACT_queries'] += result['actual_inference_queries']
            entry = {'id': point['id'], 'split': 'validation', 'offset_xy_m': point['offset_xy_m'],
                     'scene_sha256': digest(scene), 'result': result}
            report['results'].append(entry)
            save_json(output / 'evaluation.json', report)
            if result['status'] != 'completed' or result['actual_inference_queries'] != 45 or result['policy_actions_applied'] != 450:
                raise RuntimeError(f'Incomplete sparse-ensemble rollout: {result["error"]}')
            identity = json.loads((directory / 'custom/model_identity.json').read_text(encoding='utf-8'))
            if identity['model_identity']['files_sha256'] != model_before:
                raise ValueError('Worker loaded a different checkpoint or processors')
            entry['contacts_and_ensemble_audit'] = audit_sparse_rollout(scene, directory / 'custom', result)
            save_json(output / 'evaluation.json', report)
            print(json.dumps({'point': point['id'], 'success': result['metrics']['success'],
                'lift_mm': result['metrics']['maximum_lift_m'] * 1000,
                'bilateral_frames': entry['contacts_and_ensemble_audit']['bilateral_contact_frames'],
                'actual_ACT_queries': result['actual_inference_queries']}, ensure_ascii=False), flush=True)
        report['status'] = 'completed_validation_sparse_ACT_temporal_ensemble'
    except Exception as error:
        report.update({'status': 'evaluation_error', 'error': f'{type(error).__name__}: {error}'})
        raise
    finally:
        report['completed_utc'] = datetime.now(timezone.utc).isoformat()
        report['model_files_unchanged'] = model_hashes(model_dir) == model_before
        report['completed_trials'] = len(report['results'])
        report['successes'] = sum(bool(entry['result']['metrics']['success']) for entry in report['results'])
        save_json(output / 'evaluation.json', report)
    check_frozen()


if __name__ == '__main__':
    evaluate(parse_args())
