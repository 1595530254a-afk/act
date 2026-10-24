"""Compare ACT training stages under the frozen six-position simulator protocol.

Only checkpoint identity changes. Camera samples and the frozen display-camera
render sequence are retained, but no video is encoded. All three checkpoints
are rerun in this loop; old sim300 results remain a separate reference. No
teacher is read. Rasterization reproducibility is measured, not assumed.
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

import mujoco
import numpy as np
from PIL import Image

from joint_mapping import dataset_state_to_sim, model_action_to_sim, mapping_metadata
from preview import load_model, overview_camera
from run_position_generalization import model_hashes, normalized_scene_tree
from run_task import PolicyProcess, brick_metrics, check_physics, digest, save_json
from task_dynamics import initialize, read_public_state, ServoController
from task_scene import build_scene

SOURCE = ROOT / 'runs/mujoco_validation_20261023/position_generalization_20261023'
DEFAULT_OUTPUT = ROOT / 'runs/mujoco_iteration_20261023/multisite_v1/checkpoint_comparison'
MODELS = {
    'published': ROOT / 'models/act_rebot_cube',
    'desktop1000': ROOT / 'delivery/ACT_laptop_20261019_v2/models/act_controlled_step1000',
}


def now():
    return datetime.now(timezone.utc).isoformat()


def check_frozen(protocol, source_hashes):
    for filename, expected in source_hashes.items():
        if digest(SOURCE / filename) != expected:
            raise ValueError(f'Frozen source artifact changed: {filename}')
    for filename, expected in protocol['code_files_sha256'].items():
        if digest(ROOT / filename) != expected:
            raise ValueError(f'Frozen execution code changed: {filename}')
    if digest(Path(protocol['camera_fit_path'])) != protocol['camera_fit_sha256']:
        raise ValueError('Camera fit changed')
    if digest(Path(protocol['successful_reference_scene'])) != protocol['successful_reference_scene_sha256']:
        raise ValueError('Reference scene changed')


def create_scene(output, trial, protocol, fit, normalized_reference):
    scene, metadata = build_scene(output / 'scene', square_brick=True,
        square_edge_m=.040, flat_brick=True, wrist_camera=fit)
    tree = ET.parse(scene)
    position = np.asarray(protocol['reference_brick_center_m']).copy()
    position[:2] += trial['offset_xy_m']
    tree.find(".//body[@name='target_brick']").set('pos', ' '.join(f'{x:.9g}' for x in position))
    if normalized_scene_tree(tree.getroot()) != normalized_reference:
        raise ValueError('Scene differs beyond brick XY and mesh-directory spelling')
    tree.write(scene, encoding='utf-8', xml_declaration=True)
    metadata['brick']['centre_m'] = position.tolist()
    metadata['position_evaluation'] = trial
    metadata['placement_basis'] = 'Unchanged positions from previously frozen evaluation protocol'
    save_json(output / 'scene/scene_metadata.json', metadata)
    return scene, metadata


def run_without_video(scene, metadata, output, model_dir, initial, protocol):
    """Keep run_task dynamics/render order; omit only canvas/video encoding."""
    output.mkdir(parents=True, exist_ok=False)
    model = load_model(scene)
    data = mujoco.MjData(model)
    initialize(model, data, initial)
    controller = ServoController(model)
    for _ in range(500):
        controller.step(data, initial)
    initial_public = read_public_state(model, data)
    reference_initial = np.asarray(protocol['successful_reference_initial_public_state'])
    if np.max(np.abs(initial_public - reference_initial)) > 1e-7:
        raise ValueError('Initial settled robot pose changed')
    initial_position = data.body('target_brick').xpos.copy()
    renderer = mujoco.Renderer(model, height=480, width=640)
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    options.sitegroup[:] = 0
    overview = overview_camera()
    overview.lookat[:] = [.23, 0, .14]
    overview.azimuth, overview.elevation, overview.distance = 125, -40, 1.12
    saved = {name: [] for name in ('public_state', 'target_native', 'brick_position',
        'qpos', 'sim_time', 'action_chunks', 'chunk_observation_state')}
    inference, penetration = [], Counter()
    worker, status, error_text = None, 'completed', None
    lift_max = 0.
    start_wall, sim_start = time.perf_counter(), data.time
    try:
        worker = PolicyProcess(model_dir, output)
        save_json(output / 'model_identity.json', worker.identity)
        for index in range(round(protocol['duration_s'] * 30)):
            if index % 100 == 0:
                observations = {}
                for role, camera in [('wrist', 'wrist_rgb'), ('top', 'overhead_rgb')]:
                    options.geomgroup[4] = int(role != 'wrist')
                    renderer.update_scene(data, camera=camera, scene_option=options)
                    observations[role] = renderer.render().copy()
                    if index == 0:
                        Image.fromarray(observations[role]).save(output / f'initial_{role}.png')
                public = read_public_state(model, data)
                chunk, timing = worker.predict(public, observations['wrist'], observations['top'], len(inference))
                saved['action_chunks'].append(chunk)
                saved['chunk_observation_state'].append(public)
                inference.append(timing)
            target = model_action_to_sim(chunk[index % 100])
            end_time = sim_start + (index + 1) / 30
            while data.time < end_time - 1e-10:
                previous_time = data.time
                controller.step(data, target)
                check_physics(data, previous_time)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise FloatingPointError('Non-finite simulation state')
            mujoco.mj_forward(model, data)
            position = data.body('target_brick').xpos.copy()
            lift_max = max(lift_max, float(position[2] - initial_position[2]))
            saved['public_state'].append(read_public_state(model, data))
            saved['target_native'].append(target)
            saved['brick_position'].append(position)
            saved['qpos'].append(data.qpos.copy())
            saved['sim_time'].append(data.time - sim_start)
            for contact in data.contact[:data.ncon]:
                if contact.dist < -.001:
                    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or str(g)
                             for g in (contact.geom1, contact.geom2)]
                    penetration[' / '.join(names)] += 1
            if index % 2 == 0:
                # Renderer history can alter pixels by one grey level. Keep
                # all frozen render calls so policy inputs remain identical.
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
        arrays = {name: np.asarray(values) for name, values in saved.items()}
        np.savez_compressed(output / 'rollout_arrays.npz', **arrays)
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
            'policy_inputs': ['state', 'wrist', 'top'], 'object_truth_passed_to_policy': False,
            'cpu_budget': CPU_BUDGET, 'evaluation': {'policy_hz': 30, 'chunk_size': 100,
                'executed_actions_per_chunk': 100, 'replanning_period_s': 100 / 30,
                'physics_dt_s': float(model.opt.timestep)}}
        save_json(output / 'result.json', result)
    return result


def numerical_replication_diagnostic(reference_dir, candidate_dir):
    comparison = {}
    with np.load(reference_dir / 'rollout_arrays.npz', allow_pickle=False) as reference:
        with np.load(candidate_dir / 'rollout_arrays.npz', allow_pickle=False) as candidate:
            if set(reference.files) != set(candidate.files):
                raise ValueError('Replication arrays differ')
            for name in reference.files:
                difference = float(np.max(np.abs(reference[name] - candidate[name])))
                comparison[name] = {'shape': list(reference[name].shape),
                    'maximum_absolute_difference': difference,
                    'exactly_identical': bool(np.array_equal(reference[name], candidate[name]))}
    artifacts = []
    for index in range(5):
        for filename in (f'observation_{index:03d}.npz', f'chunk_{index:03d}.npz'):
            with np.load(reference_dir / filename, allow_pickle=False) as reference:
                with np.load(candidate_dir / filename, allow_pickle=False) as candidate:
                    if set(reference.files) != set(candidate.files):
                        raise ValueError(f'Inference artifact field mismatch: {filename}')
                    artifacts.append({'file': filename, 'arrays': {name: {
                        'shape': list(reference[name].shape),
                        'maximum_absolute_difference': float(np.max(np.abs(
                            reference[name].astype(float) - candidate[name].astype(float)))),
                        'unequal_values': int(np.count_nonzero(reference[name] != candidate[name]))}
                        for name in reference.files}})
    return {'reference': str(reference_dir), 'candidate': str(candidate_dir),
        'all_trace_arrays_exact': all(value['exactly_identical'] for value in comparison.values()),
        'trace_arrays': comparison, 'inference_artifacts': artifacts,
        'interpretation': 'Fresh all-checkpoint trials share conditions; old sim300 is reference only. Small rasterization differences can amplify during closed-loop execution.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((SOURCE / 'protocol.json').read_text(encoding='utf-8'))
    source = json.loads((SOURCE / 'generalization_report.json').read_text(encoding='utf-8'))
    source_hashes = {name: digest(SOURCE / name) for name in ('protocol.json', 'generalization_report.json')}
    check_frozen(protocol, source_hashes)
    if (protocol['duration_s'], protocol['policy_hz'], protocol['executed_actions_per_chunk']) != (15, 30, 100):
        raise ValueError('Unexpected protocol settings')
    if source['status'] != 'completed_frozen_position_evaluation' or len(source['results']) != 6:
        raise ValueError('Frozen sim300 results are incomplete')
    sim_model = Path(protocol['model_directory'])
    models = {'sim300': sim_model, **MODELS}
    hashes = {name: model_hashes(directory) for name, directory in models.items()}
    if hashes['sim300'] != protocol['model_files_sha256']:
        raise ValueError('Sim300 identity changed')
    experiment = {'id': 'ACT_three_checkpoint_same_six_positions_v1', 'created_utc': now(),
        'source_protocol': str(SOURCE / 'protocol.json'), 'source_files_sha256': source_hashes,
        'models': {name: {'directory': str(path), 'files_sha256': hashes[name]}
                   for name, path in models.items()},
        'script_sha256': digest(Path(__file__)), 'training_steps_allowed': 0,
        'hardware_commands_allowed': 0, 'only_changed_variable': 'checkpoint weights and saved processors',
        'same_scene_initial_state_policy_hz_chunk_duration': True,
        'sim300_full_six_positions': 'Fresh trials under the same loop as published and desktop1000; old report retained separately',
        'render_reproducibility': 'Full baseline trace and input/output arrays compared to old sim300; bitwise equality is measured, not assumed',
        'cpu_budget': CPU_BUDGET}
    save_json(output / 'protocol.json', experiment)
    report = {'status': 'running', 'started_utc': now(), 'protocol': experiment,
        'results': {}, 'frozen_sim300_reference_results': source['results'],
        'training_steps': 0, 'hardware_commands_sent': 0,
        'new_neural_forward_calls': 0, 'new_physical_trials': 0}
    save_json(output / 'comparison.json', report)
    initial = dataset_state_to_sim(protocol['initial_public_joint_target_degrees'])
    fit = json.loads(Path(protocol['camera_fit_path']).read_text(encoding='utf-8'))
    normalized_reference = normalized_scene_tree(ET.parse(protocol['successful_reference_scene']).getroot())
    try:
        for name, model_dir in models.items():
            report['results'][name] = []
            for trial in protocol['trials']:
                check_frozen(protocol, source_hashes)
                if model_hashes(model_dir) != hashes[name]:
                    raise ValueError('Model changed between trials')
                trial_dir = output / name / trial['id']
                scene, metadata = create_scene(trial_dir, trial, protocol, fit, normalized_reference)
                result = run_without_video(scene, metadata, trial_dir / 'custom', model_dir, initial, protocol)
                report['new_neural_forward_calls'] += result['actual_inference_queries']
                report['new_physical_trials'] += 1
                identity = json.loads((trial_dir / 'custom/model_identity.json').read_text(encoding='utf-8'))
                if identity['model_identity']['files_sha256'] != hashes[name]:
                    raise ValueError('ACT loaded unexpected model or processors')
                report['results'][name].append({'id': trial['id'], 'offset_xy_m': trial['offset_xy_m'],
                    'scene_sha256': digest(scene), 'result': result})
                if name == 'sim300' and trial['id'] == 'baseline' and result['status'] == 'completed':
                    report['old_baseline_replication_diagnostic'] = numerical_replication_diagnostic(
                        SOURCE / 'trials/baseline/custom', trial_dir / 'custom')
                save_json(output / 'comparison.json', report)
                print(json.dumps({'checkpoint': name, 'position': trial['id'], 'status': result['status'],
                    'success': result['metrics']['success'], 'lift_mm': result['metrics']['maximum_lift_m'] * 1000},
                    ensure_ascii=False), flush=True)
                if result['status'] != 'completed':
                    raise RuntimeError(f'Trial aborted: {result["error"]}')
        report['status'] = 'completed_same_conditions_checkpoint_comparison'
    except Exception as error:
        report['status'] = 'comparison_error'
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['completed_utc'] = now()
        report['all_models_unchanged'] = all(model_hashes(path) == hashes[name] for name, path in models.items())
        report['summary'] = {name: {'baseline_success': bool(entries[0]['result']['metrics']['success']),
            'new_position_successes': sum(bool(entry['result']['metrics']['success']) for entry in entries[1:]),
            'new_position_trials': len(entries) - 1,
            'completed_trials': sum(entry['result']['status'] == 'completed' for entry in entries)}
            for name, entries in report['results'].items() if entries}
        save_json(output / 'comparison.json', report)
    check_frozen(protocol, source_hashes)
    if not report['all_models_unchanged']:
        raise ValueError('A model changed during the comparison')
    print(json.dumps(report['summary'], ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
