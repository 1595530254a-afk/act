"""Compare recorded validation runs; no inference, teacher arrays or mj_step."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cpu_budget import CPU_BUDGET

import mujoco
import numpy as np

from preview import load_model
from run_task import digest, save_json


def tcp_trace(scene, qpos):
    model = load_model(scene)
    data = mujoco.MjData(model)
    tcp = []
    for pose in qpos:
        data.qpos[:] = pose
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        tcp.append(data.site('tcp').xpos.copy())
    return np.asarray(tcp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir', required=True, type=Path, nargs='+')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier analysis')
    reference = ROOT / 'runs/mujoco_iteration_20261023/multisite_v1/checkpoint_comparison_v3/sim300/baseline'
    record = json.loads((reference / 'custom/result.json').read_text(encoding='utf-8'))
    if not record['metrics']['success']:
        raise ValueError('Reference is not successful')
    with np.load(reference / 'custom/rollout_arrays.npz', allow_pickle=False) as arrays:
        reference_tcp = tcp_trace(reference / 'scene/task_scene.xml', arrays['qpos'])
    reference_initial = np.asarray(record['metrics']['initial_position_m'])
    result = {'status': 'completed_recorded_validation_diagnostics', 'results': {},
        'tcp_site': 'Upstream tcp is a palm reference, not an exact fingertip pinch point.',
        'same_phase_reference_frame': 163,
        'same_phase_reference': str(reference),
        'same_phase_error_definition': 'Actual TCP at recorded frame163 minus (successful baseline TCP at frame163 + initial object XY offset). This is phase-specific compensation error, not a measured visual localization error.',
        'reference_trace_sha256': digest(reference / 'custom/rollout_arrays.npz'),
        'new_ACT_calls': 0, 'new_physics_steps': 0, 'teacher_arrays_read': False,
        'hardware_IO': 0, 'cpu_budget': CPU_BUDGET}
    initial_inputs = {}
    for evaluation_dir in args.evaluation_dir:
        report_path = evaluation_dir / 'evaluation.json'
        report = json.loads(report_path.read_text(encoding='utf-8'))
        if report['status'] != 'completed_frozen_multisite_ACT_evaluation' or report['protocol']['split'] != 'validation':
            raise ValueError('Expected completed validation-only evaluation')
        entries = []
        for entry in report['results']:
            directory = evaluation_dir / 'trials' / entry['id']
            with np.load(directory / 'custom/observation_000.npz', allow_pickle=False) as arrays:
                inputs = {key: arrays[key].copy() for key in arrays.files}
            comparison = None
            if entry['id'] not in initial_inputs:
                initial_inputs[entry['id']] = inputs
            else:
                baseline = initial_inputs[entry['id']]
                if not np.array_equal(inputs['state'], baseline['state']):
                    raise ValueError('Initial robot state differs')
                comparison = {role: {'max_absolute_rgb_difference': float(np.max(np.abs(inputs[role].astype(float) - baseline[role].astype(float)))),
                    'unequal_values': int(np.count_nonzero(inputs[role] != baseline[role]))} for role in ('wrist', 'top')}
            with np.load(directory / 'custom/rollout_arrays.npz', allow_pickle=False) as arrays:
                tcp = tcp_trace(directory / 'scene/task_scene.xml', arrays['qpos'])
            metrics, contacts = entry['result']['metrics'], entry['contacts']
            initial = np.asarray(metrics['initial_position_m'])
            shift = initial - reference_initial
            shift[2] = 0
            compensation_error = tcp[163] - reference_tcp[163] - shift
            stage = ('completed_pick_place' if metrics['success'] else
                'grasp_lifted_but_placement_failed' if contacts['physical_grasp_confirmed'] else
                'bilateral_contact_but_no_sustained_lift' if contacts['bilateral_contact_frames'] else
                'no_bilateral_grasp_contact')
            entries.append({'id': entry['id'], 'offset_xy_m': entry['offset_xy_m'], 'stage': stage,
                'success': metrics['success'], 'maximum_lift_mm': metrics['maximum_lift_m'] * 1000,
                'contacts': contacts, 'same_phase_tcp_compensation_error_xyz_mm': (compensation_error * 1000).tolist(),
                'same_phase_tcp_compensation_error_xy_norm_mm': float(np.linalg.norm(compensation_error[:2]) * 1000),
                'minimum_palm_tcp_to_initial_brick_center_xyz_mm': float(np.linalg.norm(tcp - initial, axis=1).min() * 1000),
                'initial_RGB_comparison_with_first_model': comparison,
                'scene_sha256': digest(directory / 'scene/task_scene.xml'),
                'trace_sha256': digest(directory / 'custom/rollout_arrays.npz')})
        result['results'][evaluation_dir.name] = {'source_evaluation_sha256': digest(report_path),
            'model_files_sha256': report['protocol']['model_files_sha256'],
            'successes': report['successes'], 'trials': report['completed_trials'], 'entries': entries}
    save_json(args.output, result)
    print(json.dumps({name: {'successes': value['successes'], 'trials': value['trials'],
        'points': [{key: entry[key] for key in ('id', 'stage', 'maximum_lift_mm', 'same_phase_tcp_compensation_error_xyz_mm')}
                   for entry in value['entries']]} for name, value in result['results'].items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
