"""Independently inspect saved multisite experts without advancing physics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cpu_budget import CPU_BUDGET

import mujoco
import numpy as np

from preview import load_model
from task_dynamics import read_public_state


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-dir', required=True, type=Path)
    args = parser.parse_args()
    base = args.teacher_dir.resolve()
    summary = json.loads((base / 'manifest.json').read_text(encoding='utf-8'))
    protocol = json.loads((base / 'protocol.json').read_text(encoding='utf-8'))
    assert summary['protocol_sha256'] == digest(base / 'protocol.json')
    coords = [tuple(item['offset_xy_m']) for item in protocol['positions']]
    assert len(coords) == 40 and len(set(coords)) == 40
    assert {item['id'] for item in protocol['positions']} == {item['id'] for item in summary['episodes']}
    source = ROOT / 'runs/mujoco_iteration_20261023/synthetic_demo_flat40_v1/trajectory.npz'
    with np.load(source, allow_pickle=False) as arrays:
        original_before = arrays['qpos_before'].copy()
        original_after = arrays['qpos_after'].copy()
        original_action = arrays['action'].copy()
    records, initial = [], None
    for item in summary['episodes']:
        mp = base / item['manifest']
        manifest = json.loads(mp.read_text(encoding='utf-8'))
        assert digest(mp) == item['manifest_sha256'], item['id']
        scene = Path(manifest['scene']['path'])
        trajectory = mp.parent / manifest['trajectory']['file']
        assert digest(scene) == manifest['scene']['sha256']
        assert digest(trajectory) == manifest['trajectory']['sha256']
        model = load_model(scene)
        data = mujoco.MjData(model)
        with np.load(trajectory, allow_pickle=False) as arrays:
            state, action = arrays['state'].copy(), arrays['action'].copy()
            before, after = arrays['qpos_before'].copy(), arrays['qpos_after'].copy()
            times, positions = arrays['timestamp_before'].copy(), arrays['brick_position_after'].copy()
        assert state.shape == (403, 7) and action.shape == (403, 7)
        assert state.dtype == action.dtype == np.float32
        assert np.isfinite(before).all() and np.isfinite(after).all()
        assert np.array_equal(before[1:], after[:-1]), f'Trace continuity: {item["id"]}'
        assert np.max(np.abs(times - np.arange(403) / 30)) <= .00101
        assert np.array_equal(action[:, 6], original_action[:, 6]), f'Gripper labels: {item["id"]}'
        addresses = [model.joint(f'joint{i}').qposadr[0] for i in range(1, 8)]
        if initial is None:
            initial = before[0, addresses].copy()
        assert np.array_equal(initial, before[0, addresses]), f'Initial robot: {item["id"]}'
        brick_address = int(model.joint('target_brick_freejoint').qposadr[0])
        expected = np.array([.34, -.025]) + np.array(item['offset_xy_m'])
        xml_xy = np.fromstring(ET.parse(scene).getroot().find(".//body[@name='target_brick']").get('pos'), sep=' ')[:2]
        assert np.linalg.norm(xml_xy - expected) < 1e-12
        # Warmup physics can cause tiny natural planar motion, so only XML
        # placement is required to match exactly. This is not a pose overwrite.
        assert np.linalg.norm(before[0, brick_address:brick_address + 2] - expected) < 1e-5
        state_error = xyz_error = 0.
        bilateral, elevated, deep = [], [], []
        for frame in range(403):
            data.qpos[:] = before[frame]
            mujoco.mj_forward(model, data)
            state_error = max(state_error, float(np.max(np.abs(
                read_public_state(model, data).astype(np.float32) - state[frame]))))
            data.qpos[:] = after[frame]
            mujoco.mj_forward(model, data)
            xyz_error = max(xyz_error, float(np.max(np.abs(data.body('target_brick').xpos - positions[frame]))))
            fingers = set()
            for contact in data.contact[:data.ncon]:
                if contact.efc_address < 0 or contact.dist > contact.includemargin + 1e-9:
                    continue
                names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom)) or ''
                         for geom in (contact.geom1, contact.geom2)]
                if 'target_brick_geom' in names:
                    for side in ('left', 'right'):
                        if any(name.startswith(f'gripper_{side}_') for name in names):
                            fingers.add(side)
                if contact.dist < -.001:
                    deep.append({'frame': frame, 'names': names, 'distance_m': float(contact.dist)})
            if len(fingers) == 2:
                bilateral.append(frame)
                if positions[frame, 2] - before[0, brick_address + 2] >= .03:
                    elevated.append(frame)
        assert state_error == 0. and xyz_error == 0., f'Kinematics state: {item["id"]}, {state_error}, {xyz_error}'
        assert bilateral == manifest['bilateral_contact_frames']
        assert elevated == manifest['bilateral_frames_at_least_3cm_lift']
        assert len(deep) == item['penetration_pairs_over_1mm']
        if item['offset_xy_m'] == [0., 0.]:
            assert np.array_equal(before, original_before) and np.array_equal(after, original_after)
            assert np.array_equal(action, original_action)
        records.append({'id': item['id'], 'split': item['split'], 'subset': item.get('subset'),
                        'hashes_verified': True, 'state_error_public_deg': state_error,
                        'brick_xyz_error_m': xyz_error, 'continuous_qpos_trace': True,
                        'recomputed_bilateral_frames': len(bilateral),
                        'elevated_bilateral_frames': len(elevated), 'deep_contacts': deep})
        print(json.dumps({'verified': item['id'], 'bilateral': len(bilateral), 'deep_contacts': len(deep)}), flush=True)
    report = {'status': 'passed_independent_saved_trace_recomputation', 'episodes': records,
              'source_protocol_sha256': digest(base / 'protocol.json'),
              'source_manifest_sha256': digest(base / 'manifest.json'),
              'training_position_count': 25, 'validation_position_count': 4, 'test_position_count': 11,
              'whole_position_splits_disjoint': True, 'same_actual_initial_robot_state_all_episodes': True,
              'baseline_exactly_matches_original_teacher_physics': True,
              'state_image_timing': 'state and eventual images correspond to qpos_before; actions applied afterwards',
              'new_physics_steps': 0, 'ACT_queries': 0, 'training_steps': 0, 'hardware_io': False,
              'cpu_budget': CPU_BUDGET,
              'limitation': 'Contact depths checked at recorded 30 Hz states, not every 1 kHz integration step.'}
    (base / 'independent_trace_verification.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'episodes'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
