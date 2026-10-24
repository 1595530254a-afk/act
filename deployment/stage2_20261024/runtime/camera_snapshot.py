"""Read-only RealSense enumeration or one paired RGB/state snapshot.

No motor SDK is imported. State must be freshly exported by the existing
controller, already converted using its verified real-hardware mapping.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
            'wrist_flex.pos', 'wrist_yaw.pos', 'wrist_roll.pos', 'gripper.pos']


def checked_state(path, now_ns):
    value = json.loads(path.read_text(encoding='utf-8'))
    if (value.get('measured') is not True or value.get('state_order') != CHANNELS
            or value.get('units') != 'public_degree' or not value.get('mapping_id')):
        raise ValueError('State requires measured=true, known mapping_id, ordered channels and public_degree units; never fill zeros')
    stamp = value.get('receipt_monotonic_ns')
    if type(stamp) is not int or not 0 <= now_ns - stamp <= 250_000_000:
        raise ValueError('State feedback is absent/future/stale (>250ms); request fresh measured feedback')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-once', action='store_true')
    parser.add_argument('--wrist-serial')
    parser.add_argument('--top-serial')
    parser.add_argument('--state-json', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    missing = [name for name in ('numpy', 'psutil', 'pyrealsense2') if importlib.util.find_spec(name) is None]
    if missing:
        print(json.dumps({'status': 'dependency_missing', 'missing': missing,
            'action': 'Use an existing camera Python environment or report the gap; do not modify the hardware venv/pip/driver',
            'hardware_commands_sent': 0, 'camera_opened': False}), flush=True)
        return 2
    from cpu_budget import CPU_BUDGET
    import numpy as np
    import pyrealsense2 as rs
    context = rs.context()
    devices = [{'model': device.get_info(rs.camera_info.name),
                'serial': device.get_info(rs.camera_info.serial_number)} for device in context.query_devices()]
    if not args.capture_once:
        print(json.dumps({'status': 'read_only_camera_enumeration', 'devices': devices,
                          'hardware_commands_sent': 0, 'cpu_budget': CPU_BUDGET}), flush=True)
        return 0
    if (not args.wrist_serial or not args.top_serial or args.wrist_serial == args.top_serial
            or args.state_json is None or args.output_dir is None):
        parser.error('capture-once needs distinct actual serials, existing measured state-json and a new output-dir')
    if not {args.wrist_serial, args.top_serial}.issubset({device['serial'] for device in devices}):
        raise ValueError('Specified real camera serial is not connected')
    output = (ROOT / args.output_dir).resolve()
    if not output.is_relative_to(ROOT / 'local'):
        raise ValueError('Snapshot output must stay inside this package local directory')
    output.mkdir(parents=True, exist_ok=False)
    pipelines, images, capture = [], {}, {}
    try:
        for role, serial in (('wrist', args.wrist_serial), ('top', args.top_serial)):
            pipeline, config = rs.pipeline(), rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            pipeline.start(config)
            pipelines.append((role, serial, pipeline))
        for _ in range(10):
            for _, _, pipeline in pipelines:
                pipeline.wait_for_frames(3000)
        for role, serial, pipeline in pipelines:
            color = pipeline.wait_for_frames(3000).get_color_frame()
            if not color:
                raise ValueError(role + ' camera did not produce a color frame')
            stamp = time.monotonic_ns()
            images[role] = np.asanyarray(color.get_data()).copy()
            if images[role].shape != (480, 640, 3) or images[role].dtype != np.uint8:
                raise ValueError(role + ' did not match RGB640x480 uint8')
            capture[role] = {'serial': serial, 'receipt_monotonic_ns': stamp,
                'device_timestamp_ms': color.get_timestamp(),
                'timestamp_domain': str(color.get_frame_timestamp_domain()),
                'frame_number': color.get_frame_number(), 'configured_fps': 30,
                'sustained_fps_verified': False}
        state = checked_state((ROOT / args.state_json).resolve(), time.monotonic_ns())
        array = np.asarray(state['state'], dtype=np.float32)
        if array.shape != (7,) or not np.isfinite(array).all():
            raise ValueError('Measured state must contain seven finite numbers')
        capture['state'] = {key: state[key] for key in ('measured', 'mapping_id', 'state_order', 'units', 'receipt_monotonic_ns')}
        snapshot = output / 'observation.npz'
        with snapshot.open('xb') as stream:
            np.savez_compressed(stream, state=array, wrist=images['wrist'], top=images['top'])
        request = {'input': str(snapshot), 'frame_index': 0,
            'capture_monotonic_ns': max(item['receipt_monotonic_ns'] for item in capture.values()),
            'source': 'actual_read_only_camera_plus_existing_measured_state', 'capture': capture}
        (output / 'shadow_request.jsonl').write_text(json.dumps(request, ensure_ascii=False) + '\n', encoding='utf-8')
        receipts = [item['receipt_monotonic_ns'] for item in capture.values()]
        report = {'status': 'paired_snapshot_saved_no_motion', 'devices': devices,
            'receipt_timestamp_spread_ms': (max(receipts) - min(receipts)) / 1e6,
            'camera_clock_domain_note': 'Different device clocks are not assumed synchronized; local receipt timestamps recorded separately',
            'sustained_30hz_verified': False, 'hardware_commands_sent': 0,
            'request': 'shadow_request.jsonl', 'cpu_budget': CPU_BUDGET}
        (output / 'capture_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return 0
    finally:
        for _, _, pipeline in pipelines:
            pipeline.stop()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({'status': 'capture_failed', 'error': f'{type(error).__name__}: {error}',
                          'hardware_commands_sent': 0}), flush=True)
        raise SystemExit(1)
