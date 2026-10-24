"""Prepare the pinned official ALOHA ACT baseline without training or rollout.

Use the existing ACT Python for --migrate; use the isolated ALOHA Python for
--check-environment. Original checkpoint bytes and both B601 environments stay
unchanged. Migration delegates to LeRobot's installed official converter and
then verifies every network tensor and every normalization statistic on CPU.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cpu_budget import CPU_BUDGET, configure_torch

ORIGINAL = ROOT / 'models/act_aloha_sim_transfer_cube_human_original'
MIGRATION = ROOT / 'models/act_aloha_sim_transfer_cube_human_migrated'
AUDIT = ROOT / 'runs/aloha_baseline_20261024/preparation'
REVISION = 'ba73b2766f1371cdc133ca4efb97eb090d744625'
WEIGHT_HASH = '42772891cb6eba1e7bc36ad8e12c0fa0723c61f036fa235c725ce6026e6e81df'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def check_environment():
    import gym_aloha
    import gymnasium
    import mujoco
    import dm_control

    versions = {name: importlib.metadata.version(name) for name in (
        'gym-aloha', 'gymnasium', 'mujoco', 'dm-control', 'numpy', 'Pillow', 'psutil')}
    if (versions['gym-aloha'], versions['mujoco'], versions['dm-control']) != ('0.1.5', '3.2.7', '1.0.27'):
        raise ValueError('Isolated environment differs from the selected dependency pins')
    result = {'status': 'isolated_environment_imports_verified', 'python': sys.executable,
              'versions': versions,
              'registered_tasks': [name for name in gymnasium.registry if 'Aloha' in name],
              'gym_aloha_path': str(Path(gym_aloha.__file__).resolve()),
              'environment_reset_calls': 0, 'physics_steps': 0, 'ACT_calls': 0,
              'rendering_verified': False, 'cpu_budget': CPU_BUDGET}
    write_json(AUDIT / 'environment_inspection.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def prepare(migrate, verify_existing=False):
    # This preparation process never owns or initializes the shared training GPU.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['PYTHONUTF8'] = '1'
    configure_torch()
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    from lerobot.processor.migrate_policy_normalization import (
        clean_state_dict, extract_normalization_stats, remove_normalization_layers)

    model_path = ORIGINAL / 'model.safetensors'
    if model_path.stat().st_size != 206766560 or digest(model_path) != WEIGHT_HASH:
        raise ValueError('Original weight differs from the pinned official HF LFS SHA256')
    source = json.loads((ROOT / 'reports/act_sim_reference_20261024/sources.json').read_text(encoding='utf-8'))
    for item in source['downloaded_files']:
        if digest(ORIGINAL / item['file']) != item['sha256']:
            raise ValueError(f'Original official config changed: {item["file"]}')
    original_files = {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)}
                      for p in ORIGINAL.iterdir() if p.is_file()}
    with safe_open(model_path, framework='pt', device='cpu') as archive:
        shapes = {name: list(archive.get_slice(name).get_shape()) for name in archive.keys()}
    raw = load_file(model_path, device='cpu')
    stats = extract_normalization_stats(raw)
    expected_features = {'observation.images.top', 'observation.state', 'action'}
    if set(stats) != expected_features or any(set(value) != {'mean', 'std'} for value in stats.values()):
        raise ValueError('Unexpected original normalization features/statistics')
    for feature, values in stats.items():
        for name, value in values.items():
            if not torch.isfinite(value).all() or (name == 'std' and not (value > 0).all()):
                raise ValueError(f'Invalid original normalization: {feature}/{name}')
    result = {'status': 'original_checkpoint_static_inspection_verified',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'repo_id': 'lerobot/act_aloha_sim_transfer_cube_human', 'revision': REVISION,
        'original_directory': str(ORIGINAL), 'original_files': original_files,
        'tensor_keys_shapes': shapes, 'tensor_count': len(shapes),
        'normalization_stats': {feature: {name: {'shape': list(value.shape), 'values': value.tolist()}
            for name, value in values.items()} for feature, values in stats.items()},
        'model_inputs': {'state': [14], 'top_RGB': [3, 480, 640]}, 'action_chunk': [100, 14],
        'CPU_only': True, 'ACT_forward_calls': 0, 'physics_steps': 0, 'training_steps': 0,
        'hardware_io': False, 'cpu_budget': CPU_BUDGET}
    inspection_path = AUDIT / 'original_checkpoint_inspection.json'
    if verify_existing:
        prior = json.loads(inspection_path.read_text(encoding='utf-8'))
        if prior['original_files'] != original_files or prior['tensor_keys_shapes'] != shapes:
            raise ValueError('Original inspection changed before verifying converter outputs')
    else:
        write_json(inspection_path, result)
    if not (migrate or verify_existing):
        print(json.dumps({'status': result['status'], 'audit': str(AUDIT)}, ensure_ascii=False))
        return
    if MIGRATION.exists() and not verify_existing:
        raise FileExistsError(f'Preserve existing migration directory: {MIGRATION}')
    staging, converted = MIGRATION / 'converter_input', MIGRATION / 'pretrained_model'
    module = 'lerobot.processor.migrate_policy_normalization'
    converter_path = Path(sys.modules[module].__file__).resolve()
    converter_returncode = None
    if not verify_existing:
        staging.mkdir(parents=True, exist_ok=False)
        config = json.loads((ORIGINAL / 'config.json').read_text(encoding='utf-8'))
        config['device'], config['pretrained_backbone_weights'] = 'cpu', None
        write_json(staging / 'config.json', config)
        os.link(model_path, staging / 'model.safetensors')
        shutil.copyfile(ORIGINAL / 'train_config.json', staging / 'train_config.json')
        with (MIGRATION / 'converter.stdout.log').open('x', encoding='utf-8') as stdout:
            with (MIGRATION / 'converter.stderr.log').open('x', encoding='utf-8') as stderr:
                process = subprocess.run([sys.executable, '-B', '-m', module,
                    '--pretrained-path', str(staging), '--output-dir', str(converted)],
                    env=dict(os.environ), stdout=stdout, stderr=stderr, timeout=180)
        converter_returncode = process.returncode
    error_log = (MIGRATION / 'converter.stderr.log').read_text(encoding='utf-8', errors='replace')
    card_network_validation_unavailable = ('OfflineModeIsEnabled' in error_log
        and 'https://huggingface.co/api/validate-yaml' in error_log
        and 'card.validate()' in error_log)
    if converter_returncode and not card_network_validation_unavailable:
        raise RuntimeError(f'Official converter failed ({converter_returncode}); inspect {MIGRATION}')
    cleaned = clean_state_dict(remove_normalization_layers(raw))
    actual = load_file(converted / 'model.safetensors', device='cpu')
    if set(cleaned) != set(actual) or any(not torch.equal(cleaned[key], actual[key]) for key in cleaned):
        raise ValueError('Converted network tensors differ from original cleaned state dictionary')
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    converted_config = ACTConfig.from_pretrained(converted, local_files_only=True)
    if converted_config.device != 'cpu' or converted_config.pretrained_backbone_weights is not None:
        raise ValueError('Migration must remain CPU-only and offline')
    policy = ACTPolicy(converted_config)
    policy.load_state_dict(actual, strict=True)
    pre, post = make_pre_post_processors(converted_config, pretrained_path=str(converted),
        preprocessor_overrides={'device_processor': {'device': 'cpu'}})
    stats_checked = {}
    for role, pipeline in [('preprocessor', pre), ('postprocessor', post)]:
        found = [step for step in pipeline.steps if hasattr(step, '_tensor_stats')]
        if len(found) != 1:
            raise ValueError(f'Expected one normalization step in {role}')
        restored = found[0]._tensor_stats
        for feature in (expected_features if role == 'preprocessor' else {'action'}):
            for name, expected in stats[feature].items():
                if feature not in restored or name not in restored[feature] or not torch.equal(
                        restored[feature][name].cpu().reshape(-1), expected.cpu().reshape(-1)):
                    raise ValueError(f'Normalization changed: {role}/{feature}/{name}')
        stats_checked[role] = sorted(restored)
    if any(digest(ORIGINAL / name) != info['sha256'] for name, info in original_files.items()):
        raise ValueError('Original checkpoint bytes changed during migration')
    migration = {'status': 'verified_CPU_official_processor_migration_no_inference',
        'source_revision': REVISION, 'original_weight_sha256': WEIGHT_HASH,
        'converter': {'module': module, 'source': str(converter_path), 'sha256': digest(converter_path),
                      'lerobot_version': importlib.metadata.version('lerobot')},
        'loader_only_config_overrides': {'device': 'cpu', 'pretrained_backbone_weights': None},
        'converter_returncode_this_invocation': converter_returncode,
        'converter_model_card_network_validation_unavailable': card_network_validation_unavailable,
        'converter_stderr_sha256': digest(MIGRATION / 'converter.stderr.log'),
        'converter_scope_note': 'Weights and processors saved before an offline-only model-card YAML validation error; outputs independently checked below. Installed converter was not modified.',
        'original_config_preserved': True, 'original_all_files_unchanged': True,
        'network_tensors_exactly_equal': True, 'network_tensor_count': len(actual),
        'strict_model_state_load_passed': True, 'processor_stats_exactly_equal': stats_checked,
        'converted_directory': str(converted),
        'converted_files': {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)}
                            for p in converted.iterdir() if p.is_file()},
        'CUDA_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
        'ACT_forward_calls': 0, 'physics_steps': 0, 'training_steps': 0, 'hardware_io': False}
    write_json(MIGRATION / 'migration_verification.json', migration)
    print(json.dumps({'status': migration['status'], 'converted_directory': str(converted),
                      'verification': str(MIGRATION / 'migration_verification.json')}, ensure_ascii=False), flush=True)


def smoke():
    """Five declared seeds, original 100-action chunks and official environment reward."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONUTF8'] = '1'
    import gym_aloha
    import gym_aloha.env as env_module
    import gymnasium as gym
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    output = ROOT / 'runs/aloha_baseline_20261024/smoke_5seeds'
    output.mkdir(parents=True, exist_ok=False)
    converted = MIGRATION / 'pretrained_model'
    migration = json.loads((MIGRATION / 'migration_verification.json').read_text(encoding='utf-8'))
    if not (migration['network_tensors_exactly_equal'] and migration['strict_model_state_load_passed']
            and migration['original_all_files_unchanged']):
        raise ValueError('CPU migration verification must pass before any rollout')
    frozen = {str(converted / name): info['sha256'] for name, info in migration['converted_files'].items()}
    frozen[str(ORIGINAL / 'model.safetensors')] = WEIGHT_HASH
    assets = Path(env_module.ASSETS_DIR).resolve()
    source_files = [Path(__file__), ROOT / 'simulation/aloha_policy_worker_20261024.py',
                    Path(env_module.__file__), Path(gym_aloha.__file__),
                    MIGRATION / 'migration_verification.json']
    source_files.extend(path for path in assets.rglob('*') if path.is_file())
    frozen.update({str(path.resolve()): digest(path) for path in source_files})

    def check_frozen():
        if any(digest(path) != expected for path, expected in frozen.items()):
            raise ValueError('Pinned model, normalization, source, or scene asset changed')

    check_frozen()
    versions = {distribution.metadata['Name']: distribution.version
                for distribution in importlib.metadata.distributions()}
    seeds = list(range(20261024, 20261029))
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'seeds': seeds, 'checkpoint_revision': REVISION, 'model_directory': str(converted),
        'frozen_files_sha256': frozen, 'environment': 'gym_aloha/AlohaTransferCube-v0',
        'environment_versions': dict(sorted(versions.items())),
        'max_episode_steps': 400, 'control_hz': 50, 'actions_per_chunk': 100,
        'policy_input_fields': ['state14', 'top_RGB_480x640x3'],
        'environment_state_or_object_truth_passed_to_policy': False,
        'ACT_device': 'cpu', 'video_fps': 50, 'CUDA_VISIBLE_DEVICES': '',
        'windows_path_compatibility': {'actual_assets_directory': str(assets),
            'method': "Temporarily chdir to existing assets and set env_module.ASSETS_DIR=Path('.') only while constructing official environment, then restore both.",
            'installed_source_or_xml_modified': False},
        'termination': 'Stop on official terminated/truncated or 400 steps; official success is reward==4 / info.is_success.',
        'scope': 'Five fixed-seed CPU smoke episodes; not a replication of publisher 500-episode 83% metric and not B601.',
        'training_steps_allowed': 0, 'hardware_commands_allowed': 0, 'cpu_budget': CPU_BUDGET}
    write_json(output / 'protocol.json', protocol)
    report = {'status': 'running', 'protocol': protocol, 'results': [], 'actual_ACT_forward_calls': 0,
              'training_steps': 0, 'hardware_io': False}
    # This report is task-owned progress state, never a source evaluation artifact.
    def save_report():
        (output / 'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2,
            allow_nan=False) + '\n', encoding='utf-8')
    save_report()
    ffmpeg, ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        raise RuntimeError('ffmpeg and ffprobe are required for the smoke evidence')
    log = (output / 'ACT_worker.stderr.log').open('x', encoding='utf-8')
    worker = subprocess.Popen([str(ROOT / '.venv/Scripts/python.exe'), '-B', '-u',
        str(ROOT / 'simulation/aloha_policy_worker_20261024.py'), '--model-dir', str(converted),
        '--device', 'cpu'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
        text=True, encoding='utf-8', env=dict(os.environ), bufsize=1)
    messages = queue.Queue()
    def receive():
        for line in worker.stdout:
            messages.put(line)
        messages.put(None)
    threading.Thread(target=receive, daemon=True).start()
    def read_worker():
        line = messages.get(timeout=90)
        if line is None:
            raise RuntimeError('ALOHA CPU worker exited; inspect ACT_worker.stderr.log')
        return json.loads(line)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 19)
    try:
        identity = read_worker()
        if identity.get('status') != 'ready' or identity['model_identity']['device'] != 'cpu':
            raise RuntimeError(f'Unexpected worker startup: {identity}')
        write_json(output / 'model_identity.json', identity)
        for seed in seeds:
            check_frozen()
            directory = output / f'seed_{seed}'
            directory.mkdir(exist_ok=False)
            env, encoder = None, None
            result = {'seed': seed, 'status': 'running', 'inference': [], 'success': False}
            arrays = {name: [] for name in ('state_before', 'state_after', 'action', 'qpos_before',
                'qpos_after', 'physics_time', 'reward', 'is_success', 'terminated', 'truncated')}
            chunks = []
            video = ROOT / '视频' / f'ACT_ALOHA_官方基线_seed{seed}_20261024.mp4'
            pending = video.with_name(video.stem + '.pending.mp4')
            if video.exists() or pending.exists():
                raise FileExistsError(f'Preserve previous evidence video: {video}')
            try:
                old_cwd, old_assets = Path.cwd(), env_module.ASSETS_DIR
                try:
                    os.chdir(assets)
                    env_module.ASSETS_DIR = Path('.')
                    env = gym.make(protocol['environment'], obs_type='pixels_agent_pos',
                        render_mode='rgb_array', max_episode_steps=400)
                finally:
                    env_module.ASSETS_DIR = old_assets
                    os.chdir(old_cwd)
                observation, info = env.reset(seed=seed)
                physics = env.unwrapped._env.physics
                start_time = float(physics.data.time)
                if abs(env.unwrapped._env.control_timestep() - .02) > 1e-12:
                    raise ValueError('Official control timestep differs from 50 Hz')
                Image.fromarray(observation['pixels']['top']).save(directory / 'initial_top.png')
                encoder = subprocess.Popen([ffmpeg, '-hide_banner', '-loglevel', 'error', '-n',
                    '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '640x560', '-r', '50', '-i', 'pipe:0',
                    '-an', '-c:v', 'libx264', '-threads', '2', '-preset', 'veryfast', '-crf', '20',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(pending)],
                    stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                for index in range(400):
                    before = np.asarray(observation['agent_pos'], dtype=np.float32)
                    if index % 100 == 0:
                        request_id = index // 100
                        input_path = directory / f'observation_{request_id:03d}.npz'
                        output_path = directory / f'chunk_{request_id:03d}.npz'
                        np.savez(input_path, state=before, top=observation['pixels']['top'])
                        worker.stdin.write(json.dumps({'input': str(input_path), 'output': str(output_path)}) + '\n')
                        worker.stdin.flush()
                        response = read_worker()
                        if (response.get('status') != 'ok' or response['input_sha256'] != digest(input_path)
                                or response['output_sha256'] != digest(output_path)):
                            raise RuntimeError(f'CPU ACT inference failed: {response}')
                        with np.load(output_path, allow_pickle=False) as archive:
                            chunk = archive['action'].copy()
                        if chunk.shape != (100, 14) or not np.isfinite(chunk).all():
                            raise ValueError('Expected a fresh finite ACT output[100,14]')
                        chunks.append(chunk)
                        result['inference'].append(response)
                        report['actual_ACT_forward_calls'] += 1
                    action = chunk[index % 100].copy()
                    arrays['state_before'].append(before.copy())
                    arrays['qpos_before'].append(physics.data.qpos.copy())
                    # Native official action only. No clipping, IK, object coordinates, or teacher path.
                    observation, reward, terminated, truncated, info = env.step(action)
                    elapsed = float(physics.data.time) - start_time
                    if abs(elapsed - (index + 1) / 50) > 1e-8:
                        raise ValueError('Official physics timeline differs from recorded control steps')
                    arrays['state_after'].append(np.asarray(observation['agent_pos']).copy())
                    arrays['action'].append(action)
                    arrays['qpos_after'].append(physics.data.qpos.copy())
                    arrays['physics_time'].append(elapsed)
                    arrays['reward'].append(float(reward))
                    arrays['is_success'].append(bool(info.get('is_success', False)))
                    arrays['terminated'].append(bool(terminated))
                    arrays['truncated'].append(bool(truncated))
                    canvas = Image.new('RGB', (640, 560), '#101820')
                    canvas.paste(Image.fromarray(observation['pixels']['top']), (0, 0))
                    draw = ImageDraw.Draw(canvas)
                    draw.text((12, 484), f'官方 ALOHA ACT · seed {seed} · reward {float(reward):g}', font=font, fill='white')
                    draw.text((12, 510), f't={elapsed:.2f}s · 动作 {index + 1}/400 · 推理 {len(chunks)}/4', font=font, fill='#c9e8ff')
                    draw.text((12, 536), '视频50fps；ACT每100步（2秒）看图一次；CPU推理', font=font, fill='#ffcc72')
                    encoder.stdin.write(np.asarray(canvas).tobytes())
                    if index == 220:
                        canvas.save(directory / 'frame220.png')
                    if bool(info.get('is_success', False)):
                        result['success'] = True
                    if terminated or truncated:
                        break
                encoder.stdin.close()
                encoder.stdin = None
                _, error = encoder.communicate(timeout=60)
                if encoder.returncode:
                    raise RuntimeError(error.decode('utf-8', errors='replace'))
                pending.rename(video)
                probe = json.loads(subprocess.check_output([ffprobe, '-v', 'error',
                    '-select_streams', 'v:0', '-count_frames', '-show_entries',
                    'stream=width,height,r_frame_rate,nb_read_frames,duration', '-of', 'json', str(video)], text=True))
                stream = probe['streams'][0]
                if (stream['r_frame_rate'] != '50/1' or int(stream['nb_read_frames']) != len(arrays['action'])
                        or abs(float(stream['duration']) - len(arrays['action']) / 50) > 1e-5):
                    raise ValueError('Evidence video timeline differs from actual official episode')
                result.update({'status': 'completed', 'executed_actions': len(arrays['action']),
                    'simulated_duration_s': float(arrays['physics_time'][-1]),
                    'maximum_reward': max(arrays['reward']), 'episode_return': sum(arrays['reward']),
                    'actual_ACT_forward_calls': len(result['inference']),
                    'video': str(video), 'video_sha256': digest(video), 'ffprobe': probe,
                    'environment_state_or_object_truth_passed_to_policy': False,
                    'actions_clipped': False, 'teacher_actions_used': False})
            except Exception as error:
                result.update({'status': 'execution_error', 'error': f'{type(error).__name__}: {error}',
                               'traceback': traceback.format_exc()})
                raise
            finally:
                if encoder and encoder.poll() is None:
                    encoder.terminate()
                    encoder.wait(timeout=10)
                if env:
                    env.close()
                np.savez_compressed(directory / 'trajectory.npz',
                    **{name: np.asarray(values) for name, values in arrays.items()},
                    action_chunks=np.asarray(chunks))
                write_json(directory / 'result.json', result)
                report['results'].append(result)
                save_report()
            check_frozen()
            print(json.dumps({'seed': seed, 'success': result['success'],
                'maximum_reward': result['maximum_reward'], 'steps': result['executed_actions'],
                'ACT_calls': result['actual_ACT_forward_calls']}, ensure_ascii=False), flush=True)
        report['status'] = 'completed_five_seed_official_ALOHA_CPU_smoke'
    except Exception as error:
        report['status'] = 'execution_error'
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if worker.poll() is None:
            worker.stdin.write('{"stop":true}\n')
            worker.stdin.flush()
            worker.wait(timeout=15)
        worker.stdin.close()
        worker.stdout.close()
        log.close()
        report['completed_utc'] = datetime.now(timezone.utc).isoformat()
        report['successes'] = sum(result['status'] == 'completed' and result['success'] for result in report['results'])
        report['completed_episodes'] = sum(result['status'] == 'completed' for result in report['results'])
        report['frozen_files_unchanged'] = all(digest(path) == expected for path, expected in frozen.items())
        save_report()
    print(json.dumps({'status': report['status'], 'successes': report['successes'],
                      'episodes': report['completed_episodes'], 'report': str(output / 'evaluation.json')}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-environment', action='store_true')
    parser.add_argument('--migrate', action='store_true')
    parser.add_argument('--verify-existing-migration', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.smoke:
        smoke()
    elif args.check_environment:
        check_environment()
    else:
        prepare(args.migrate, args.verify_existing_migration)


if __name__ == '__main__':
    main()
