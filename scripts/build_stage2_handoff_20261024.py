"""Build a separate, portable stage2 observation-only ACT handoff."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cpu_budget import CPU_BUDGET

BUNDLE_ID = 'ACT_stage2_handoff_20261024_v1'
STAGING = ROOT / 'output' / BUNDLE_ID
TEMPLATE = ROOT / 'deployment/stage2_20261024'
SOURCE = ROOT / 'runs/mujoco_iteration_20261024/long5000_v1/checkpoints/step_003000/pretrained_model'
ROLLOUT = ROOT / 'runs/mujoco_iteration_20261024/final8_selected_v1/trials/newtest_00/custom'
EXPECTED_MODEL = 'c25ffa08fd2b67343f37eab06bb948642e2a6d2578e387a563ba81105c506c2c'
MODEL_FILES = ('config.json', 'model.safetensors', 'policy_preprocessor.json', 'policy_postprocessor.json',
              'policy_preprocessor_step_3_normalizer_processor.safetensors',
              'policy_postprocessor_step_0_unnormalizer_processor.safetensors')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def payload_files():
    return sorted(path for path in STAGING.rglob('*') if path.is_file()
                  and path.name != 'bundle_manifest.json' and path.suffix != '.pyc'
                  and 'local' not in path.relative_to(STAGING).parts
                  and '__pycache__' not in path.parts)


def copy(source, relative):
    destination = STAGING / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def prepare():
    if STAGING.exists():
        raise FileExistsError('Preserve existing package directory; use --archive-only to archive the prepared payload')
    source_hashes = {name: digest(SOURCE / name) for name in MODEL_FILES}
    if source_hashes['model.safetensors'] != EXPECTED_MODEL:
        raise ValueError('Source checkpoint differs from selected step3000')
    STAGING.mkdir(parents=True, exist_ok=False)
    for path in sorted(TEMPLATE.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
            copy(path, path.relative_to(TEMPLATE))
    model_relative = 'models/act_long5000_step3000'
    for name in MODEL_FILES:
        copy(SOURCE / name, f'{model_relative}/{name}')
    original = json.loads((ROLLOUT / 'result.json').read_text(encoding='utf-8'))
    if original['model_sha256'] != EXPECTED_MODEL or original['evaluation']['mode'] != 'ensemble1':
        raise ValueError('Fixture source is not the selected model ensemble1 rollout')
    fixtures = []
    for frame in (0, 170, 300):
        record = original['inference'][frame]
        observation, reference = ROLLOUT / f'observation_{frame:03d}.npz', ROLLOUT / f'chunk_{frame:03d}.npz'
        if (record['frame_index'] != frame or digest(observation) != record['input_sha256']
                or digest(reference) != record['output_sha256']):
            raise ValueError('Frozen fixture source changed')
        input_relative, reference_relative = f'fixtures/observation_{frame:03d}.npz', f'fixtures/reference_{frame:03d}_cuda.npz'
        copy(observation, input_relative)
        copy(reference, reference_relative)
        fixtures.append({'input': input_relative, 'reference': reference_relative,
            'source_frame_index': frame, 'source_input_sha256': record['input_sha256'],
            'source_reference_sha256': record['output_sha256'], 'reference_device': 'cuda',
            'limit': 'Three sparse observations check fresh model outputs; they are not a replay of the450-frame physical rollout'})
    training = json.loads((ROOT / 'runs/mujoco_iteration_20261024/long5000_v1/training_report.json').read_text(encoding='utf-8'))
    selected = next(item for item in training['checkpoints'] if item['step'] == 3000)
    origin = {'base_model': 'nikodembartnik/act_rebot_cube',
        'base_revision': '409b639946cc4140b3934a6e3d2c255f4335ee9d',
        'base_training_dataset': 'nikodembartnik/rebot_cube; publisher reports50 episodes',
        'local_real_data_updates': 1000, 'source_single_scene_sim_updates': 300,
        'new_continuous_branch_updates_completed': 5000,
        'updates_of_selected_checkpoint_within_new_branch': 3000,
        'selected_checkpoint_sha256': EXPECTED_MODEL,
        'batch_size': 8, 'learning_rate': 1e-5, 'precision': 'float32', 'seed': 20261024,
        'synthetic_train_positions': 25, 'validation_positions': 4,
        'selection_uses_test': False,
        'selection_metric': 'Original-position admission first, then physical success count and stable grasp; offline MAE only breaks ties',
        'selected_offline_validation': selected['temporal_diagnostic'],
        'saved_processor_files_copied_byte_for_byte': True,
        'source_training_report_sha256': digest(ROOT / 'runs/mujoco_iteration_20261024/long5000_v1/training_report.json'),
        'source_ensemble_math_sha256': digest(ROOT / 'simulation/evaluate_act_ensemble.py'),
        'source_feedback_mode_sha256': digest(ROOT / 'simulation/evaluate_feedback_20261024.py'),
        'real_robot_execution_verified': False, 'hardware_commands_sent': 0}
    save_json(STAGING / 'model_origin.json', origin)
    copy(ROOT / 'models/act_rebot_cube/README.md', 'licenses/original_model_card.md')
    copy(ROOT / 'delivery/ACT_laptop_20261019_v2/licenses/uv-LICENSE-APACHE.txt', 'licenses/Apache-2.0.txt')
    manifest = {'bundle_id': BUNDLE_ID, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'model_directory': model_relative, 'model_files_sha256': source_hashes,
        'selected_model_sha256': EXPECTED_MODEL, 'fixtures': fixtures,
        'hardware_execution_enabled': False, 'hardware_execution_qualified': False,
        'default_mode': 'offline', 'live_mode': 'shadow_dry_run_only',
        'old_model_driver_venv_stop_chain_overwritten': False,
        'environment_installation_or_driver_update_performed': False,
        'normal_exit_chain': 'Preserve already verified hold -> existing safe home path -> operator support confirmation -> disable',
        'wallclock_30hz_verified_on_laptop': False,
        'requirements': {'python': '3.12', 'torch': '2.10.0', 'lerobot': '0.6.1',
                         'numpy': 'existing ACT environment', 'psutil': 'existing ACT environment',
                         'pyrealsense2': 'optional, existing camera environment only'},
        'cpu_budget': CPU_BUDGET, 'files': []}
    refresh_manifest(manifest)
    if {name: digest(SOURCE / name) for name in MODEL_FILES} != source_hashes:
        raise ValueError('Source model changed while copying')
    return manifest


def refresh_manifest(manifest):
    manifest['files'] = [{'path': path.relative_to(STAGING).as_posix(),
        'sha256': digest(path), 'bytes': path.stat().st_size} for path in payload_files()]
    save_json(STAGING / 'bundle_manifest.json', manifest)


def archive(manifest, archive_name):
    target = ROOT / 'output' / archive_name
    if Path(archive_name).name != archive_name or not archive_name.endswith('.zip'):
        raise ValueError('archive-name must be a simple new.zip filename')
    if target.exists():
        raise FileExistsError('Never overwrite an earlier ZIP')
    refresh_manifest(manifest)
    expected = {f'{BUNDLE_ID}/{item["path"]}': item['sha256'] for item in manifest['files']}
    expected[f'{BUNDLE_ID}/bundle_manifest.json'] = digest(STAGING / 'bundle_manifest.json')
    with zipfile.ZipFile(target, 'x', zipfile.ZIP_DEFLATED, compresslevel=1) as output:
        for relative in expected:
            output.write(STAGING / Path(relative).relative_to(BUNDLE_ID), relative)
    with zipfile.ZipFile(target) as saved:
        if saved.testzip() is not None or set(saved.namelist()) != set(expected) or len(saved.namelist()) != len(expected):
            raise ValueError('ZIP CRC/member set verification failed')
        for member, sha in expected.items():
            with saved.open(member) as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != sha:
                    raise ValueError('ZIP member hash differs: ' + member)
    sha = digest(target)
    target.with_suffix('.zip.sha256').write_text(f'{sha}  {target.name}\n', encoding='ascii')
    result = {'status': 'built_and_all_zip_member_hashes_verified', 'archive': str(target),
        'archive_sha256': sha, 'archive_bytes': target.stat().st_size,
        'payload_files_including_manifest': len(expected), 'selected_model_sha256': EXPECTED_MODEL,
        'hardware_commands_sent': 0, 'old_packages_overwritten': False,
        'cpu_budget': CPU_BUDGET}
    save_json(target.with_suffix('.build.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--archive-only', action='store_true')
    parser.add_argument('--archive-name', default=BUNDLE_ID + '.zip')
    parser.add_argument('--guide', type=Path, help='Optional user-facing Word/PDF guide to include before creating a new archive')
    args = parser.parse_args()
    if args.prepare_only and args.archive_only:
        parser.error('prepare-only and archive-only are mutually exclusive')
    manifest = (json.loads((STAGING / 'bundle_manifest.json').read_text(encoding='utf-8'))
                if args.archive_only else prepare())
    if args.guide:
        if args.guide.suffix.lower() not in ('.docx', '.pdf') or not args.guide.is_file():
            raise ValueError('guide must be an existing.docx or.pdf file')
        copy(args.guide, Path('guide') / args.guide.name)
        refresh_manifest(manifest)
    if args.prepare_only:
        print(json.dumps({'status': 'prepared_no_archive', 'root': str(STAGING),
                          'hardware_commands_sent': 0}, ensure_ascii=False), flush=True)
    else:
        print(json.dumps(archive(manifest, args.archive_name), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
