"""Create credited stage1 display copies; never alter frozen source videos.

Default mode checks and prints the plan. --preview V05 writes one PNG to tmp.
Only --publish encodes the thirteen B601 videos, with an appended credit bar.
No ACT, simulation, camera or robot API is imported or called.
After moving this tool, --source-root locates the unchanged historical archive.
"""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cpu_budget import CPU_BUDGET

from PIL import Image, ImageDraw, ImageFont


SOURCE_MANIFEST_RELATIVE = Path('output/ACT第一阶段记录_20261024/media_manifest.json')
OUTPUT = ROOT / 'results/stage1_20261024/videos'
PREVIEW = ROOT / 'tmp/stage1_author_preview_20261024'
REPOSITORY = '1595530254a-afk/act'
BAR_HEIGHT = 64
NAMES = {
    'V01': 'v01_sim300.mp4',
    'V02': 'v02_shifted_positions.mp4',
    'V03': 'v03_old_step600_test6.mp4',
    'V04': 'v04_final8_overview.mp4',
    **{f'V{i + 5:02d}': f'v{i + 5:02d}_final8_test{i:02d}.mp4' for i in range(8)},
    'V13': 'v13_extra_p10_0.mp4',
}
FINAL8 = 'runs/mujoco_iteration_20261024/final8_selected_v1/evaluation.json'
EVALUATIONS = {
    'V01': ('runs/mujoco_validation_20261023/fixed_flat40_20261023/'
            'policy_step300_progress_fix/custom/result.json', ['baseline']),
    'V02': ('runs/mujoco_validation_20261023/position_generalization_20261023/'
            'generalization_report.json', None),
    'V03': ('runs/mujoco_iteration_20261023/multisite_v1/'
            'final_heldout_v1_step600/evaluation.json', None),
    'V04': (FINAL8, None),
    **{f'V{i + 5:02d}': (FINAL8, [f'newtest_{i:02d}']) for i in range(8)},
    'V13': ('runs/mujoco_iteration_20261024/additional_point_p10_0_v1/'
            'extra_p10_0_rollout/evaluation.json', ['extra_p10_0']),
}


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def run(command):
    flags = 0
    if os.name == 'nt':
        flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
    return subprocess.run(command, check=True, capture_output=True, text=True,
                          encoding='utf-8', errors='replace', creationflags=flags)


def probe(path, ffprobe):
    value = json.loads(run([ffprobe, '-v', 'error', '-select_streams', 'v:0',
        '-count_frames', '-show_entries',
        'stream=width,height,r_frame_rate,avg_frame_rate,duration,nb_frames,nb_read_frames',
        '-of', 'json', str(path)]).stdout)['streams'][0]
    value['frames'] = int(value.get('nb_read_frames') or value['nb_frames'])
    return value


def evaluation_identity(video_id, source_root):
    relative, selected = EVALUATIONS[video_id]
    path = source_root / relative
    value = read_json(path)
    ids = [row['id'] for row in value.get('results', [])]
    if selected is not None and ids and not set(selected).issubset(ids):
        raise ValueError(f'Evaluation trial mismatch: {video_id}')
    return {'path': relative, 'sha256': digest(path),
            'run_id': path.parent.name if ids else path.parent.parent.name,
            'trial_ids': ids if selected is None else selected,
            'source_status': value['status']}


def prepare(manifest_path, ffprobe, source_root):
    entries = read_json(manifest_path)['entries']
    selected = [entry for entry in entries if entry['id'] in NAMES]
    if [entry['id'] for entry in selected] != list(NAMES):
        raise ValueError('Expected exactly V01 through V13 in original order')
    result = []
    for entry in selected:
        source = (source_root / entry['source_relative_path']).resolve()
        if not source.is_relative_to(source_root) or source.suffix.lower() != '.mp4':
            raise ValueError('Source must stay inside the historical archive root')
        if digest(source) != entry['sha256'] or source.stat().st_size != entry['bytes']:
            raise ValueError(f'Original video identity changed: {entry["id"]}')
        result.append({'id': entry['id'], 'title': entry['title'],
            'source_relative_path': source.relative_to(source_root).as_posix(),
            'source_sha256': entry['sha256'], 'source_bytes': entry['bytes'],
            'source_probe': probe(source, ffprobe),
            'output_filename': NAMES[entry['id']],
            'evaluation': evaluation_identity(entry['id'], source_root)})
    return result


def credit_bar(author_file, video_id, width, destination):
    if destination.exists():
        raise FileExistsError(destination)
    author = author_file.read_text(encoding='utf-8')
    font_path = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts/msyh.ttc'
    if not font_path.is_file():
        raise FileNotFoundError('Microsoft YaHei is required for the Chinese credit')
    image = Image.new('RGB', (width, BAR_HEIGHT), '#111a24')
    draw = ImageDraw.Draw(image)
    line = f'仿真复现作者：{author}    GitHub：{REPOSITORY}    文件：{video_id}'
    size = min(22, max(14, width // 64))
    while size >= 10:
        font = ImageFont.truetype(str(font_path), size)
        if draw.textbbox((0, 0), line, font=font)[2] <= width - 24:
            break
        size -= 1
    else:
        raise ValueError('Author text is too long to remain legible')
    draw.text((12, 6), line, font=font, fill='#f2f5fa')
    small = ImageFont.truetype(str(font_path), min(16, size))
    draw.text((12, 36), '展示层署名副本 · 原始策略观察文件未修改',
              font=small, fill='#b5c9df')
    image.save(destination)


def encode_command(ffmpeg, source, bar, entry, destination, preview=False):
    fps = entry['source_probe']['r_frame_rate']
    filters = '[0:v][1:v]vstack=inputs=2:shortest=1[v]'
    if preview:
        frame = min(221, entry['source_probe']['frames'] - 1)
        filters = (f'[0:v]select=eq(n\\,{frame}),setpts=PTS-STARTPTS[frame];'
                   '[frame][1:v]vstack=inputs=2:shortest=1[v]')
    command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-n', '-threads', '2',
        '-i', str(source), '-loop', '1', '-framerate', fps, '-i', str(bar),
        '-filter_complex_threads', '2', '-filter_complex', filters, '-map', '[v]', '-an']
    if preview:
        return [*command, '-frames:v', '1', '-threads', '2', str(destination)]
    return [*command, '-frames:v', str(entry['source_probe']['frames']),
        '-fps_mode', 'passthrough', '-c:v', 'libx264', '-preset', 'veryfast',
        '-crf', '20', '-pix_fmt', 'yuv420p', '-threads', '2',
        '-movflags', '+faststart', str(destination)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--author', default='1595530254a-afk')
    parser.add_argument('--source-root', type=Path, default=ROOT,
                        help='Unchanged original archive; outputs stay beside this tool')
    parser.add_argument('--manifest', type=Path,
                        help='Defaults to the old report media_manifest inside source-root')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preview', choices=list(NAMES))
    mode.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    if not args.author.strip() or args.author != args.author.strip() or any(
            ord(c) < 32 for c in args.author):
        parser.error('Author must be a nonempty literal single-line name')
    source_root = args.source_root.resolve()
    args.manifest = (args.manifest or source_root / SOURCE_MANIFEST_RELATIVE).resolve()
    if not args.manifest.is_relative_to(source_root):
        parser.error('The source manifest must stay inside source-root')
    ffmpeg, ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        raise RuntimeError('Existing ffmpeg and ffprobe are required')
    before = digest(args.manifest)
    entries = prepare(args.manifest, ffprobe, source_root)
    plan = {'author': args.author, 'github': REPOSITORY, 'entries': entries,
            'source_manifest_sha256': before, 'cpu_budget': CPU_BUDGET}
    if args.preview:
        PREVIEW.mkdir(parents=True, exist_ok=True)
        tag = hashlib.sha256(args.author.encode('utf-8')).hexdigest()[:8]
        author_file = PREVIEW / f'{tag}_author_utf8.txt'
        if author_file.exists():
            if author_file.read_text(encoding='utf-8') != args.author:
                raise ValueError('Existing preview author text differs')
        else:
            author_file.write_text(args.author, encoding='utf-8')
        entry = next(row for row in entries if row['id'] == args.preview)
        bar = PREVIEW / f'{tag}_{args.preview.lower()}_bar.png'
        destination = PREVIEW / f'{tag}_{args.preview.lower()}_preview.png'
        if destination.exists():
            raise FileExistsError(destination)
        credit_bar(author_file, args.preview, entry['source_probe']['width'], bar)
        run(encode_command(ffmpeg, source_root / entry['source_relative_path'], bar,
                           entry, destination, preview=True))
        print(json.dumps({'preview': str(destination), 'author_file': str(author_file),
                          'author_literal': args.author}, ensure_ascii=False))
        return
    if not args.publish:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if OUTPUT.exists():
        raise FileExistsError('Preserve existing published outputs; refusing to overwrite')
    OUTPUT.mkdir(parents=True, exist_ok=False)
    assets = OUTPUT / 'watermark_assets'
    assets.mkdir()
    author_file = OUTPUT / 'author_utf8.txt'
    author_file.write_text(args.author, encoding='utf-8')
    report = {'status': 'encoding', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'author': args.author, 'github_repository': REPOSITORY,
        'author_utf8_file': 'author_utf8.txt', 'author_utf8_sha256': digest(author_file),
        'source_manifest_relative': args.manifest.relative_to(source_root).as_posix(),
        'source_path_base': 'Historical original archive supplied by --source-root, not credited outputs',
        'source_manifest_sha256': before, 'included_ids': list(NAMES), 'excluded_ids': ['V14'],
        'display_copy_only': True, 'bar_height_px': BAR_HEIGHT,
        'pixel_identity_note': 'Re-encoded display copies append a credit bar. Original video bytes '
            'and original policy-input files are untouched; display pixels are not byte-identical evidence.',
        'ffmpeg_threads': 2, 'cpu_budget': CPU_BUDGET, 'entries': []}
    manifest = OUTPUT / 'video_manifest.json'
    for entry in entries:
        source = source_root / entry['source_relative_path']
        bar = assets / (entry['id'].lower() + '_bar.png')
        destination = OUTPUT / entry['output_filename']
        credit_bar(author_file, entry['id'], entry['source_probe']['width'], bar)
        print(f'Encoding credited display copy {entry["id"]}/V13', flush=True)
        run(encode_command(ffmpeg, source, bar, entry, destination))
        actual = probe(destination, ffprobe)
        expected = entry['source_probe']
        if (actual['width'] != expected['width'] or actual['height'] != expected['height'] + BAR_HEIGHT
                or actual['frames'] != expected['frames']
                or Fraction(actual['r_frame_rate']) != Fraction(expected['r_frame_rate'])
                or abs(float(actual['duration']) - float(expected['duration'])) > .002):
            raise ValueError(f'Credited video timing/geometry mismatch: {entry["id"]}')
        if digest(source) != entry['source_sha256']:
            raise ValueError('Frozen original changed while encoding')
        report['entries'].append({**entry, 'output_sha256': digest(destination),
            'output_bytes': destination.stat().st_size, 'output_probe': actual,
            'credit_bar_relative': bar.relative_to(OUTPUT).as_posix(),
            'credit_bar_sha256': digest(bar)})
        manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if digest(args.manifest) != before or any(
            digest(source_root / row['source_relative_path']) != row['source_sha256'] for row in entries):
        raise ValueError('Original manifest or videos changed during publication')
    report.update({'status': 'verified_credited_stage1_display_copies',
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'original_videos_unchanged': True, 'original_policy_observation_files_modified': False,
        'new_ACT_calls': 0, 'new_physics_steps': 0})
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'manifest': str(manifest), 'videos': len(entries),
                      'author_literal': args.author}, ensure_ascii=False))


if __name__ == '__main__':
    main()
