"""Fetch only the pinned Seeed MuJoCo scene and its referenced assets.

Upstream XML and mesh bytes are preserved. Importing this module performs no I/O.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


REPOSITORY = 'Seeed-Projects/reBot_RS_Sim'
COMMIT = '438f7713d83fb8cd0d0cb05e8cc335264dd3997a'
ENTRY = 'rebotarm_mujoco_rs/models/rs_grasp_scene.xml'
ARM = 'rebotarm_mujoco_rs/models/rs_arm.xml'
ATTRIBUTION = 'rebotarm_mujoco_rs/models/THIRD_PARTY_MODELS.md'
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'local/mujoco_upstream'
MANIFEST = ROOT / 'simulation/upstream_manifest.json'
RAW_BASE = f'https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/'


def request_bytes(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'ACT-MuJoCo-resource-fetch'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=40) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code not in (408, 429, 500, 502, 503, 504):
                raise
            failure = error
        except (urllib.error.URLError, TimeoutError) as error:
            failure = error
        if attempt < 2:
            time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f'Could not fetch after three attempts: {url}') from failure


def repo_path(value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise ValueError(f'Invalid repository path: {value!r}')
    result = posixpath.normpath(value)
    if PurePosixPath(result).is_absolute() or result == '..' or result.startswith('../'):
        raise ValueError(f'Repository path escapes its root: {value!r}')
    return result


def cached_path(relative):
    relative = repo_path(relative)
    path = CACHE.joinpath(*PurePosixPath(relative).parts)
    if not path.resolve().is_relative_to(CACHE.resolve()):
        raise ValueError('Cache path escapes the assigned resource directory')
    return path


def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = path.parent if path.is_relative_to(CACHE) else CACHE / '_staging'
    staging_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='wb', dir=staging_dir,
            prefix=path.name + '.', suffix='.part', delete=False) as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
        staged = Path(stream.name)
    os.replace(staged, path)


def git_blob_sha(content):
    return hashlib.sha1(b'blob ' + str(len(content)).encode('ascii') + b'\0' + content).hexdigest()


def get_file(relative, tree):
    relative = repo_path(relative)
    entry = tree.get(relative)
    if entry is None or entry.get('type') != 'blob':
        raise FileNotFoundError(f'Referenced upstream file is absent at the pinned commit: {relative}')
    path = cached_path(relative)
    reused = False
    if path.is_file():
        content = path.read_bytes()
        reused = git_blob_sha(content) == entry['sha']
    if not reused:
        url = RAW_BASE + urllib.parse.quote(relative, safe='/')
        content = request_bytes(url)
        if len(content) != entry['size'] or git_blob_sha(content) != entry['sha']:
            raise ValueError(f'Download does not match the pinned Git blob: {relative}')
        atomic_write(path, content)
    if content.startswith(b'version https://git-lfs.github.com/spec/'):
        raise ValueError(f'Asset is an LFS pointer rather than mesh bytes: {relative}')
    record = {'path': relative, 'bytes': len(content),
        'sha256': hashlib.sha256(content).hexdigest(), 'git_blob_sha1': entry['sha'],
        'source_url': f'https://github.com/{REPOSITORY}/blob/{COMMIT}/{urllib.parse.quote(relative, safe="/")}',
        'download_url': RAW_BASE + urllib.parse.quote(relative, safe='/')}
    return content, record, reused


def verify_cached():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if manifest.get('repository') != REPOSITORY or manifest.get('commit') != COMMIT:
        raise ValueError('Resource manifest is not for the pinned upstream revision')
    for entry in manifest['files']:
        content = cached_path(entry['path']).read_bytes()
        if (len(content) != entry['bytes'] or hashlib.sha256(content).hexdigest() != entry['sha256']
                or git_blob_sha(content) != entry['git_blob_sha1']):
            raise ValueError(f'Cached resource validation failed: {entry["path"]}')
    return manifest


def fetch():
    response = json.loads(request_bytes(
        f'https://api.github.com/repos/{REPOSITORY}/git/trees/{COMMIT}?recursive=1'))
    if response.get('truncated'):
        raise RuntimeError('Upstream tree is truncated; do not guess missing resource paths')
    tree = {entry['path']: entry for entry in response['tree']}
    records, xml_roots, pending_assets = {}, {}, set()
    reused_count = 0

    def xml_file(relative, inherited=None):
        nonlocal reused_count
        relative = repo_path(relative)
        if relative in records:
            return
        content, record, reused = get_file(relative, tree)
        records[relative] = record
        reused_count += int(reused)
        root = ET.fromstring(content)
        xml_roots[relative] = root
        compiler = dict(inherited or {})
        for element in root.findall('compiler'):
            compiler.update(element.attrib)
        for include in root.iter('include'):
            xml_file(posixpath.join(posixpath.dirname(relative), include.attrib['file']), compiler)
        for element in root.iter():
            filename = element.get('file')
            if not filename or element.tag == 'include':
                continue
            if element.tag == 'mesh':
                asset_dir = compiler.get('meshdir', compiler.get('assetdir', ''))
            elif element.tag in ('texture', 'hfield'):
                asset_dir = compiler.get('texturedir', compiler.get('assetdir', ''))
            else:
                asset_dir = compiler.get('assetdir', '')
            # MuJoCo asset directories resolve relative to the main XML.
            pending_assets.add(repo_path(posixpath.join(posixpath.dirname(ENTRY), asset_dir, filename)))

    xml_file(ENTRY)
    if ARM not in records:
        raise ValueError('Pinned scene no longer includes the expected RS arm XML')
    pending_assets.add(ATTRIBUTION)
    with ThreadPoolExecutor(max_workers=3) as pool:
        for content, record, reused in pool.map(lambda path: get_file(path, tree), sorted(pending_assets)):
            records[record['path']] = record
            reused_count += int(reused)
    joints, cameras, actuators = [], [], []
    for relative, root in xml_roots.items():
        for element in root.iter():
            if element.tag in ('joint', 'freejoint') and element.get('name'):
                joints.append({'source': relative, **element.attrib, 'tag': element.tag})
            elif element.tag == 'camera':
                cameras.append({'source': relative, **element.attrib})
        for actuator in root.findall('./actuator/*'):
            actuators.append({'tag': actuator.tag, **actuator.attrib})
    mount = 'rebotarm_mujoco_rs/models/meshes/rebot_b601_rs_d405_mount_30deg.stl'
    if records[mount]['sha256'] != 'bf9ffe0eafc1c5f34afeac5485eced94ce3dc87977e3a995142e9cda287930d0':
        raise ValueError('D405 mount does not match the upstream attribution SHA256')
    license_names = ('LICENSE', 'LICENSE.MD', 'LICENSE.TXT', 'COPYING', 'NOTICE')
    manifest = {'repository': REPOSITORY, 'commit': COMMIT,
        'commit_url': f'https://github.com/{REPOSITORY}/commit/{COMMIT}',
        'fetched_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'scope': 'Pinned grasp scene, recursive MJCF includes, referenced mesh/texture assets and their attribution only',
        'entry_path': ENTRY, 'cache_root': str(CACHE), 'upstream_bytes_modified': False,
        'files': [records[path] for path in sorted(records)],
        'total_files': len(records), 'total_bytes': sum(record['bytes'] for record in records.values()),
        'licenses': {'repository_root_license_files': [path for path in tree if '/' not in path and path.upper() in license_names],
            'repository_wide_license_status': 'No root license file found in the pinned tree; do not infer a blanket asset license',
            'third_party_attribution_file': ATTRIBUTION,
            'd405_mount': {'path': mount, 'license': 'CERN-OHL-W-2.0',
                'attribution': 'Bowen Zhu, reBot B601-RS RealSense D405 Wrist Mount',
                'source_url': 'https://github.com/bowenszhu/rebot-b601-rs-d405-wrist-mount',
                'source_commit': 'ca379b9a4e19096730cd255dfc35ec419a7ad144'}},
        'xml_declared_joints': joints, 'xml_declared_cameras': cameras, 'xml_declared_actuators': actuators,
        'limitations': ['Joint/camera lists are parsed XML declarations, not a compiled MuJoCo validation.',
            'The upstream scene is not the authors real red-cube/grey-box training environment.',
            'No physical robot or CAN interface is used.']}
    atomic_write(MANIFEST, (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    verify_cached()
    print(json.dumps({'entry': str(cached_path(ENTRY)), 'commit': COMMIT,
        'files': manifest['total_files'], 'bytes': manifest['total_bytes'],
        'reused_verified_files': reused_count, 'manifest': str(MANIFEST)}, ensure_ascii=False))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true', help='Verify existing cached bytes without network access')
    args = parser.parse_args()
    if args.verify_only:
        manifest = verify_cached()
        print(json.dumps({'status': 'verified', 'files': manifest['total_files'], 'bytes': manifest['total_bytes']}))
    else:
        fetch()


if __name__ == '__main__':
    main()
