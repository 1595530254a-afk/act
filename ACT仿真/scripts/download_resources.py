"""Download the pinned dataset and final policy, then verify every file."""

import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest_path = ROOT / 'resources/resource_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    api = HfApi()
    results = {}

    for kind, folder in (('model', 'models/act_rebot_cube'), ('dataset', 'data/rebot_cube')):
        source = manifest[kind]
        info_method = api.model_info if kind == 'model' else api.dataset_info
        info = info_method(source['repo_id'], revision=source['revision'], files_metadata=True)
        if info.sha != source['revision']:
            raise RuntimeError(f'Unexpected {kind} revision: {info.sha}')
        files = [
            item for item in info.siblings
            if item.rfilename != '.gitattributes'
            and (kind == 'dataset' or '/' not in item.rfilename)
        ]
        destination = ROOT / folder
        print(f'Downloading {kind}: {source["repo_id"]} at {info.sha}', flush=True)
        snapshot_download(
            repo_id=source['repo_id'],
            repo_type=kind,
            revision=info.sha,
            local_dir=destination,
            allow_patterns=[item.rfilename for item in files],
            max_workers=3,
        )
        verified = []
        for item in files:
            path = destination / item.rfilename
            size = path.stat().st_size
            if item.size is not None and size != item.size:
                raise RuntimeError(f'File size mismatch: {path}')
            digest = hashlib.sha256()
            git_digest = hashlib.sha1(f'blob {size}\0'.encode())
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    digest.update(chunk)
                    git_digest.update(chunk)
            if item.lfs is not None:
                expected = item.lfs.sha256
                actual = digest.hexdigest()
                algorithm = 'sha256'
            else:
                expected = item.blob_id
                actual = git_digest.hexdigest()
                algorithm = 'git-blob-sha1'
            if actual != expected:
                raise RuntimeError(f'Content hash mismatch: {path}')
            verified.append({
                'file': item.rfilename, 'bytes': size,
                'sha256': digest.hexdigest(),
                'upstream_hash_algorithm': algorithm, 'upstream_hash': expected,
            })
        results[kind] = {
            'repo_id': source['repo_id'], 'revision': info.sha, 'directory': folder,
            'verified_files': verified, 'total_bytes': sum(item['bytes'] for item in verified),
        }
        print(f'Verified {kind}: {len(verified)} files', flush=True)

    report_dir = ROOT / 'reports'
    report_dir.mkdir(exist_ok=True)
    (report_dir / 'download_manifest.json').write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    manifest['download_status'].update({
        'model_weights': 'downloaded_and_hash_verified',
        'dataset_videos_and_trajectories': 'downloaded_and_hash_verified',
        'normalization_tensor_files': 'downloaded_and_hash_verified',
    })
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
