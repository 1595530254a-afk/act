"""Read-only DOCX Package/Ole10Native verification against an MP4 manifest."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import posixpath
import re
import struct
import sys
import xml.etree.ElementTree as ET
import zipfile

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
O = 'urn:schemas-microsoft-com:office:office'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
REL = 'http://schemas.openxmlformats.org/package/2006/relationships'
APP = 'http://schemas.openxmlformats.org/officeDocument/2006/extended-properties'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def media_entries(value, base_directory):
    rows = value['entries'] if isinstance(value, dict) and 'entries' in value else value
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected entries[{id,path,sha256}] or a nonempty top-level list')
    entries, ids = [], set()
    for row in rows:
        name, expected = row.get('id'), row.get('sha256')
        if (not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name in ids
                or not isinstance(expected, str) or not re.fullmatch(r'[a-fA-F0-9]{64}', expected)):
            raise ValueError('Media IDs must be unique and SHA256 values must be explicit')
        path = Path(row['path'])
        path = path.resolve() if path.is_absolute() else (base_directory / path).resolve()
        if path.suffix.lower() != '.mp4' or not path.is_file() or digest(path) != expected.lower():
            raise ValueError(f'Missing MP4 or source SHA mismatch: {name}')
        ids.add(name)
        entries.append({'id': name, 'path': path, 'sha256': expected.lower(), 'bytes': path.stat().st_size})
    return entries


def native_payload(stream):
    """Parse the standard Package Ole10Native stream emitted by kwps/Packager."""
    if len(stream) < 6:
        raise ValueError('Truncated Ole10Native header')
    declared = struct.unpack_from('<I', stream, 0)[0]
    if declared > len(stream) - 4:
        raise ValueError('Ole10Native declared length exceeds the actual stream')
    cursor = 6  # uint32 native-data size, uint16 flags

    def cstring():
        nonlocal cursor
        end = stream.find(b'\0', cursor)
        if end < 0:
            raise ValueError('Unterminated Ole10Native filename/path')
        value = stream[cursor:end]
        cursor = end + 1
        return value.decode('utf-8', errors='replace')

    filename, source_path = cstring(), cstring()
    cursor += 8  # two Packager metadata uint32 fields
    temporary_path = cstring()
    if cursor + 4 > len(stream):
        raise ValueError('Missing Ole10Native payload length')
    size = struct.unpack_from('<I', stream, cursor)[0]
    cursor += 4
    if cursor + size > len(stream):
        raise ValueError('Truncated Ole10Native payload')
    return stream[cursor:cursor + size], {'filename': filename, 'source_path': source_path,
        'temporary_path': temporary_path, 'payload_offset': cursor, 'payload_bytes': size,
        'native_stream_bytes': len(stream), 'trailing_bytes': len(stream) - cursor - size}


def verify(docx, entries):
    try:
        import olefile
    except ImportError as error:
        raise RuntimeError('olefile is required; use an existing document Python runtime, without installing into ACT/hardware environments') from error
    source_paths = [docx, *(entry['path'] for entry in entries)]
    before = {str(path): digest(path) for path in source_paths}
    matched, objects, marker_parts = set(), [], []
    with zipfile.ZipFile(docx) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None:
            raise ValueError('DOCX duplicate members or ZIP CRC failure')
        documents = {}
        for name in names:
            if name.endswith('.xml'):
                tree = ET.fromstring(archive.read(name))
                if '[[VIDEO:' in ''.join(tree.itertext()):
                    marker_parts.append(name)
                if name.startswith('word/'):
                    documents[name] = tree
        if marker_parts:
            raise ValueError('VIDEO marker remains in ' + ', '.join(marker_parts))
        for part, tree in documents.items():
            for node in tree.iter(f'{{{O}}}OLEObject'):
                if node.get('Type') != 'Embed' or node.get('ProgID') != 'Package':
                    raise ValueError(f'Expected Type=Embed, ProgID=Package in {part}')
                relationship_id = node.get(f'{{{R}}}id')
                rel_part = posixpath.join(posixpath.dirname(part), '_rels', posixpath.basename(part) + '.rels')
                relationships = ET.fromstring(archive.read(rel_part))
                matches = [row for row in relationships.findall(f'{{{REL}}}Relationship') if row.get('Id') == relationship_id]
                if len(matches) != 1:
                    raise ValueError('Missing or duplicate OLE relationship')
                relation = matches[0]
                if relation.get('TargetMode') == 'External' or not relation.get('Type', '').endswith('/oleObject'):
                    raise ValueError('Linked/external object cannot count as an embedded video')
                target = relation.get('Target', '')
                member = target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join(posixpath.dirname(part), target))
                data = archive.read(member)
                with olefile.OleFileIO(io.BytesIO(data)) as storage:
                    if not storage.exists('\x01Ole10Native'):
                        raise ValueError('Package has no Ole10Native stream: ' + member)
                    payload, native = native_payload(storage.openstream('\x01Ole10Native').read())
                payload_sha = hashlib.sha256(payload).hexdigest()
                candidates = [entry for entry in entries if entry['id'] not in matched
                              and entry['sha256'] == payload_sha and entry['bytes'] == len(payload)]
                if not candidates:
                    raise ValueError('Embedded payload does not match a remaining manifest MP4: ' + member)
                entry = candidates[0]
                if payload != entry['path'].read_bytes():
                    raise ValueError('Complete MP4 byte comparison failed: ' + entry['id'])
                matched.add(entry['id'])
                objects.append({'id': entry['id'], 'document_part': part, 'embedding_part': member,
                    'relationship_id': relationship_id, 'Type': node.get('Type'), 'ProgID': node.get('ProgID'),
                    'source_mp4': str(entry['path']), 'payload_sha256': payload_sha,
                    'complete_mp4_bytes_equal': True, 'native': native})
        if len(objects) != len(entries) or matched != {entry['id'] for entry in entries}:
            raise ValueError('Embedded Package count differs from manifest entries')
        used_parts = {item['embedding_part'] for item in objects}
        embedded_parts = {name for name in names if name.startswith('word/embeddings/') and not name.endswith('/')}
        if used_parts != embedded_parts:
            raise ValueError('Unreferenced or unexpected embedded parts remain')
        document = documents['word/document.xml']
        properties = ET.fromstring(archive.read('docProps/app.xml')) if 'docProps/app.xml' in names else None

        def app_value(name):
            value = None if properties is None else properties.find(f'{{{APP}}}{name}')
            return None if value is None else value.text

        statistics = {'document_paragraphs_including_tables': len(list(document.iter(f'{{{W}}}p'))),
            'tables_including_nested': len(list(document.iter(f'{{{W}}}tbl'))),
            'sections': len(list(document.iter(f'{{{W}}}sectPr'))),
            'drawing_elements': len(list(document.iter(f'{{{W}}}drawing'))),
            'image_media_parts_including_OLE_icons': len([name for name in names if name.startswith('word/media/') and not name.endswith('/')]),
            'explicit_page_breaks': sum(node.get(f'{{{W}}}type') == 'page' for node in document.iter(f'{{{W}}}br')),
            'last_rendered_page_breaks': len(list(document.iter(f'{{{W}}}lastRenderedPageBreak'))),
            'pages_from_office_app_properties': app_value('Pages'), 'words_from_office_app_properties': app_value('Words'),
            'page_count_limit': 'Office app properties may be stale; final PDF/Poppler rendering determines actual pagination.'}
    after = {str(path): digest(path) for path in source_paths}
    if before != after:
        raise ValueError('DOCX or source MP4 changed during read-only verification')
    return {'status': 'passed_embedded_package_full_MP4_verification', 'docx': str(docx),
        'docx_sha256': before[str(docx)], 'manifest_entries': len(entries),
        'ole_Type_Embed_ProgID_Package_count': len(objects), 'marker_residue_count': 0,
        'all_payloads_complete_byte_equal_and_SHA256_equal': True,
        'objects': objects, 'statistics': statistics,
        'source_files_sha256_before': before, 'source_files_sha256_after': after,
        'sources_unchanged': True, 'office_COM_opened': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--docx', type=Path, required=True)
    parser.add_argument('--media-manifest', type=Path, required=True)
    parser.add_argument('--json-output', type=Path)
    args = parser.parse_args(argv)
    manifest = args.media_manifest.resolve()
    initial_manifest_sha = digest(manifest)
    entries = media_entries(json.loads(manifest.read_text(encoding='utf-8-sig')), manifest.parent)
    result = verify(args.docx.resolve(), entries)
    if digest(manifest) != initial_manifest_sha:
        raise ValueError('Media manifest changed during verification')
    result.update({'media_manifest': str(manifest), 'media_manifest_sha256': initial_manifest_sha})
    if args.json_output:
        with args.json_output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'verification_failed', 'error': f'{type(error).__name__}: {error}'}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
