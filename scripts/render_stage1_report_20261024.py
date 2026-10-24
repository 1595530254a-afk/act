"""Rasterize a WPS-exported PDF with the canonical document renderer.

The standard renderer was attempted first but this host has no soffice.exe.
Only its PDF-conversion hook is replaced; its Poppler rasterization is retained.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cpu_budget import CPU_BUDGET


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--docx', type=Path, required=True)
    p.add_argument('--pdf', type=Path, required=True)
    p.add_argument('--renderer', type=Path, required=True)
    p.add_argument('--poppler', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    for path in (a.docx, a.pdf, a.renderer):
        if not path.is_file():
            raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location('canonical_document_renderer', a.renderer)
    renderer = importlib.util.module_from_spec(spec); spec.loader.exec_module(renderer)
    renderer.convert_to_pdf = lambda *_args, **_kwargs: (str(a.pdf.resolve()), 'Native kwps COM PDF export; original soffice route unavailable')
    original = renderer.convert_from_path
    def bounded_poppler(*args, **kwargs):
        kwargs['thread_count'] = 2
        kwargs['poppler_path'] = str(a.poppler)
        return original(*args, **kwargs)
    renderer.convert_from_path = bounded_poppler
    paths = renderer.rasterize(str(a.docx), str(a.output_dir), 110, verbose=False, emit_pdf=False)
    record = {'pages': len(paths), 'page_images': paths, 'PDF_source': str(a.pdf),
              'canonical_renderer': str(a.renderer), 'original_route_error': 'LibreOffice soffice.exe was not found on PATH',
              'fallback': 'Existing native WPS COM PDF export + bundled Poppler; no software installed', 'cpu_budget': CPU_BUDGET}
    (a.output_dir / 'render_record.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'pages':len(paths),'output_dir':str(a.output_dir)},ensure_ascii=False))


if __name__ == '__main__':
    main()
