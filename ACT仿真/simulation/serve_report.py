"""Serve only the generated local simulation report, bound to loopback."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8786)
    args = parser.parse_args()
    directory = ROOT / 'runs/mujoco_validation_20261023'
    if not (directory / 'index.html').is_file():
        raise FileNotFoundError('Build the simulation preview before opening its report')
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    with ThreadingHTTPServer(('127.0.0.1', args.port), handler) as server:
        print(f'Simulation report: http://127.0.0.1:{args.port}/index.html', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
