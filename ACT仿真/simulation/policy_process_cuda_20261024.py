"""Use the established artifact IPC with the new CUDA-only ACT worker."""
import os
import queue
import subprocess
import threading
import time

from run_task import PolicyProcess, ROOT


class CudaPolicyProcess(PolicyProcess):
    def __init__(self, model_dir, output):
        self.output = output
        self.log = (output / 'policy_stderr.log').open('w', encoding='utf-8')
        self.process = subprocess.Popen([
            str(ROOT / '.venv/Scripts/python.exe'), '-u',
            str(ROOT / 'simulation/policy_worker_cuda_20261024.py'), '--model-dir', str(model_dir)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, encoding='utf-8', bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        self.messages = queue.Queue()

        def receive():
            for line in self.process.stdout:
                self.messages.put(line)
            self.messages.put(None)

        self.reader = threading.Thread(target=receive, daemon=True)
        self.reader.start()
        try:
            self.identity = self.read(timeout=120)
            if (self.identity.get('status') != 'ready'
                    or self.identity.get('model_identity', {}).get('device') != 'cuda'):
                raise RuntimeError(f'CUDA ACT worker did not initialize: {self.identity}')
        except Exception:
            self.close()
            raise

    def predict(self, state, wrist, top, index):
        started = time.perf_counter()
        action, response = super().predict(state, wrist, top, index)
        response['client_roundtrip_ms'] = (time.perf_counter() - started) * 1000
        response['client_roundtrip_includes'] = 'Input compression, disk I/O, IPC, preprocessing and fresh forward'
        return action, response
