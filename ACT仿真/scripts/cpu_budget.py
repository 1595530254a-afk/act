"""Keep desktop ACT jobs within a conservative, process-local CPU budget."""

import os

import psutil

# Set these before NumPy, PyTorch and video libraries initialize their pools.
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '2'
os.environ['RAYON_NUM_THREADS'] = '2'

process = psutil.Process()
allowed_cpus = process.cpu_affinity()[:4]
process.cpu_affinity(allowed_cpus)
process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == 'nt' else 10)
CPU_BUDGET = {
    'logical_cpus_allowed': process.cpu_affinity(),
    'system_logical_cpu_count': psutil.cpu_count(),
    'priority': 'below_normal', 'torch_threads': 2, 'torch_interop_threads': 1,
    'scope': 'This ACT process only; CPU affinity is not a temperature or power limit.',
}
_torch_configured = False


def configure_torch():
    global _torch_configured
    if not _torch_configured:
        import torch

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        _torch_configured = True
        print(f'CPU budget: {CPU_BUDGET}', flush=True)
    return CPU_BUDGET
