"""Run a bounded, episode-separated ACT fine-tuning experiment with reproducible resume."""

import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from cpu_budget import CPU_BUDGET

import numpy as np
import torch

from lerobot.common.train_utils import load_training_state, save_training_state
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.random_utils import set_seed
from offline_evaluation import configure_runtime, episode_frames, episode_split, evaluate, load_dataset
from short_training import ROOT, file_hash, load_policy, write_json


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items(), key=lambda item: str(item[0])):
        digest.update(str(name).encode())
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        elif isinstance(value, dict):
            digest.update(state_digest(value).encode())
        else:
            digest.update(json.dumps(value, sort_keys=True).encode())
    return digest.hexdigest()


def update(policy, preprocess, optimizer, dataset, indices):
    policy.train()
    started = time.perf_counter()
    raw = lerobot_collate_fn([dataset[int(index)] for index in indices])
    batch = preprocess(raw)
    optimizer.zero_grad(set_to_none=True)
    loss, details = policy.forward(batch)
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite training loss.')
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0, error_if_nonfinite=True).item()
    if not np.isfinite(grad_norm) or grad_norm <= 0:
        raise RuntimeError('Invalid training gradients.')
    optimizer.step()
    torch.cuda.synchronize()
    return {'loss': loss.item(), **details, 'grad_norm_before_clip': grad_norm,
            'update_and_data_s': time.perf_counter() - started}


def save_checkpoint(directory, step, policy, preprocess, postprocess, optimizer, batch_size):
    directory.mkdir(parents=True, exist_ok=False)
    model_dir = directory / 'pretrained_model'
    policy.save_pretrained(model_dir)
    preprocess.save_pretrained(model_dir)
    postprocess.save_pretrained(model_dir)
    save_training_state(directory, step, optimizer, num_processes=1, batch_size=batch_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', type=Path, help='Resume an existing experiment directory.')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-6)
    parser.add_argument('--stop-after', type=int, help='Save at this evaluation boundary and return.')
    parser.add_argument('--verify-next-step', action='store_true')
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error('steps, batch-size and learning-rate must be positive')
    if args.stop_after is not None and (args.stop_after < 1 or args.stop_after % 250 != 0):
        parser.error('stop-after must be a positive multiple of 250')
    configure_runtime()
    set_seed(1000)
    dataset = load_dataset()
    if args.resume:
        run_dir = args.resume.resolve()
        protocol = json.loads((run_dir / 'protocol.json').read_text(encoding='utf-8'))
        state = json.loads((run_dir / 'experiment_state.json').read_text(encoding='utf-8'))
        policy, preprocess, postprocess = load_policy(Path(state['last_checkpoint']) / 'pretrained_model')
    else:
        now = datetime.now(timezone(timedelta(hours=8)))
        run_dir = ROOT / 'runs' / f'act_controlled_{now:%Y%m%d_%H%M%S}'
        run_dir.mkdir(parents=True, exist_ok=False)
        split = episode_split(dataset)
        protocol = {
            'started_at': now.isoformat(), 'steps': args.steps, 'batch_size': args.batch_size,
            'learning_rate': args.learning_rate, 'eval_every': 250, 'seed': 1000,
            'split_seed': 20261019, 'episode_split': split, 'frames_per_validation_episode': 10,
            'selection_metric': 'normalized_chunk_mae', 'selection_direction': 'minimize',
            'minimum_selection_improvement': 1e-4,
            'source_model': str(ROOT / 'models/act_rebot_cube'),
            'source_sha256': file_hash(ROOT / 'models/act_rebot_cube/model.safetensors'),
            'dataset_revision': 'c5cdae89d2e1f474a8e2f1ac5f16302126d0c33f',
            'precision': 'float32', 'deterministic_algorithms': True,
            'cpu_budget': CPU_BUDGET,
            'normalization': 'Preserve the published model processors and statistics',
            'assessment_caveat': 'Partitions exclude this fine-tuning only; the pretrained model already used all episodes.',
        }
        train_indices = np.concatenate([
            np.arange(dataset.meta.episodes[episode]['dataset_from_index'],
                      dataset.meta.episodes[episode]['dataset_to_index']) for episode in split['train']
        ])
        rng = np.random.default_rng(protocol['seed'])
        count = protocol['steps'] * protocol['batch_size']
        plan = np.concatenate([rng.permutation(train_indices) for _ in range(int(np.ceil(count / len(train_indices))))])[:count]
        plan = plan.reshape(protocol['steps'], protocol['batch_size'])
        np.save(run_dir / 'batch_plan.npy', plan)
        protocol['batch_plan_sha256'] = file_hash(run_dir / 'batch_plan.npy')
        write_json(run_dir / 'protocol.json', protocol)
        policy, preprocess, postprocess = load_policy(Path(protocol['source_model']))
        validation_indices = episode_frames(dataset, split['validation'])
        baseline, _, _ = evaluate(policy, preprocess, postprocess, dataset, validation_indices)
        state = {
            'status': 'running', 'completed_steps': 0, 'best_step': 0,
            'best_validation_mae': baseline['normalized_chunk_mae'],
            'selected_model_directory': protocol['source_model'], 'last_checkpoint': None,
            'validation_history': [{'step': 0, **baseline}], 'resume_check': 'not_yet_tested',
            'training_wall_s': 0.0, 'cuda_peak_allocated_mib': 0.0,
        }
        write_json(run_dir / 'experiment_state.json', state)
    if file_hash(run_dir / 'batch_plan.npy') != protocol['batch_plan_sha256']:
        raise RuntimeError('Saved batch schedule was modified.')
    plan = np.load(run_dir / 'batch_plan.npy', allow_pickle=False)
    train_episodes = set(protocol['episode_split']['train'])
    for episode in protocol['episode_split']['validation'] + protocol['episode_split']['assessment']:
        row = dataset.meta.episodes[episode]
        if np.any((plan >= row['dataset_from_index']) & (plan < row['dataset_to_index'])):
            raise RuntimeError('Episode leakage in the fine-tuning batch schedule.')
    if not train_episodes.isdisjoint(protocol['episode_split']['validation'] + protocol['episode_split']['assessment']):
        raise RuntimeError('Episode split overlap.')
    policy.config.optimizer_lr = protocol['learning_rate']
    policy.config.optimizer_lr_backbone = protocol['learning_rate']
    optimizer = policy.config.get_optimizer_preset().build(policy.get_optim_params())
    if args.resume:
        restored_step, optimizer, _ = load_training_state(Path(state['last_checkpoint']), optimizer, None)
        if restored_step != state['completed_steps']:
            raise RuntimeError('Restored optimizer step differs from experiment state.')
        # A stopped process may have logged updates newer than the last complete checkpoint.
        metrics_path = run_dir / 'training_metrics.csv'
        with metrics_path.open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            fieldnames = reader.fieldnames
            committed_rows = [row for row in reader if int(row['step']) <= restored_step]
        with metrics_path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(committed_rows)
    validation_indices = episode_frames(dataset, protocol['episode_split']['validation'])
    stop_at = min(args.stop_after or protocol['steps'], protocol['steps'])
    expected_probe = None
    if args.resume and (run_dir / 'resume_probe_expected.json').exists():
        expected_probe = json.loads((run_dir / 'resume_probe_expected.json').read_text(encoding='utf-8'))
    print(f'Experiment: {run_dir}', flush=True)
    print(f'Continuing at step {state["completed_steps"]}; baseline/best val MAE={state["best_validation_mae"]:.6f}', flush=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with (run_dir / 'training_metrics.csv').open('a', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['step', 'loss', 'l1_loss', 'kld_loss',
                                                   'grad_norm_before_clip', 'update_and_data_s'])
        if stream.tell() == 0:
            writer.writeheader()
        for step in range(state['completed_steps'] + 1, stop_at + 1):
            metrics = update(policy, preprocess, optimizer, dataset, plan[step - 1])
            writer.writerow({'step': step, **metrics})
            stream.flush()
            if expected_probe and step == expected_probe['step']:
                if plan[step - 1].tolist() != expected_probe['batch_indices']:
                    raise RuntimeError('Resumed sample order differs from uninterrupted training.')
                if state_digest(policy.state_dict()) != expected_probe['model_digest']:
                    raise RuntimeError('Resumed model differs from the uninterrupted next update.')
                if state_digest(optimizer.state_dict()['state']) != expected_probe['optimizer_digest']:
                    raise RuntimeError('Resumed optimizer differs from the uninterrupted next update.')
                if metrics['loss'] != expected_probe['loss']:
                    raise RuntimeError('Resumed stochastic loss differs from the uninterrupted next update.')
                state['resume_check'] = 'bitwise_model_optimizer_loss_and_sample_order_match'
                print('Resume proof passed: next update matches uninterrupted training bit for bit.', flush=True)
            if step == 1 or step % 50 == 0:
                print(f'Step {step}/{protocol["steps"]}: loss={metrics["loss"]:.6f}', flush=True)
            if step % protocol['eval_every'] == 0 or step == protocol['steps']:
                summary, _, _ = evaluate(policy, preprocess, postprocess, dataset, validation_indices)
                checkpoint = run_dir / 'checkpoints' / f'step_{step:06d}'
                if checkpoint.exists():
                    checkpoint = checkpoint.with_name(f'{checkpoint.name}_retry_{time.time_ns()}')
                save_checkpoint(checkpoint, step, policy, preprocess, postprocess, optimizer, protocol['batch_size'])
                state['completed_steps'] = step
                state['last_checkpoint'] = str(checkpoint)
                state['validation_history'].append({'step': step, **summary})
                if summary['normalized_chunk_mae'] < state['best_validation_mae'] - protocol['minimum_selection_improvement']:
                    state['best_step'] = step
                    state['best_validation_mae'] = summary['normalized_chunk_mae']
                    state['selected_model_directory'] = str(checkpoint / 'pretrained_model')
                state['status'] = 'complete' if step == protocol['steps'] else 'checkpointed'
                state['cuda_peak_allocated_mib'] = max(state['cuda_peak_allocated_mib'], torch.cuda.max_memory_allocated() / 1024**2)
                write_json(run_dir / 'experiment_state.json', state)
                print(f'Validation step {step}: normalized MAE={summary["normalized_chunk_mae"]:.6f}; '
                      f'selected step={state["best_step"]}', flush=True)
    state['training_wall_s'] += time.perf_counter() - started
    if args.verify_next_step and state['completed_steps'] < protocol['steps']:
        next_step = state['completed_steps'] + 1
        probe_metrics = update(policy, preprocess, optimizer, dataset, plan[next_step - 1])
        write_json(run_dir / 'resume_probe_expected.json', {
            'step': next_step, 'batch_indices': plan[next_step - 1].tolist(), 'loss': probe_metrics['loss'],
            'model_digest': state_digest(policy.state_dict()),
            'optimizer_digest': state_digest(optimizer.state_dict()['state']),
            'note': 'One discarded diagnostic update; checkpoint remains at the previous saved step.',
        })
    if state['status'] == 'complete':
        selected, selected_pre, selected_post = load_policy(Path(state['selected_model_directory']))
        selected_validation, _, _ = evaluate(selected, selected_pre, selected_post, dataset, validation_indices)
        if abs(selected_validation['normalized_chunk_mae'] - state['best_validation_mae']) > 1e-6:
            raise RuntimeError('Selected checkpoint metric failed independent reload reproduction.')
        state['selected_checkpoint_reload_check'] = 'passed'
        state['selected_model_sha256'] = file_hash(Path(state['selected_model_directory']) / 'model.safetensors')
    if file_hash(Path(protocol['source_model']) / 'model.safetensors') != protocol['source_sha256']:
        raise RuntimeError('Published source weights changed.')
    write_json(run_dir / 'experiment_state.json', state)
    report = {**protocol, **state, 'run_directory': str(run_dir),
              'robot_commands_sent': 0, 'robot_connected': False,
              'deployment_baseline': 'Published original remains the deployment baseline pending real-robot evidence.'}
    write_json(ROOT / 'reports/controlled_training.json', report)
    (ROOT / 'reports/controlled_training_metrics.csv').write_bytes((run_dir / 'training_metrics.csv').read_bytes())
    print(json.dumps({key: report[key] for key in ['status', 'completed_steps', 'best_step', 'best_validation_mae',
                                                 'resume_check', 'run_directory']}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
