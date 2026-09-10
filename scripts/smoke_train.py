"""Generate a small real-stimulus dataset and verify training/checkpoint resume.

Run with PYTHONPATH=src python scripts/smoke_train.py [new_output_directory].
"""
import hashlib
import json
from pathlib import Path
import sys
import time
from dataclasses import asdict

import numpy as np
import torch

from data.generate_flow_xt_data import StimConfig, generate_flow_xt_dataset
from data.precompute_reichardt import precompute_dataset
from features.reichardt import ReichardtBank
from train.train_flow_xt_net import Settings, train
from train.test_flow_xt_net import run_flow_xt_model_on_test_data


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = Path(sys.argv[1] if len(sys.argv) > 1 else 'data/reichardt-pilot').resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    started = time.perf_counter()
    config = StimConfig(stim_wid_px=128, n_world_dots=120, x_supersampling=10)
    source = Path(generate_flow_xt_dataset(
        output_root=str(output / 'raw'), n_samples=80, seed=42, config=config))
    raw_files = sorted(source.rglob('*.pt'))
    raw = [torch.load(p, weights_only=True) for p in raw_files]
    assert all(x.dtype == torch.float32 and torch.isfinite(x).all() for x in raw)
    assert any(((x > 0) & (x < 1)).any() for x in raw), 'Antialiasing was lost'
    bank = ReichardtBank([-3, -2, -1, 1, 2, 3])
    cache = output / 'reichardt' / source.name
    precompute_dataset(source, cache, bank, torch.float16)
    preprocessing_seconds = time.perf_counter() - started
    for p in raw_files:
        saved = torch.load(cache / p.relative_to(source), weights_only=True)
        expected = bank(torch.load(p, weights_only=True)).half()
        torch.testing.assert_close(saved, expected, rtol=0, atol=0)
    cfg = Settings(root_folder=str(cache), checkpoints_dir=str(output / 'checkpoints'),
                   num_epochs=3, batch_size=8)
    print('Training first three epochs...', flush=True)
    train(cfg)
    checkpoint_dir = output / 'checkpoints' / source.name
    latest_path = checkpoint_dir / '_latest.pt'
    first = torch.load(latest_path, map_location='cpu', weights_only=True)
    assert first['epoch'] == 3
    first_hashes = {p.name: digest(p) for p in checkpoint_dir.glob('epoch_*.pt')}
    first_steps = [int(v['step']) for v in first['optimizer_state_dict']['state'].values()]
    assert first_steps and set(first_steps) == {24}
    cfg.num_epochs = 5
    print('Resuming to epoch five...', flush=True)
    train(cfg)
    latest = torch.load(latest_path, map_location='cpu', weights_only=True)
    assert latest['epoch'] == 5
    assert all(digest(checkpoint_dir / name) == value for name, value in first_hashes.items())
    steps = [int(v['step']) for v in latest['optimizer_state_dict']['state'].values()]
    assert set(steps) == {40}, 'Adam steps did not resume'
    assert any(not torch.equal(first['model_state_dict'][k], v)
               for k, v in latest['model_state_dict'].items() if v.is_floating_point())
    epochs = [torch.load(checkpoint_dir / f'epoch_{i:06d}.pt', map_location='cpu', weights_only=True)
              for i in range(1, 6)]
    best = torch.load(checkpoint_dir / '_best.pt', map_location='cpu', weights_only=True)
    minimum = min(x['val_rmse'] for x in epochs)
    assert best['val_rmse'] == latest['best_val_rmse'] == minimum
    for checkpoint in epochs:
        assert checkpoint['reichardt'] == bank.metadata
        assert all(torch.isfinite(v).all() for v in checkpoint['model_state_dict'].values())
    # Resuming an already-completed target must not alter any checkpoints.
    hashes = {p.name: digest(p) for p in checkpoint_dir.glob('*.pt')}
    train(cfg)
    assert all(digest(checkpoint_dir / name) == value for name, value in hashes.items())
    for name in ('_latest.pt', '_best.pt'):
        results = run_flow_xt_model_on_test_data(str(cache), str(checkpoint_dir / name), 8)
        assert results['preds'].shape == (8, 3) and np.isfinite(results['preds']).all()
        repeated = run_flow_xt_model_on_test_data(str(cache), str(checkpoint_dir / name), 8)
        np.testing.assert_array_equal(results['preds'], repeated['preds'])
    report = {
        'status': 'passed', 'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'stimulus_config': asdict(config), 'seed': 42,
        'splits': {s: len(list((source / s).glob('*.pt'))) for s in ('train', 'validate', 'test')},
        'raw_shape': list(raw[0].shape), 'response_shape': list(expected.shape),
        'raw_levels': sorted(set(v.item() for x in raw for v in x.unique())),
        'reichardt': bank.metadata, 'storage_dtype': 'float16',
        'raw_bytes': sum(p.stat().st_size for p in raw_files),
        'response_bytes': sum(p.stat().st_size for p in cache.rglob('*.pt')),
        'preprocessing_seconds': preprocessing_seconds,
        'total_seconds': time.perf_counter() - started,
        'epochs': [{'epoch': x['epoch'], 'train_loss': x['loss'], 'val_rmse': x['val_rmse']} for x in epochs],
        'best_epoch': best['epoch'], 'best_val_rmse': minimum,
        'adam_steps_before_resume': first_steps[0], 'adam_steps_after_resume': steps[0],
        'checks': ['fractional raw intensities', 'exact float16 cache products',
                   'finite training weights', 'resume epoch and Adam state',
                   'weights update after resume', 'earlier epoch files preserved',
                   'best checkpoint matches minimum validation RMSE',
                   'completed-target resume changes no files',
                   'latest and best checkpoints reload and evaluate reproducibly'],
        'raw_directory': str(source), 'cache_directory': str(cache),
        'checkpoint_directory': str(checkpoint_dir),
    }
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
