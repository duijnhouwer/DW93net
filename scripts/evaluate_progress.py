"""Reusable translation checkpoint evaluation; run with --help for options."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import subprocess
from datetime import datetime, timezone

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from stimuli.load_flow_xt_data import ReichardtDataset, require_pure_translation
from models.pure_translation import PureTranslationNet

ROOT = Path(__file__).resolve().parents[1]


def metrics(pred, target):
    """Heading uses atan2(dx,dz); errors wrap across +/-180 degrees."""
    pred, target = np.asarray(pred, dtype=float), np.asarray(target, dtype=float)
    if pred.shape != target.shape or pred.ndim != 2 or pred.shape[1] != 2:
        raise ValueError('Expected matching [N,2] translation vectors')
    if not len(pred) or not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError('Empty or nonfinite predictions/targets')
    speed, truth = np.linalg.norm(pred, axis=1), np.linalg.norm(target, axis=1)
    valid = (speed > 1e-12) & (truth > 1e-12)
    delta = np.arctan2(pred[valid, 0], pred[valid, 1]) - np.arctan2(target[valid, 0], target[valid, 1])
    angles = np.degrees(np.arctan2(np.sin(delta), np.cos(delta)))
    absolute = np.abs(angles)
    return {
        'n': len(pred), 'undefined_heading_count': int((~valid).sum()),
        'heading_mae_deg': float(absolute.mean()) if len(angles) else None,
        'heading_median_deg': float(np.median(absolute)) if len(angles) else None,
        'heading_p90_deg': float(np.percentile(absolute, 90)) if len(angles) else None,
        'heading_bias_deg': float(angles.mean()) if len(angles) else None,
        # Undefined estimates count as failures in threshold accuracy.
        'heading_within_15_pct': float(100 * (absolute <= 15).sum() / len(pred)),
        'speed_mae': float(np.abs(speed - truth).mean()),
        'speed_bias': float((speed - truth).mean()),
        'speed_rmse': float(np.sqrt(np.mean((speed - truth)**2))),
        'vector_rmse': float(np.sqrt(np.mean(np.sum((pred - target)**2, axis=1)))),
        'true_speed_min': float(truth.min()), 'true_speed_max': float(truth.max()),
    }


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', type=Path, help='Default: sole stimulus directory under checkpoints/<commit date>-<current commit>')
    parser.add_argument('--data', type=Path, help='Default: stimuli/<checkpoint directory name>/reichardt')
    parser.add_argument('--split', choices=['validate', 'test'], default='validate')
    parser.add_argument('--epochs', nargs='+', type=int,
                        help='Evaluate only these epoch numbers; omit to evaluate every saved epoch')
    parser.add_argument('--samples', type=int, default=256, help='Fixed seeded subset; 0 means all')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--output', type=Path, default=ROOT / 'evaluations')
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    if args.samples < 0 or args.batch_size < 1 or args.threads < 1:
        parser.error('samples must be >=0; batch-size and threads must be positive')
    if args.checkpoints is None:
        commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        date = subprocess.check_output(
            ['git', '-C', str(ROOT), 'show', '-s', '--format=%cd', '--date=format:%Y%m%d', commit], text=True).strip()
        candidates = sorted((ROOT / 'checkpoints' / f'{date}-{commit}').glob('*/_latest.pt'))
        if len(candidates) != 1:
            parser.error('Specify --checkpoints (could not identify a unique checkpoint directory)')
        args.checkpoints = candidates[0].parent
    args.checkpoints = args.checkpoints.resolve()
    args.data = (args.data or ROOT / 'stimuli' / args.checkpoints.name / 'reichardt').resolve()
    dataset = ReichardtDataset(args.data / args.split)
    require_pure_translation(dataset)
    if not len(dataset):
        parser.error('Dataset is empty')
    indices = np.random.default_rng(args.seed).permutation(len(dataset))
    indices = indices[:args.samples or len(dataset)].tolist()
    selected = [dataset.files[i] for i in indices]
    manifest = {'schema': 1, 'data': str(args.data), 'split': args.split,
                'seed': args.seed, 'files': selected, 'metadata': dataset.metadata,
                'file_stats': [(f, (args.data / args.split / f).stat().st_size,
                                (args.data / args.split / f).stat().st_mtime_ns) for f in selected],
                'code': fingerprint([Path(__file__).read_text(),
                    (ROOT / 'src/models/pure_translation.py').read_text(),
                    (ROOT / 'src/stimuli/load_flow_xt_data.py').read_text()]),
                'torch': torch.__version__, 'device': args.device}
    out = args.output / fingerprint(manifest)[:16]
    out.mkdir(parents=True, exist_ok=True)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    torch.set_num_threads(args.threads)
    rows = []
    paths = sorted(args.checkpoints.glob('epoch_*.pt'))
    if args.epochs is not None:
        requested = set(args.epochs)
        paths = [path for path in paths if int(path.stem.removeprefix('epoch_')) in requested]
    paths += [p for p in [args.checkpoints / '_best.pt', args.checkpoints / '_latest.pt'] if p.exists()]
    if not paths:
        parser.error('No checkpoints found')
    for path in paths:
        before = path.stat()
        identity = fingerprint({'path': str(path), 'size': before.st_size, 'mtime': before.st_mtime_ns})
        saved = out / (identity + '.json')
        if saved.exists() and not args.refresh:
            row = json.loads(saved.read_text())
            print(f'{path.name}: cached', flush=True)
        else:
            print(f'{path.name}: evaluating {len(indices)} {args.split} examples on {args.device}', flush=True)
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            if checkpoint['architecture'] != PureTranslationNet.architecture or checkpoint['reichardt'] != dataset.metadata:
                raise ValueError(f'Incompatible architecture/detectors: {path}')
            model = PureTranslationNet(checkpoint['reichardt']['dx_values']).to(args.device).eval()
            model.load_state_dict(checkpoint['model_state_dict'])
            loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, indices),
                                                batch_size=args.batch_size, num_workers=0)
            predictions, targets = [], []
            with torch.inference_mode():
                for x, y in loader:
                    predictions.append(model(x.to(args.device)).cpu().numpy())
                    targets.append(y[:, :2].numpy())
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f'Checkpoint changed during evaluation; rerun: {path}')
            row = {'checkpoint': str(path), 'epoch': checkpoint['epoch'],
                   'saved_at': datetime.fromtimestamp(before.st_mtime, timezone.utc).isoformat(),
                   'settings': checkpoint['settings'],
                   'metrics': metrics(np.concatenate(predictions), np.concatenate(targets))}
            saved.write_text(json.dumps(row, indent=2, allow_nan=False))
        rows.append(row)
    rows.sort(key=lambda r: r['saved_at'])
    report = ['# Translation progress', '',
              f'{len(indices)} fixed {args.split} examples; seed {args.seed}. Heading is primary.',
              'Speed/vector errors are scene units per step. Fixed-speed results do not establish variable-speed recovery.',
              'Ordered by save time. Legacy checkpoints may belong to overlapping runs; epoch numbers alone do not identify a run.', '',
              '| Checkpoint | Epoch | Saved (UTC) | Heading MAE ° | Median ° | Within 15° % | Speed RMSE | Vector RMSE |',
              '|---|---:|---|---:|---:|---:|---:|---:|']
    for row in rows:
        m = row['metrics']
        def number(key):
            return 'undefined' if m[key] is None else f'{m[key]:.6g}'
        report.append(f"| {Path(row['checkpoint']).name} | {row['epoch']} | {row['saved_at']} | " +
                      ' | '.join(number(k) for k in ['heading_mae_deg', 'heading_median_deg',
                          'heading_within_15_pct', 'speed_rmse', 'vector_rmse']) + ' |')
    (out / 'progress.md').write_text('\n'.join(report) + '\n')
    print('\n'.join(report))
    print(f'\nSaved report: {out / "progress.md"}')


if __name__ == '__main__':
    main()
