"""Reusable translation checkpoint evaluation; run with --help for options."""
import argparse
import hashlib
import json
import os
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


def mean_ci95(values):
    """Normal-approximation 95% confidence interval for a sample mean."""
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean
    margin = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values))
    return max(0.0, mean - margin), mean + margin


def metrics(pred, target):
    """Compute angular, speed, and Euclidean translation-vector tip errors."""
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
    relative_heading_error_pct = 100 * absolute / 180
    tip_distance = np.linalg.norm(pred - target, axis=1)
    speed_absolute = np.abs(speed - truth)
    relative_speed_error_pct = 100 * speed_absolute / truth
    relative_tip_error_pct = 100 * tip_distance / truth
    cosine_similarity = np.sum(pred[valid] * target[valid], axis=1) / (speed[valid] * truth[valid])
    heading_ci_low, heading_ci_high = mean_ci95(absolute) if len(absolute) else (None, None)
    relative_heading_ci_low, relative_heading_ci_high = (
        mean_ci95(relative_heading_error_pct) if len(relative_heading_error_pct) else (None, None))
    speed_ci_low, speed_ci_high = mean_ci95(speed_absolute)
    tip_ci_low, tip_ci_high = mean_ci95(tip_distance)
    relative_speed_ci_low, relative_speed_ci_high = mean_ci95(relative_speed_error_pct)
    relative_tip_ci_low, relative_tip_ci_high = mean_ci95(relative_tip_error_pct)
    return {
        'n': len(pred), 'undefined_heading_count': int((~valid).sum()),
        'heading_mae_deg': float(absolute.mean()) if len(angles) else None,
        'heading_mae_deg_ci95_low': heading_ci_low,
        'heading_mae_deg_ci95_high': heading_ci_high,
        'heading_relative_mae_pct': float(relative_heading_error_pct.mean()) if len(absolute) else None,
        'heading_relative_mae_pct_ci95_low': relative_heading_ci_low,
        'heading_relative_mae_pct_ci95_high': relative_heading_ci_high,
        'heading_median_deg': float(np.median(absolute)) if len(angles) else None,
        'heading_p90_deg': float(np.percentile(absolute, 90)) if len(angles) else None,
        'heading_bias_deg': float(angles.mean()) if len(angles) else None,
        'heading_cosine_similarity': float(cosine_similarity.mean()) if len(cosine_similarity) else None,
        # Undefined estimates count as failures in threshold accuracy.
        'heading_within_15_pct': float(100 * (absolute <= 15).sum() / len(pred)),
        'speed_mae': float(speed_absolute.mean()),
        'speed_mae_ci95_low': speed_ci_low,
        'speed_mae_ci95_high': speed_ci_high,
        'speed_relative_mae_pct': float(relative_speed_error_pct.mean()),
        'speed_relative_mae_pct_ci95_low': relative_speed_ci_low,
        'speed_relative_mae_pct_ci95_high': relative_speed_ci_high,
        'speed_bias': float((speed - truth).mean()),
        'speed_rmse': float(np.sqrt(np.mean((speed - truth)**2))),
        # The physical error of the predicted vector tip, ||[dx,dz]-[dx_hat,dz_hat]||.
        'vector_tip_mae': float(tip_distance.mean()),
        'vector_tip_mae_ci95_low': tip_ci_low,
        'vector_tip_mae_ci95_high': tip_ci_high,
        'vector_tip_relative_mae_pct': float(relative_tip_error_pct.mean()),
        'vector_tip_relative_mae_pct_ci95_low': relative_tip_ci_low,
        'vector_tip_relative_mae_pct_ci95_high': relative_tip_ci_high,
        'vector_tip_median': float(np.median(tip_distance)),
        'vector_tip_p90': float(np.percentile(tip_distance, 90)),
        # Retained for comparisons with reports produced before vector_tip_mae.
        'vector_rmse': float(np.sqrt(np.mean(tip_distance**2))),
        'true_speed_min': float(truth.min()), 'true_speed_max': float(truth.max()),
    }


CARDINAL_HEADINGS = {0: 'Front', 90: 'Right', 180: 'Back', 270: 'Left'}


def empty_metrics():
    """Metric schema for a requested stratum that contains no samples."""
    return {
        'n': 0, 'undefined_heading_count': 0,
        'heading_mae_deg': None, 'heading_median_deg': None,
        'heading_mae_deg_ci95_low': None, 'heading_mae_deg_ci95_high': None,
        'heading_relative_mae_pct': None, 'heading_relative_mae_pct_ci95_low': None,
        'heading_relative_mae_pct_ci95_high': None,
        'heading_p90_deg': None, 'heading_bias_deg': None, 'heading_cosine_similarity': None,
        'heading_within_15_pct': None, 'speed_mae': None,
        'speed_mae_ci95_low': None, 'speed_mae_ci95_high': None,
        'speed_relative_mae_pct': None, 'speed_relative_mae_pct_ci95_low': None,
        'speed_relative_mae_pct_ci95_high': None,
        'speed_bias': None, 'speed_rmse': None, 'vector_tip_mae': None,
        'vector_tip_mae_ci95_low': None, 'vector_tip_mae_ci95_high': None,
        'vector_tip_relative_mae_pct': None, 'vector_tip_relative_mae_pct_ci95_low': None,
        'vector_tip_relative_mae_pct_ci95_high': None,
        'vector_tip_median': None, 'vector_tip_p90': None, 'vector_rmse': None,
        'true_speed_min': None, 'true_speed_max': None,
    }


def heading_bin_label(center_deg, count):
    """Name a truth-heading bin; zero degrees is front and +90 is right."""
    center_deg = int(round(center_deg)) % 360
    if count == 1:
        return 'All headings'
    if count == 4:
        return CARDINAL_HEADINGS[center_deg]
    return f'{center_deg}°'


def binned_metrics(pred, target, *, dimension, bin_counts):
    """Return equal-width truth-heading or truth-speed metric strata.

    Heading bins are circular and centred at zero/front. Speed bins span the
    minimum and maximum true speeds in this fixed evaluation sample.
    """
    pred, target = np.asarray(pred), np.asarray(target)
    if dimension == 'heading':
        values = np.mod(np.degrees(np.arctan2(target[:, 0], target[:, 1])), 360)
        span = 360.0
    elif dimension == 'speed':
        values = np.linalg.norm(target, axis=1)
        span = None
    else:
        raise ValueError(f'Unknown dimension: {dimension}')

    result = {}
    for count in bin_counts:
        if dimension == 'heading':
            width = span / count
            indices = np.floor(np.mod(values + width / 2, span) / width).astype(int)
            indices = np.minimum(indices, count - 1)
            bins = []
            for index in range(count):
                center = index * width
                lower = (center - width / 2) % span
                upper = (center + width / 2) % span
                mask = indices == index
                bins.append({
                    'label': heading_bin_label(center, count),
                    'center_deg': float(center), 'lower_deg': float(lower),
                    'upper_deg': float(upper), 'wraps': bool(lower > upper),
                    'metrics': metrics(pred[mask], target[mask]) if mask.any() else empty_metrics(),
                })
        else:
            low, high = float(values.min()), float(values.max())
            edges = np.linspace(low, high, count + 1)
            indices = np.minimum(np.searchsorted(edges, values, side='right') - 1, count - 1)
            bins = []
            for index in range(count):
                mask = indices == index
                bins.append({
                    'label': 'All speeds' if count == 1 else f'Speed {index + 1}',
                    'lower_speed': float(edges[index]), 'upper_speed': float(edges[index + 1]),
                    'upper_inclusive': index == count - 1,
                    'metrics': metrics(pred[mask], target[mask]) if mask.any() else empty_metrics(),
                })
        result[str(count)] = bins
    return result


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def bin_description(dimension, bin_data):
    """Human-readable truth range corresponding to a serialized bin."""
    if dimension == 'heading':
        if bin_data['label'] == 'All headings':
            return bin_data['label']
        lower, upper = bin_data['lower_deg'], bin_data['upper_deg']
        return f"{bin_data['label']} [{lower:g}°, {upper:g}°)" + (' (wraps)' if bin_data['wraps'] else '')
    close = ']' if bin_data['upper_inclusive'] else ')'
    return f"{bin_data['label']} [{bin_data['lower_speed']:.6g}, {bin_data['upper_speed']:.6g}{close}"


PLOT_METRICS = [
    ('heading_mae_deg', 'Heading MAE (°)'),
    ('speed_relative_mae_pct', 'Speed MAE (% of true speed)'),
    ('vector_tip_relative_mae_pct', 'Vector-tip MAE (% of true step length)'),
]

PERCENT_PLOT_METRICS = [
    ('heading_relative_mae_pct', 'Relative heading MAE (% of 180° maximum)'),
]


def plot_metric(axis, x, values, key, *, label=None):
    """Draw a mean metric with its asymmetric (or empty) 95% CI error bars."""
    means = np.asarray([value[key] if value[key] is not None else np.nan for value in values])
    low = np.asarray([value[f'{key}_ci95_low'] if value[f'{key}_ci95_low'] is not None else np.nan
                      for value in values])
    high = np.asarray([value[f'{key}_ci95_high'] if value[f'{key}_ci95_high'] is not None else np.nan
                       for value in values])
    errors = np.vstack((means - low, high - means))
    axis.errorbar(x, means, yerr=errors, marker='o', capsize=3, label=label)
    axis.set_ylim(bottom=0)


def save_plots(rows, output_dir, heading_counts, speed_counts, *, heading_percent=False, show=False):
    """Plot global progress and every requested non-global truth-value breakdown."""
    os.environ.setdefault('MPLCONFIGDIR', str(output_dir / '.matplotlib'))
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plot_dir = output_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)
    figures = []
    labels = [f"{Path(row['checkpoint']).name}\nepoch {row['epoch']}" for row in rows]
    figure, axes = plt.subplots(3, 1, figsize=(9, 9), layout='constrained')
    for axis, (key, title) in zip(axes, PLOT_METRICS):
        plot_metric(axis, labels, [row['metrics'] for row in rows], key)
        axis.set_ylabel(title)
        axis.grid(axis='y', alpha=0.3)
    axes[0].set_title('Heading and relative self-motion errors')
    global_path = plot_dir / 'global_metrics.png'
    figure.savefig(global_path, dpi=160)
    figures.append(figure)

    paths = [global_path]
    for dimension, counts in [('heading', heading_counts), ('speed', speed_counts)]:
        for count in counts:
            if count == 1:
                continue
            bins = rows[0]['breakdowns'][dimension][str(count)]
            x = np.arange(len(bins))
            figure, axes = plt.subplots(3, 1, figsize=(max(9, count * 1.1), 9), layout='constrained')
            for axis, (key, title) in zip(axes, PLOT_METRICS):
                for row in rows:
                    checkpoint_name = Path(row['checkpoint']).name.lstrip('_')
                    values = [bin_data['metrics'] for bin_data in row['breakdowns'][dimension][str(count)]]
                    plot_metric(axis, x, values, key,
                                label=f"{checkpoint_name} (epoch {row['epoch']})")
                axis.set_ylabel(title)
                axis.grid(axis='y', alpha=0.3)
            axes[0].set_title(f'Heading and relative errors by true {dimension}: {count} bins')
            for axis in axes:
                axis.set_xticks(x, [bin_data['label'] for bin_data in bins])
            axes[0].legend()
            path = plot_dir / f'{dimension}_bins_{count}.png'
            figure.savefig(path, dpi=160)
            figures.append(figure)
            paths.append(path)
    if heading_percent:
        figure, axes = plt.subplots(len(PERCENT_PLOT_METRICS), 1,
                                    figsize=(9, 3 * len(PERCENT_PLOT_METRICS)), layout='constrained')
        axes = np.atleast_1d(axes)
        for axis, (key, title) in zip(axes, PERCENT_PLOT_METRICS):
            plot_metric(axis, labels, [row['metrics'] for row in rows], key)
            axis.set_ylabel(title)
            axis.grid(axis='y', alpha=0.3)
        axes[0].set_title('Global relative errors')
        path = plot_dir / 'global_metrics_percent.png'
        figure.savefig(path, dpi=160)
        figures.append(figure)
        paths.append(path)
        for dimension, counts in [('heading', heading_counts), ('speed', speed_counts)]:
            for count in counts:
                if count == 1:
                    continue
                bins = rows[0]['breakdowns'][dimension][str(count)]
                x = np.arange(len(bins))
                figure, axes = plt.subplots(len(PERCENT_PLOT_METRICS), 1,
                                            figsize=(max(9, count * 1.1), 3 * len(PERCENT_PLOT_METRICS)),
                                            layout='constrained')
                axes = np.atleast_1d(axes)
                for axis, (key, title) in zip(axes, PERCENT_PLOT_METRICS):
                    for row in rows:
                        checkpoint_name = Path(row['checkpoint']).name.lstrip('_')
                        values = [bin_data['metrics'] for bin_data in row['breakdowns'][dimension][str(count)]]
                        plot_metric(axis, x, values, key,
                                    label=f"{checkpoint_name} (epoch {row['epoch']})")
                    axis.set_ylabel(title)
                    axis.grid(axis='y', alpha=0.3)
                axes[0].set_title(f'Relative errors by true {dimension}: {count} bins')
                for axis in axes:
                    axis.set_xticks(x, [bin_data['label'] for bin_data in bins])
                axes[0].legend()
                path = plot_dir / f'{dimension}_bins_{count}_percent.png'
                figure.savefig(path, dpi=160)
                figures.append(figure)
                paths.append(path)
    if show:
        plt.show()
    for figure in figures:
        plt.close(figure)
    return paths


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
    parser.add_argument('--heading-bins', nargs='+', type=int, default=[1, 4],
                        help='Numbers of equal-width circular truth-heading bins (default: 1 4)')
    parser.add_argument('--speed-bins', nargs='+', type=int, default=[1, 3],
                        help='Numbers of equal-width truth-speed bins (default: 1 3)')
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument('--plots', dest='plots', action='store_true', default=True,
                            help='Save PNG plots alongside the report (default)')
    plot_group.add_argument('--no-plots', dest='plots', action='store_false',
                            help='Do not save PNG plots')
    parser.add_argument('--show-plots', action='store_true',
                        help='Open the saved plots in local interactive windows (implies --plots)')
    parser.add_argument('--plot-heading-percent', '--plot-percent', dest='plot_heading_percent', action='store_true',
                        help='Also save heading error as a percentage of the 180° maximum')
    parser.add_argument('--output', type=Path, default=ROOT / 'evaluations')
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    if (args.samples < 0 or args.batch_size < 1 or args.threads < 1
            or any(count < 1 for count in args.heading_bins + args.speed_bins)):
        parser.error('samples must be >=0; batch-size, threads, and all bin counts must be >=1')
    args.heading_bins = sorted(set(args.heading_bins))
    args.speed_bins = sorted(set(args.speed_bins))
    if args.show_plots:
        args.plots = True
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
                'heading_bins': args.heading_bins, 'speed_bins': args.speed_bins,
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
                    predictions.append(model(x.to(args.device))[:, :2].cpu().numpy())
                    targets.append(y[:, :2].numpy())
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f'Checkpoint changed during evaluation; rerun: {path}')
            row = {'checkpoint': str(path), 'epoch': checkpoint['epoch'],
                   'saved_at': datetime.fromtimestamp(before.st_mtime, timezone.utc).isoformat(),
                   'settings': checkpoint['settings']}
            prediction = np.concatenate(predictions)
            target = np.concatenate(targets)
            row['metrics'] = metrics(prediction, target)
            row['breakdowns'] = {
                'heading': binned_metrics(prediction, target, dimension='heading',
                                          bin_counts=args.heading_bins),
                'speed': binned_metrics(prediction, target, dimension='speed',
                                        bin_counts=args.speed_bins),
            }
            saved.write_text(json.dumps(row, indent=2, allow_nan=False))
        rows.append(row)
    rows.sort(key=lambda r: r['saved_at'])
    report = ['# Translation progress', '',
              f'{len(indices)} fixed {args.split} examples; seed {args.seed}. Heading is primary.',
              'Speed and vector-tip errors are scene units per step. Vector-tip error is ||[dx,dz]-[dx_hat,dz_hat]||. Plot error bars are normal-approximation 95% confidence intervals of the mean. Fixed-speed results do not establish variable-speed recovery.',
              'Ordered by save time. Legacy checkpoints may belong to overlapping runs; epoch numbers alone do not identify a run.', '',
              '| Checkpoint | Epoch | Saved (UTC) | Heading MAE ° | Cosine similarity | Median ° | Within 15° % | Speed MAE % | Speed RMSE | Vector-tip MAE % | Vector-tip median |',
              '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        m = row['metrics']
        def number(key):
            return 'undefined' if m[key] is None else f'{m[key]:.6g}'
        report.append(f"| {Path(row['checkpoint']).name} | {row['epoch']} | {row['saved_at']} | " +
                      ' | '.join(number(k) for k in ['heading_mae_deg', 'heading_cosine_similarity', 'heading_median_deg',
                          'heading_within_15_pct', 'speed_relative_mae_pct', 'speed_rmse',
                          'vector_tip_relative_mae_pct', 'vector_tip_median']) + ' |')
    for dimension in ('heading', 'speed'):
        explanation = ('Bins use the true heading; ranges are circular, with 0° at front.'
                       if dimension == 'heading' else
                       'Bins use the true speed; edges span this fixed evaluation sample.')
        report.extend(['', f'## By true {dimension}',
                       explanation])
        for count in (args.heading_bins if dimension == 'heading' else args.speed_bins):
            report.extend(['', f'### {count} bin' + ('' if count == 1 else 's'),
                           '| Checkpoint | True-value bin | N | Heading MAE ° | Speed MAE % | Vector-tip MAE % |',
                           '|---|---|---:|---:|---:|---:|'])
            for row in rows:
                for bin_data in row['breakdowns'][dimension][str(count)]:
                    m = bin_data['metrics']
                    report.append(f"| {Path(row['checkpoint']).name} | {bin_description(dimension, bin_data)} | {m['n']} | " +
                                  ' | '.join(number(key) for key in
                                             ['heading_mae_deg', 'speed_relative_mae_pct',
                                              'vector_tip_relative_mae_pct']) + ' |')
    plot_paths = save_plots(rows, out, args.heading_bins, args.speed_bins,
                            heading_percent=args.plot_heading_percent,
                            show=args.show_plots) if args.plots else []
    if plot_paths:
        report.extend(['', '## Plots', ''] + [f'- `plots/{path.name}`' for path in plot_paths])
    (out / 'progress.md').write_text('\n'.join(report) + '\n')
    print('\n'.join(report))
    print(f'\nSaved report: {out / "progress.md"}')


if __name__ == '__main__':
    main()
