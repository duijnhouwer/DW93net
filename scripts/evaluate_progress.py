"""Reusable translation checkpoint evaluation; run with --help for options."""
import argparse
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from stimuli.load_flow_xt_data import DoGDataset
from checkpoints import load_checkpoint
from run_revision import committed_revision
from evaluation import metrics, binned_metrics, inference_batches, METRIC_DEFINITIONS
from training_log import EventLog

ROOT = Path(__file__).resolve().parents[1]



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
    parser.add_argument('--checkpoints', type=Path, required=True,
                        help='Recipe checkpoint directory containing candidates/')
    parser.add_argument('--data', type=Path, required=True,
                        help='Existing signed DoG cache root containing the selected split')
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
    parser.add_argument('--plot-heading-percent', dest='plot_heading_percent', action='store_true',
                        help='Also save heading error as a percentage of the 180° maximum')
    parser.add_argument('--output', type=Path, default=ROOT / 'evaluations')
    parser.add_argument('--log', type=Path,
                        help='Central training_log.jsonl; defaults to the run output folder')
    args = parser.parse_args()
    if (args.samples < 0 or args.batch_size < 1 or args.threads < 1
            or any(count < 1 for count in args.heading_bins + args.speed_bins)):
        parser.error('samples must be >=0; batch-size, threads, and all bin counts must be >=1')
    args.heading_bins = sorted(set(args.heading_bins))
    args.speed_bins = sorted(set(args.speed_bins))
    if args.show_plots:
        args.plots = True
    commit = committed_revision(ROOT)
    args.checkpoints = args.checkpoints.resolve()
    args.data = args.data.resolve()
    dataset = DoGDataset(args.data / args.split)
    if not len(dataset):
        parser.error('Dataset is empty')
    indices = np.random.default_rng(args.seed).permutation(len(dataset))
    indices = indices[:args.samples or len(dataset)].tolist()
    selected = [dataset.sample_ids[i] for i in indices]
    bundles = sorted({dataset.bundle_index[i][1] for i in indices})
    manifest = {'schema': 1, 'data': str(args.data), 'split': args.split,
                'seed': args.seed, 'sample_ids': selected, 'metadata': dataset.storage_metadata,
                'heading_bins': args.heading_bins, 'speed_bins': args.speed_bins,
                'bundle_stats': [(f, (args.data / args.split / f).stat().st_size,
                                  (args.data / args.split / f).stat().st_mtime_ns) for f in bundles],
                'bundle_index': [dataset.bundle_index[i] for i in indices],
                'git_commit': commit,
                'torch': torch.__version__, 'device': args.device}
    out = (args.output / args.checkpoints.parent.name / commit[:12] /
           datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True, exist_ok=True)
    log = EventLog(args.log or args.checkpoints.parent.parent / 'training_log.jsonl',
                   str(args.checkpoints))
    evaluation_id = out.name
    log.append('progress_evaluation_started', evaluation_id=evaluation_id,
               manifest=manifest, metric_definitions=METRIC_DEFINITIONS)
    torch.set_num_threads(args.threads)
    rows = []
    paths = sorted((args.checkpoints / 'candidates').glob('epoch-*.pt'))
    if args.epochs is not None:
        requested = set(args.epochs)
        paths = [path for path in paths if int(path.stem.removeprefix('epoch-')) in requested]
    if not paths:
        parser.error('No checkpoints found')
    for path in paths:
        started = time.perf_counter()
        before = path.stat()
        print(f'{path.name}: evaluating {len(indices)} {args.split} examples on {args.device}', flush=True)
        committed_revision(ROOT, commit)
        checkpoint = load_checkpoint(path, expected_commit=commit)
        predictions, targets = [], []
        for prediction, target in inference_batches(checkpoint | {'allow_tp32': False}, dataset, device=args.device,
                batch_size=args.batch_size, indices=indices):
            predictions.append(prediction[:, :2])
            targets.append(target[:, :2])
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f'Checkpoint changed during evaluation; rerun: {path}')
        row = {'checkpoint': str(path), 'epoch': checkpoint['global_epoch'],
               'saved_at': datetime.fromtimestamp(before.st_mtime, timezone.utc).isoformat(),
               'git_commit': checkpoint['git_commit'], 'evaluator_git_commit': commit,
               'training_seed': checkpoint['training_seed']}
        prediction = np.concatenate(predictions)
        target = np.concatenate(targets)
        row['metrics'] = metrics(prediction, target)
        row['breakdowns'] = {
            'heading': binned_metrics(prediction, target, dimension='heading',
                                      bin_counts=args.heading_bins),
            'speed': binned_metrics(prediction, target, dimension='speed',
                                    bin_counts=args.speed_bins),
        }
        row['evaluation_duration_s'] = time.perf_counter() - started
        row['allow_tf32'] = False
        committed_revision(ROOT, commit)
        log.append('progress_evaluation_result', evaluation_id=evaluation_id,
                   split=args.split, data=str(args.data), sample_seed=args.seed,
                   n_samples=len(indices), **row)
    log.append('progress_evaluation_completed', evaluation_id=evaluation_id,
               checkpoints_evaluated=len(paths))
    # Reports/plots are views of the central record, not separate metric stores.
    rows = [row for row in log.events('progress_evaluation_result')
            if row['evaluation_id'] == evaluation_id]
    rows.sort(key=lambda r: r['epoch'])
    report = ['# Translation progress', '',
              f'{len(indices)} fixed {args.split} examples; seed {args.seed}. Heading is primary.',
              'Speed and vector-tip errors are scene units per step. Vector-tip error is ||[dx,dz]-[dx_hat,dz_hat]||. Plot error bars are normal-approximation 95% confidence intervals of the mean. Fixed-speed results do not establish variable-speed recovery.',
              'Ordered by global epoch within this recipe run.', '',
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
