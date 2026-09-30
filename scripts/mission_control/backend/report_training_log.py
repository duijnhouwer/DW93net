"""Read training progress and evaluation measurements from the central log only."""
import argparse
from datetime import datetime
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
from training_log import EventLog


def report(events, run_id=None):
    def cell(value):
        if value is None:
            return 'not recorded / undefined'
        if isinstance(value, float):
            return f'{value:.6g}'
        return str(value).replace('|', '\\|').replace('\n', ' ')

    lines = ['# Training log progress', '',
             'Recorded measurements only; no checkpoint loading or inference.',
             'Compare accuracy only on matching splits and sample selections. '
             'Speed/vector percentages are not bounded by 100%.', '']
    runs = sorted({e['run_id'] for e in events if e.get('run_id') is not None})
    if run_id is not None:
        if run_id not in runs:
            raise ValueError(f'Run not present in log: {run_id}')
        runs = [run_id]
    for run in runs:
        records = [e for e in events if e.get('run_id') == run]
        lines += [f'## {cell(run)}', '']
        summary = dashboard_data(records, run)[0]
        completed = summary['training']
        lines.append(f'Completed epochs recorded: {len(completed)}')
        if completed:
            latest = completed[-1]
            lines.append(f'Latest completed epoch: {latest["epoch"]}; '
                         f'wall duration: {cell(latest["epoch_wall_duration_s"])} s')
        for group in summary['validation']:
            measured = [row for row in group['rows'] if row['validation_loss'] is not None]
            if not measured:
                continue
            best = min(measured, key=lambda row: row['validation_loss'])
            lines += ['', cell(group['label']),
                      f'Best checkpoint: epoch {best["epoch"]}; validation loss: {cell(best["validation_loss"])}',
                      f'Heading MAE: {cell(best["heading_mae_deg"])} deg; '
                      f'speed MAE: {cell(best["speed_relative_mae_pct"])}%; '
                      f'yaw MAE: {cell(best["rotation_mae_deg"])} deg/step; '
                      f'dx/dz vector-tip MAE: {cell(best["vector_tip_relative_mae_pct"])}%']
        lines += ['', '### Epochs', '',
                  '| Completed UTC | Epoch | Attempt | Training s | Validation s | Wall s incl. retries/save | Save s | Loader wait s | Samples/s | Training loss | Validation loss | Next LR |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for e in records:
            if e['event'] == 'epoch_completed':
                keys = ('timestamp', 'global_epoch', 'attempt', 'training_duration_s', 'validation_duration_s',
                        'epoch_wall_duration_s', 'checkpoint_duration_s', 'loader_wait_s',
                        'samples_per_second', 'training_loss', 'validation_loss', 'learning_rate')
                lines.append('| ' + ' | '.join(cell(e.get(k)) for k in keys) + ' |')
        failures = sum(e['event'] == 'numerical_failure' for e in records)
        fallbacks = sum(e['event'] == 'precision_fallback' for e in records)
        lines += ['', f'Numerical failures: {failures}; precision fallbacks: {fallbacks}.', '',
                  '### Evaluations', '',
                  '| Recorded UTC | Split / selection | Epoch | N | Eval s | Heading MAE ° | Heading % of 180° | Cosine similarity | Cosine error % | Within 15° % | Speed MAE % | Vector-tip MAE % | Undefined headings |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for e in records:
            if e['event'] == 'validation_result':
                m = e['result']
                split = 'validate'
            elif e['event'] in ('test_result', 'duffy_wurtz_1993_result'):
                m = e['metrics']
                split = 'test' if e['event'] == 'test_result' else 'duffy_wurtz_1993'
            elif e['event'] == 'progress_evaluation_result':
                m = e['metrics']
                selection = (f'{e["split"]} / session {e["evaluation_id"]} / '
                             f'subset seed {e["sample_seed"]} / {e["data"]}')
            else:
                continue
            if e['event'] != 'progress_evaluation_result':
                dataset = m.get('dataset', {})
                selection = (f'{split} / all / generation seed '
                             f'{dataset.get("generation", {}).get("seed", "unknown")} / '
                             f'{dataset.get("path", "unknown dataset")}')
            values = [e.get('timestamp'), selection, m.get('global_epoch', e.get('epoch')),
                      m.get('n_samples', m.get('n')), m.get('evaluation_duration_s', e.get('evaluation_duration_s'))]
            values += [m.get(k) for k in ('heading_mae_deg', 'heading_relative_mae_pct',
                       'heading_cosine_similarity', 'heading_cosine_error_pct', 'heading_within_15_pct',
                       'speed_relative_mae_pct', 'vector_tip_relative_mae_pct', 'undefined_heading_count')]
            lines.append('| ' + ' | '.join(cell(value) for value in values) + ' |')
        lines += ['']
    return '\n'.join(lines)


def dashboard_data(events, run_id=None):
    """Collect latest recorded values per epoch; keep validation datasets separate."""
    runs = {}

    def finite(value):
        return value if isinstance(value, (int, float)) and math.isfinite(value) else None

    def timestamp(event):
        try:
            return datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00')).timestamp()
        except (KeyError, ValueError, TypeError):
            return None

    for event in events:
        identity = event.get('run_id')
        if not identity or (run_id is not None and identity != run_id):
            continue
        run = runs.setdefault(identity, dict(id=identity, name=Path(identity).parent.name,
                                            training={}, validation={}, start=None, latest=None))
        time = timestamp(event)
        if time is not None:
            run['start'] = min(run['start'], time) if run['start'] is not None else time
            run['latest'] = max(run['latest'], time) if run['latest'] is not None else time
        kind = event['event']
        if kind not in ('epoch_completed', 'validation_result'):
            continue
        metrics = event.get('result', {}) if kind == 'validation_result' else event
        epoch = metrics.get('global_epoch')
        if not isinstance(epoch, int):
            continue
        row = dict(epoch=epoch, time=time)
        if kind == 'epoch_completed':
            for key in ('training_loss', 'training_duration_s', 'validation_duration_s',
                        'epoch_wall_duration_s', 'learning_rate'):
                row[key] = finite(metrics.get(key))
            run['training'][epoch] = row
        else:
            dataset = metrics.get('dataset', {})
            signature = json.dumps(dict(dataset=dataset, n_samples=metrics.get('n_samples')),
                                   sort_keys=True)
            generation = dataset.get('generation', {})
            label = (f"Validation · seed {generation.get('seed', 'unknown')} · "
                     f"{metrics.get('n_samples', '?')} samples · {dataset.get('path', 'dataset not recorded')}")
            group = run['validation'].setdefault(signature, dict(label=label, rows={}))
            for key in ('validation_loss', 'heading_mae_deg', 'translation_speed_mae',
                        'speed_relative_mae_pct', 'rotation_mae_deg',
                        'translation_vector_tip_mae', 'vector_tip_relative_mae_pct'):
                row[key] = finite(metrics.get(key))
            if row['validation_loss'] is None:
                row['validation_loss'] = finite(metrics.get('normalized_mse'))
            group['rows'][epoch] = row
    if run_id is not None and run_id not in runs:
        raise ValueError(f'No training log records for {run_id}')
    for run in runs.values():
        run['training'] = sorted(run['training'].values(), key=lambda row: row['epoch'])
        run['validation'] = [dict(label=group['label'], rows=sorted(
            group['rows'].values(), key=lambda row: row['epoch']))
            for group in run['validation'].values()]
    return list(runs.values())


def write_dashboard(events, destination, run_id=None):
    runs = dashboard_data(events, run_id)
    if not runs:
        raise ValueError('No training log records available')
    # Escape markup characters so paths and other log text cannot become HTML/script.
    payload = json.dumps(runs, allow_nan=False).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    template = Path(__file__).with_name('report_dashboard.html').read_text()
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(template.replace('__TRAINING_DATA__', payload))
    return destination


def show_dashboard(events, run_id=None):
    """Own the viewer lifetime and remove its temporary HTML when it exits."""
    viewer = shutil.which('gjs')
    if viewer is None:
        raise RuntimeError('Dashboard GUI requires GJS, GTK 3 and WebKit2 4.1. '
                           'Run without --dashboard-gui for the terminal report.')
    with tempfile.TemporaryDirectory(prefix='dw93net-dashboard-') as temporary:
        destination = write_dashboard(events, Path(temporary) / 'dw92net_trainging_dashboard.html', run_id)
        print('Opening training dashboard. Close its window to return; temporary files will be deleted.', flush=True)
        try:
            subprocess.run([viewer, str(Path(__file__).with_name('dashboard_window.js')),
                            destination.as_uri()], check=True)
        except (subprocess.CalledProcessError, OSError) as error:
            raise RuntimeError('Dashboard window could not run. It requires a graphical session, '
                               'GJS, GTK 3 and WebKit2 4.1; temporary files were removed.') from error


def main():
    parser = argparse.ArgumentParser(description='Print recorded training progress and evaluation measurements.')
    parser.add_argument('log', type=Path, help='Path to training_log.jsonl')
    parser.add_argument('--run-id', help='Restrict to one checkpoint-directory run ID')
    parser.add_argument('--dashboard-gui', action='store_true',
                        help='Open a temporary graphical dashboard; delete its files when the window closes')
    args = parser.parse_args()
    if not args.log.is_file():
        parser.error(f'Log not found: {args.log}')
    events = EventLog(args.log).read()
    try:
        if args.dashboard_gui:
            show_dashboard(events, args.run_id)
        else:
            print(report(events, args.run_id))
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nReport closed.', file=sys.stderr)
