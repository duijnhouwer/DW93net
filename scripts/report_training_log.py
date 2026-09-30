"""Read training progress and evaluation measurements from the central log only."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
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
        lines += [f'## {cell(run)}', '', '### Epochs', '',
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path, help='Path to training_log.jsonl')
    parser.add_argument('--run-id', help='Restrict to one checkpoint-directory run ID')
    args = parser.parse_args()
    if not args.log.is_file():
        parser.error(f'Log not found: {args.log}')
    print(report(EventLog(args.log).read(), args.run_id))


if __name__ == '__main__':
    main()
