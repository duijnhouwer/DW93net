import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import unquote, urlparse

from scripts.mission_control.backend.report_training_log import dashboard_data, write_dashboard, show_dashboard


class DashboardTests(unittest.TestCase):
    def events(self):
        common = dict(version=1, run_id='/output/panorama/checkpoints',
                      timestamp='2026-09-30T12:00:00+00:00')
        return [common | dict(event='epoch_completed', global_epoch=1, training_loss=.2,
                              epoch_wall_duration_s=150),
                common | dict(event='validation_result', result=dict(global_epoch=1,
                    validation_loss=.3, heading_mae_deg=2, translation_speed_mae=.001,
                    speed_relative_mae_pct=3, rotation_mae_deg=.004,
                    translation_vector_tip_mae=.002, vector_tip_relative_mae_pct=4,
                    n_samples=25, dataset={'path': '/validation', 'generation': {'seed': 2000}}))]

    def test_validation_metrics_and_timings_are_preserved(self):
        run = dashboard_data(self.events())[0]
        self.assertEqual(run['training'][0]['epoch_wall_duration_s'], 150)
        metrics = run['validation'][0]['rows'][0]
        self.assertEqual(metrics['rotation_mae_deg'], .004)
        self.assertEqual(metrics['vector_tip_relative_mae_pct'], 4)
        self.assertEqual(metrics['heading_mae_deg'], 2)

    def test_separate_datasets_deduplicate_epochs_and_ignore_test_results(self):
        events = self.events()
        changed = events[1] | {'result': events[1]['result'] | {'validation_loss': .1}}
        other = events[1] | {'result': events[1]['result'] | {'dataset': {'path': '/different'}}}
        events += [changed, other, events[1] | {'event': 'test_result'}]
        run = dashboard_data(events)[0]
        self.assertEqual(len(run['validation']), 2)
        self.assertEqual(len(run['validation'][0]['rows']), 1)
        self.assertEqual(run['validation'][0]['rows'][0]['validation_loss'], .1)

    def test_missing_and_nonfinite_values_are_gaps_and_runs_are_filtered(self):
        events = self.events()
        events[1]['result']['rotation_mae_deg'] = float('nan')
        events.append(events[0] | {'run_id': '/other/checkpoints'})
        runs = dashboard_data(events, '/output/panorama/checkpoints')
        self.assertEqual(len(runs), 1)
        self.assertIsNone(runs[0]['validation'][0]['rows'][0]['rotation_mae_deg'])
        self.assertIsNone(runs[0]['training'][0]['validation_duration_s'])
        with self.assertRaisesRegex(ValueError, 'No training log records'):
            dashboard_data(events, '/missing')

    def test_html_escapes_log_text_and_preserves_input(self):
        events = self.events()
        events[1]['result']['dataset']['path'] = '</script><script>alert(1)</script>'
        with tempfile.TemporaryDirectory() as directory:
            path = write_dashboard(events, Path(directory) / 'report.html')
            html = path.read_text()
            self.assertNotIn('<script>alert(1)</script>', html)
            self.assertNotIn('__TRAINING_DATA__', html)
            data = html.split('<script id="training-data" type="application/json">')[1].split('</script>')[0]
            self.assertEqual(json.loads(data)[0]['validation'][0]['rows'][0]['validation_loss'], .3)
            self.assertIn('</script><script>alert(1)</script>', json.loads(data)[0]['validation'][0]['label'])

    def test_cli_defaults_to_text_without_creating_files(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'training_log.jsonl'
            log.write_text(''.join(json.dumps(e) + '\n' for e in self.events()))
            before = log.read_bytes()
            script = Path(__file__).resolve().parents[1] / 'scripts/mission_control/backend/report_training_log.py'
            result = subprocess.run([sys.executable, str(script), str(log)],
                                    check=True, capture_output=True, text=True)
            self.assertIn('# Training log progress', result.stdout)
            self.assertIn('Best checkpoint: epoch 1', result.stdout)
            self.assertIn('wall duration: 150 s', result.stdout)
            self.assertEqual(list(Path(directory).iterdir()), [log])
            self.assertEqual(log.read_bytes(), before)

    def test_gui_temporary_files_exist_until_viewer_exits_then_are_removed(self):
        for failure in (None, subprocess.CalledProcessError(1, 'gjs'), KeyboardInterrupt()):
            paths = []

            def viewer(command, **kwargs):
                path = Path(unquote(urlparse(command[-1]).path))
                paths.append(path)
                self.assertEqual(path.name, 'dw92net_trainging_dashboard.html')
                self.assertTrue(path.is_file())
                self.assertIn('Training progress', path.read_text())
                self.assertTrue(kwargs['check'])
                if failure:
                    raise failure

            with patch('shutil.which', return_value='/usr/bin/gjs'), patch('subprocess.run', side_effect=viewer):
                if isinstance(failure, KeyboardInterrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        show_dashboard(self.events())
                elif failure:
                    with self.assertRaisesRegex(RuntimeError, 'temporary files were removed'):
                        show_dashboard(self.events())
                else:
                    show_dashboard(self.events())
            self.assertFalse(paths[0].exists())
            self.assertFalse(paths[0].parent.exists())

    def test_gui_flag_is_explicit_and_old_output_options_are_rejected(self):
        from scripts.mission_control.backend import report_training_log as script
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'training_log.jsonl'
            log.write_text(''.join(json.dumps(e) + '\n' for e in self.events()))
            with patch('sys.argv', ['report', str(log), '--dashboard-gui']), \
                    patch.object(script, 'show_dashboard') as show:
                script.main()
                show.assert_called_once()
            with patch('sys.argv', ['report', str(log), '--output', str(Path(directory) / 'out.html')]):
                with self.assertRaises(SystemExit) as error:
                    script.main()
                self.assertEqual(error.exception.code, 2)
