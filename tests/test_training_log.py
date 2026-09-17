import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr

from training_log import EventLog, training_log


class TrainingLogTests(unittest.TestCase):
    def test_timestamped_append_flush_and_console_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'training_log.jsonl'
            console = io.StringIO()
            with redirect_stdout(console):
                for run in range(2):
                    with training_log(path):
                        print(f'Run {run}')
                        self.assertIn(f'Run {run}', path.read_text())
                        print('partial', end='')
            self.assertEqual(console.getvalue(), 'Run 0\npartialRun 1\npartial')
            events = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(events), 4)
            for event in events:
                self.assertRegex(event['timestamp'], r'^\d{4}-\d\d-\d\dT.*\+00:00$')
                self.assertEqual(event['level'], 'INFO')

    def test_failure_traceback_and_stream_restoration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'training_log.jsonl'
            console = io.StringIO()
            with redirect_stderr(console):
                before = sys.stdout
                with self.assertRaisesRegex(RuntimeError, 'injected failure'):
                    with training_log(path):
                        raise RuntimeError('injected failure')
                self.assertIs(sys.stdout, before)
                self.assertIs(sys.stderr, console)
            text = path.read_text()
            self.assertIn('Training run stopped', text)
            self.assertIn('Traceback', text)
            self.assertIn('RuntimeError: injected failure', text)

    def test_failed_publish_preserves_history_and_run_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'training_log.jsonl'
            a, b = EventLog(path, 'a'), EventLog(path, 'b')
            a.append('precision_fallback', new_allow_amp=False)
            b.append('run_started')
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                a.append('bad', loss=float('nan'))
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(len(a.events('precision_fallback')), 1)
            self.assertFalse(b.events('precision_fallback'))
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_concurrent_processes_do_not_lose_events(self):
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'training_log.jsonl'
            code = ('from training_log import EventLog; import sys; '
                    'log=EventLog(sys.argv[1], sys.argv[2]); '
                    '[log.append("sample", index=i) for i in range(20)]')
            children = [subprocess.Popen([sys.executable, '-c', code, str(path), str(i)])
                        for i in range(3)]
            for child in children:
                self.assertEqual(child.wait(timeout=20), 0)
            self.assertEqual(len(EventLog(path).read()), 60)

    def test_incomplete_tail_is_recovered_without_rewriting_history(self):
        from unittest.mock import patch
        for tail in (b'{"event":', b'x' * 10000, b'\xff\xfe'):
            with self.subTest(tail_size=len(tail)), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'training_log.jsonl'
                log = EventLog(path, 'run')
                log.append('precision_fallback', new_allow_amp=False)
                before = path.read_bytes()
                inode = path.stat().st_ino
                with path.open('ab') as stream:
                    stream.write(tail)
                self.assertEqual(len(log.events('precision_fallback')), 1)
                # Appending must not parse or rewrite the complete historical log.
                with patch.object(log, 'read', side_effect=AssertionError('history read')):
                    log.append('resumed', message='line one\nline two — unicode')
                self.assertEqual(path.stat().st_ino, inode)
                self.assertTrue(path.read_bytes().startswith(before))
                events = log.events()
                self.assertEqual([e['event'] for e in events],
                                 ['precision_fallback', 'log_tail_recovered', 'resumed'])
                self.assertEqual(events[1]['discarded_bytes'], len(tail))
                self.assertEqual(events[2]['message'], 'line one\nline two — unicode')

    def test_complete_corrupt_record_is_not_silently_discarded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'training_log.jsonl'
            path.write_bytes(b'{broken}\n')
            with self.assertRaisesRegex(ValueError, 'Invalid training event'):
                EventLog(path).read()
            self.assertEqual(path.read_bytes(), b'{broken}\n')

    def test_old_json_format_is_explicitly_rejected_and_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory) / 'training_log.json'
            old.write_text('{"version": 1, "events": []}')
            with self.assertRaisesRegex(ValueError, 'Obsolete training_log.json'):
                EventLog(Path(directory) / 'training_log.jsonl')
            self.assertEqual(old.read_text(), '{"version": 1, "events": []}')


if __name__ == '__main__':
    unittest.main()
