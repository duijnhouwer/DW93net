import tempfile
import subprocess
from pathlib import Path
import unittest
from unittest.mock import patch

from train.train_flow_xt_net import committed_revision, commit_run_id, main, Settings


class TrainingEntrypointTests(unittest.TestCase):
    def test_requires_clean_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            def git(*args):
                return subprocess.check_output(['git', '-C', directory, *args], text=True).strip()
            git('init', '-q')
            path = Path(directory) / 'settings.txt'
            path.write_text('original')
            git('add', '.')
            git('-c', 'user.name=Test', '-c', 'user.email=test@example.com', 'commit', '-qm', 'initial')
            self.assertEqual(committed_revision(directory), git('rev-parse', 'HEAD'))
            self.assertRegex(commit_run_id(directory, git('rev-parse', 'HEAD')), r'^\d{8}-[0-9a-f]{40}$')
            path.write_text('changed')
            with self.assertRaises(SystemExit):
                committed_revision(directory)
            git('add', '.')
            with self.assertRaises(SystemExit):
                committed_revision(directory)
            git('-c', 'user.name=Test', '-c', 'user.email=test@example.com', 'commit', '-qm', 'update')
            (Path(directory) / 'untracked').touch()
            with self.assertRaises(SystemExit):
                committed_revision(directory)

    def test_main_uses_settings_and_commit_folder(self):
        with patch('sys.argv', ['train']), patch('train.train_flow_xt_net.os.chdir'), \
                patch('train.train_flow_xt_net.committed_revision', return_value='abc123'), \
                patch('train.train_flow_xt_net.commit_run_id', return_value='20260911-abc123'), \
                patch('train.train_flow_xt_net.train') as train:
            main()
            cfg = train.call_args.args[0]
            self.assertEqual(cfg.checkpoints_dir, str(Path(Settings.checkpoints_dir) / '20260911-abc123'))
            self.assertEqual(cfg.num_epochs, Settings.num_epochs)

    def test_rejects_cli_overrides(self):
        with patch('sys.argv', ['train', '--epochs', '10']), self.assertRaises(SystemExit):
            main()
