import tempfile
import subprocess
from pathlib import Path
import unittest

from train.train_flow_xt_net import committed_revision


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
