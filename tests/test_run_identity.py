"""Run names resolve the exact commit and its committer time, not wall-clock time."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from run_revision import run_directory_name
from scripts.mission_control.mission_control import _mission_plan_info


class RunIdentityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.env = os.environ | {'GIT_AUTHOR_DATE': '2000-01-01T00:00:00+00:00',
                                'GIT_COMMITTER_DATE': '2026-09-30T09:15:23-04:00'}
        self.git('init', '-q')
        (self.root / 'scripts/mission_control/mission_plan').mkdir(parents=True)
        self.mission_plan = self.root / 'scripts/mission_control/mission_plan/trans_rot.toml'
        self.mission_plan.write_text('[network]\noutput_path = "' + str(self.root / 'output') + '"\n')
        self.git('add', '.')
        self.git('-c', 'user.name=Test', '-c', 'user.email=test@example.com', 'commit', '-qm', 'first')
        self.commit = self.git('rev-parse', 'HEAD')

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.root), *args],
                                       env=self.env, text=True).strip()

    def test_full_hash_and_utc_committer_timestamp(self):
        expected = f'trans_rot-20260930131523-{self.commit}'
        with patch.dict(os.environ, {'TZ': 'Pacific/Honolulu'}):
            self.assertEqual(run_directory_name('trans_rot', self.commit, root=self.root), expected)
        self.assertEqual(run_directory_name('trans_rot', root=self.root), expected)
        _, _, network = _mission_plan_info(self.root)
        self.assertEqual(network, self.root / 'output' / expected)

    def test_new_commit_is_distinct_and_old_revision_is_stable(self):
        original = run_directory_name('trans_rot', self.commit, root=self.root)
        self.git('-c', 'user.name=Test', '-c', 'user.email=test@example.com',
                 'commit', '--allow-empty', '-qm', 'second')
        self.assertNotEqual(run_directory_name('trans_rot', root=self.root), original)
        self.assertEqual(run_directory_name('trans_rot', self.commit, root=self.root), original)
        self.assertNotEqual(run_directory_name('other_mission_plan', self.commit, root=self.root), original)

    def test_rejects_unresolvable_and_obsolete_commit_identifiers(self):
        for commit in ('HEAD', self.commit[:12], '../bad'):
            with self.assertRaisesRegex(ValueError, 'full 40-character'):
                run_directory_name('trans_rot', commit, root=self.root)
        with self.assertRaisesRegex(ValueError, 'Cannot read timestamp'):
            run_directory_name('trans_rot', '0' * 40, root=self.root)
