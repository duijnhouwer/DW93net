from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from recipe_helpers import module, recipe_text
from scripts.mission_control.backend.run_lock import run_lock


class RunLockTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_second_run_rejected(self):
        with run_lock():
            with self.assertRaisesRegex(RuntimeError, 'Another recipe run'), run_lock():
                pass

    def test_exception_releases_lock(self):
        with self.assertRaises(ValueError), run_lock():
            raise ValueError('simulated run failure')
        with run_lock():
            pass

    def test_inherited_lock_survives_parent_release_until_child_exits(self):
        child = None
        try:
            with run_lock() as descriptors:
                child = subprocess.Popen(
                    [sys.executable, '-c', 'import sys; sys.stdin.read()'],
                    stdin=subprocess.PIPE, pass_fds=descriptors)
            with self.assertRaises(RuntimeError), run_lock():
                pass
            child.communicate(timeout=10)
            with run_lock():
                pass
        finally:
            if child is not None:
                if child.poll() is None:
                    child.kill()
                child.communicate(timeout=10)

    def test_process_death_releases_lock(self):
        script = ('from scripts.mission_control.backend.run_lock import run_lock; import sys; '
                  'ctx=run_lock(); ctx.__enter__(); '
                  'print("locked", flush=True); sys.stdin.read()')
        child = subprocess.Popen([sys.executable, '-c', script],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                 cwd=module.ROOT)
        try:
            self.assertEqual(child.stdout.readline().strip(), 'locked')
            with self.assertRaises(RuntimeError), run_lock():
                pass
        finally:
            child.kill()
            child.communicate(timeout=10)
        with run_lock():
            pass

    def test_runner_locks_before_mutating(self):
        path = self.root / 'recipe.toml'
        path.write_text(recipe_text(self.root) + '\n[parallel_generation]\nenabled = true\n')
        with run_lock(), patch.object(module, '_run_recipe') as run, \
                patch.object(module, 'Prefetch') as prefetch:
            with self.assertRaisesRegex(RuntimeError, 'Another recipe run'):
                module.run_recipe(path, require_clean=False)
            run.assert_not_called()
            prefetch.assert_not_called()

    def test_runner_cleanup_occurs_before_release(self):
        (self.root / 'output').mkdir()
        path = self.root / 'recipe.toml'
        path.write_text(recipe_text(self.root))

        def check_locked():
            with self.assertRaises(RuntimeError), run_lock():
                pass

        with patch.object(module, '_run_recipe', side_effect=ValueError('failed')), \
                patch.object(module, 'Prefetch') as prefetch:
            prefetch.return_value.close.side_effect = check_locked
            with self.assertRaisesRegex(ValueError, 'failed'):
                module.run_recipe(path, require_clean=False)
            prefetch.return_value.close.assert_called_once()
            self.assertTrue(prefetch.call_args.kwargs['lock_fds'])
        with run_lock():
            pass
