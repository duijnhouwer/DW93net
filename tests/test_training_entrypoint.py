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
            original_commit = committed_revision(directory)
            self.assertEqual(committed_revision(directory, original_commit), original_commit)
            path.write_text('changed')
            with self.assertRaises(SystemExit):
                committed_revision(directory)
            git('add', '.')
            with self.assertRaises(SystemExit):
                committed_revision(directory)
            git('-c', 'user.name=Test', '-c', 'user.email=test@example.com', 'commit', '-qm', 'update')
            with self.assertRaisesRegex(SystemExit, 'commit changed'):
                committed_revision(directory, original_commit)
            (Path(directory) / 'untracked').touch()
            with self.assertRaises(SystemExit):
                committed_revision(directory)

    def test_generation_and_prefetch_refuse_changed_repository_before_writing(self):
        from unittest.mock import patch
        from test_training_recipe import module, recipe_text, DETECTOR
        from scripts.mission_control.backend.parallel_stimuli import Prefetch
        with tempfile.TemporaryDirectory() as directory:
            recipe_path = Path(directory) / 'recipe.toml'
            recipe_path.write_text(recipe_text(directory) + '\n[parallel_generation]\nenabled = true\n')
            recipe = module.load_recipe(recipe_path)
            section = dict(seed=4000, n_samples=2, _detector=DETECTOR,
                           _repo_root=directory, _git_commit='a' * 40)
            destination = Path(directory) / 'data'
            with patch.object(module, 'committed_revision', side_effect=SystemExit('ERROR')):
                with self.assertRaises(SystemExit):
                    module._prepare_dataset(destination, 'train', section, recipe['stimuli'], persistent=False)
            self.assertFalse(destination.exists())
            prefetch = Prefetch(recipe)
            with patch('run_revision.committed_revision', side_effect=SystemExit('ERROR')):
                with self.assertRaises(SystemExit):
                    prefetch.start(section)
            self.assertIsNone(prefetch.process)
            self.assertIsNone(prefetch.temporary)

    def test_edit_during_training_stops_before_next_set_and_preserves_checkpoint(self):
        from unittest.mock import patch
        from test_training_recipe import module, recipe_text
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'repo'
            (root / 'recipe').mkdir(parents=True)
            (Path(directory) / 'output').mkdir()
            recipe_path = root / 'recipe' / 'smoke.toml'
            recipe_path.write_text(recipe_text(directory))
            readme = root / 'README.md'
            readme.write_text('original')
            def git(*args):
                return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
            git('init', '-q')
            git('add', '.')
            git('-c', 'user.name=Test', '-c', 'user.email=test@example.com', 'commit', '-qm', 'initial')
            commit = git('rev-parse', 'HEAD')
            latest = Path(directory) / 'output' / module.run_directory_name('smoke', commit, root=root) / 'checkpoints' / 'latest.pt'
            def train(cfg, **kwargs):
                latest.write_bytes(b'last completed checkpoint')
                readme.write_text('edited during training')
            with patch.object(module, 'capture_provenance', return_value={'git_commit': commit}), \
                    patch.object(module, '_prepare_dataset', return_value=Path(directory) / 'cache') as generate, \
                    patch.object(module, 'train_recipe_set', side_effect=train), \
                    patch.object(module, '_select_final') as select:
                with self.assertRaisesRegex(SystemExit, 'ERROR: Repository changed'):
                    module.run_recipe(recipe_path, root=root)
            self.assertEqual(generate.call_count, 3)
            select.assert_not_called()
            self.assertEqual(latest.read_bytes(), b'last completed checkpoint')
