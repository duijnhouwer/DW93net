import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

spec = importlib.util.spec_from_file_location(
    'run_training_recipe',
    Path(__file__).resolve().parents[1] / 'scripts/run_training_recipe.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

DETECTOR = {'displacement_min_px': -60, 'displacement_max_px': 60,
            'displacement_step_px': 1, 'dog_sigma_center_px': 1.0,
            'dog_sigma_surround_px': 3.0}


def recipe_text(root):
    return f'''
version = 1
require_cuda = false
[network]
output_path = "{root}/output"
[stimuli]
speed_range = [0.001, 0.006]
heading_deg_range = [-180, 180]
rot_deg_range = [0, 0]
dog_bundle_size = 64
[detector]
displacement_min_px = -60
displacement_max_px = 60
displacement_step_px = 1
dog_sigma_center_px = 1.0
dog_sigma_surround_px = 3.0
[training_sets]
path = "{root}/stimuli/training"
number_of_sets = 3
samples_per_set = 5000
epochs_per_set = 10
start_seed = 1000
seed_increment = 1
shuffle_seed = 7000
[validation_set]
path = "{root}/stimuli/validation"
seed = 2000
n_samples = 1000
keep_after_run = false
selection_metric = "normalized_rmse"
[test_set]
path = "{root}/stimuli/test"
seed = 3000
n_samples = 1000
[optimization]
batch_size = 64
mixed_precision = true
profile_batches = 20
num_workers = 0
mirror_train_stimuli = true
[learning_rate]
initial = 0.001
scheduler = "reduce_on_plateau"
monitor = "training_loss"
factor = 0.5
patience_epochs = 10
minimum = 0.000001
relative_improvement_threshold = 0.0001
reset_on_new_training_set = false
[checkpoints]
candidate_interval_epochs = 1
keep_previous_resumable = true
[evaluation]
batch_size = 64
num_workers = 0
shuffle = false
'''


class TrainingRecipeTests(unittest.TestCase):
    def test_loads_schema_and_rejects_rotation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'translation_only.toml'
            path.write_text(recipe_text(directory))
            recipe = module.load_recipe(path)
            self.assertEqual(recipe['training_sets']['number_of_sets'], 3)
            self.assertEqual(recipe['stimuli']['dog_bundle_size'], 64)
            path.write_text(recipe_text(directory).replace(
                'rot_deg_range = [0, 0]', 'rot_deg_range = [-0.5, 0.5]'))
            with self.assertRaisesRegex(ValueError, 'rotation is disabled'):
                module.load_recipe(path)

    def test_rejects_relative_or_nested_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'translation_only.toml'
            path.write_text(recipe_text(directory).replace(
                f'{directory}/stimuli/test', f'{directory}/stimuli/training/test'))
            with self.assertRaisesRegex(ValueError, 'separate, non-nested'):
                module.load_recipe(path)
            path.write_text(recipe_text(directory).replace(f'{directory}/output', 'relative'))
            with self.assertRaisesRegex(ValueError, 'absolute path'):
                module.load_recipe(path)

    def test_output_uses_empty_full_commit_marker_and_refuses_mismatch(self):
        commit = 'a' * 40
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'output'
            output.mkdir()
            recipe = {'network': {'output_path': str(output)}}
            checkpoints, final = module._initialize_output(recipe, 'translation_only', commit)
            marker = output / 'translation_only' / f'commit_id={commit}'
            self.assertTrue(marker.is_file())
            self.assertEqual(marker.stat().st_size, 0)
            self.assertTrue(checkpoints.is_dir())
            self.assertTrue(final.is_dir())
            marker.rename(marker.with_name('commit_id=' + 'b' * 40))
            with self.assertRaisesRegex(ValueError, 'does not belong'):
                module._initialize_output(recipe, 'translation_only', commit)

    def test_generation_deletes_raw_and_reuses_exact_spec(self):
        stimuli = {'speed_range': [0.001, 0.006], 'heading_deg_range': [-180, 180],
                   'rot_deg_range': [0, 0], 'dog_bundle_size': 64}
        section = {'seed': 10, 'n_samples': 12, '_detector': DETECTOR}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'training'
            raw = path / 'parameters' / 'raw'
            cache = path / 'parameters' / 'dog'

            def generate(**kwargs):
                (cache / 'train').mkdir(parents=True)
                (cache / 'dog.json').write_text('{}')
                (cache / 'train' / 'bundles.json').write_text('[]')
                return str(cache)

            with patch.object(module, 'generate_dog_dataset', side_effect=generate) as mocked:
                self.assertEqual(module._prepare_dataset(
                    path, 'train', section, stimuli, persistent=False), cache)
                self.assertFalse(raw.exists())
                self.assertEqual(module._prepare_dataset(
                    path, 'train', section, stimuli, persistent=False), cache)
            self.assertEqual(mocked.call_count, 1)

    def test_persistent_test_set_is_never_replaced_on_mismatch(self):
        stimuli = {'speed_range': [0.001, 0.006], 'heading_deg_range': [-180, 180],
                   'rot_deg_range': [0, 0], 'dog_bundle_size': 64}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test'
            path.mkdir()
            (path / module.MANAGED_MARKER).touch()
            (path / module.DATASET_STATE).write_text(json.dumps({
                'spec': {'wrong': True}, 'cache': 'missing'}))
            with self.assertRaisesRegex(ValueError, 'differs from the recipe'):
                module._prepare_dataset(path, 'test', {'seed': 1, 'n_samples': 10,
                                                       '_detector': DETECTOR},
                                        stimuli, persistent=True)
            self.assertTrue(path.exists())

    def test_managed_deletion_refuses_unmarked_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data'
            path.mkdir()
            with self.assertRaisesRegex(ValueError, 'unmanaged'):
                module._safe_delete_managed(path)
            (path / module.MANAGED_MARKER).touch()
            module._safe_delete_managed(path)
            self.assertFalse(path.exists())

    def test_candidate_validation_resumes_after_last_saved_result(self):
        commit = 'a' * 40
        metric = {'n_samples': 2, 'translation_speed_mae': 0.1,
                  'translation_vector_tip_mae': 0.2, 'rotation_mae_deg': 0.0,
                  'heading_mae_deg': 1.0}
        recipe = {'evaluation': {'batch_size': 2, 'num_workers': 0, 'shuffle': False}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / 'checkpoints' / 'candidates'
            final = root / 'final'
            candidates.mkdir(parents=True)
            final.mkdir()
            for epoch in range(1, 4):
                torch.save({'git_commit': commit, 'model_state_dict': {},
                            'global_epoch': epoch}, candidates / f'epoch-{epoch:06d}.pt')

            first = {**metric, 'normalized_rmse': 3.0}
            with patch.object(module, 'evaluate_model',
                              side_effect=[first, RuntimeError('interrupted')]) as evaluate:
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    module._select_final(root / 'checkpoints', final, root / 'validation',
                                         recipe, commit)
                self.assertEqual(evaluate.call_count, 2)
            progress = json.loads((final / 'validation-progress.json').read_text())
            self.assertFalse(progress['complete'])
            self.assertEqual([row['candidate'] for row in progress['candidates']],
                             ['epoch-000001.pt'])

            remaining = [{**metric, 'normalized_rmse': 2.0},
                         {**metric, 'normalized_rmse': 1.0}]
            with patch.object(module, 'evaluate_model', side_effect=remaining) as evaluate:
                promoted = module._select_final(root / 'checkpoints', final,
                                                root / 'validation', recipe, commit)
                self.assertEqual(evaluate.call_count, 2)
            self.assertTrue(promoted)
            progress = json.loads((final / 'validation-progress.json').read_text())
            self.assertTrue(progress['complete'])
            self.assertEqual(progress['selected'], 'epoch-000003.pt')
            results = json.loads((final / 'validation-results.json').read_text())
            self.assertEqual(len(results['candidates']), 3)
            self.assertNotIn('checkpoint_sha256', results['candidates'][0])

    def test_tiny_end_to_end_run_replaces_training_sets_and_selects_final(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'output').mkdir()
            (root / 'recipe').mkdir()
            path = root / 'recipe' / 'smoke.toml'
            text = recipe_text(directory)
            text = text.replace('number_of_sets = 3', 'number_of_sets = 2')
            text = text.replace('samples_per_set = 5000', 'samples_per_set = 2')
            text = text.replace('epochs_per_set = 10', 'epochs_per_set = 1')
            text = text.replace('n_samples = 1000', 'n_samples = 2')
            text = text.replace('dog_bundle_size = 64', '''
dog_bundle_size = 2
fov_deg = 3
n_world_dots = 20
x_supersampling = 4''')
            text = text.replace('batch_size = 64', 'batch_size = 1')
            text = text.replace('profile_batches = 20', 'profile_batches = 0')
            text = text.replace('mirror_train_stimuli = true',
                                'mirror_train_stimuli = false')
            path.write_text(text)
            module.run_recipe(path, root=root, commit='c' * 40)
            network = root / 'output' / 'smoke'
            self.assertTrue((network / ('commit_id=' + 'c' * 40)).is_file())
            self.assertEqual(len(list((network / 'checkpoints' / 'candidates').glob('*.pt'))), 2)
            self.assertTrue((network / 'final' / 'model.pt').is_file())
            self.assertTrue((network / 'final' / 'validation-results.json').is_file())
            self.assertTrue((network / 'final' / 'test-results.json').is_file())
            self.assertFalse((root / 'stimuli' / 'training').exists())
            self.assertFalse((root / 'stimuli' / 'validation').exists())
            self.assertTrue((root / 'stimuli' / 'test' / module.DATASET_STATE).is_file())
            progress_spec = importlib.util.spec_from_file_location(
                'evaluate_progress', module.ROOT / 'scripts' / 'evaluate_progress.py')
            progress = importlib.util.module_from_spec(progress_spec)
            progress_spec.loader.exec_module(progress)
            test_path = root / 'stimuli' / 'test'
            state = json.loads((test_path / module.DATASET_STATE).read_text())
            report_path = root / 'reports'
            args = ['evaluate_progress', '--checkpoints', str(network / 'checkpoints'),
                    '--data', str(test_path / state['cache']), '--split', 'test',
                    '--samples', '0', '--no-plots', '--output', str(report_path)]
            with patch('sys.argv', args), patch.object(
                    progress.subprocess, 'check_output', return_value='c' * 40):
                progress.main()
            report = next(report_path.glob('*/progress.md')).read_text()
            self.assertIn('epoch-000001.pt', report)
            self.assertIn('epoch-000002.pt', report)


if __name__ == '__main__':
    unittest.main()
