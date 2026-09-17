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
start_seed = 4000
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


def candidate_checkpoint(commit, epoch=1):
    from checkpoints import SCHEMA, compatibility
    return dict(schema=SCHEMA, kind='candidate', compatibility=compatibility(),
                data_contract={}, git_commit=commit, global_epoch=epoch,
                training_set_index=0, training_seed=1, epoch_within_set=epoch,
                epochs_per_set=epoch, training_loss=.1, learning_rate=.001,
                detector={}, n_input_channels=3, model_state_dict={})


class TrainingRecipeTests(unittest.TestCase):
    def test_speed_bounds_must_be_representable_in_float32(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'recipe.toml'
            for bounds in ('[1e-12, 2e-12]', '[1e-12, 1e-12]', '[1e-50, 1e-40]', '[1e38, 1e40]'):
                path.write_text(recipe_text(directory).replace(
                    'speed_range = [0.001, 0.006]', f'speed_range = {bounds}'))
                with self.subTest(bounds=bounds):
                    if '1e-12' in bounds:
                        module.load_recipe(path)
                    else:
                        with self.assertRaisesRegex(ValueError, 'float32'):
                            module.load_recipe(path)

    def test_near_exclusion_radius_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'recipe.toml'
            for radius in ('0', '0.04', '0.999999', '-0.01', '1', '1.1', 'nan', 'inf', '-inf', 'true'):
                with self.subTest(radius=radius):
                    path.write_text(recipe_text(directory).replace(
                        '[stimuli]', f'[stimuli]\nnear_exclusion_radius = {radius}'))
                    if radius in ('0', '0.04', '0.999999'):
                        module.load_recipe(path)
                    else:
                        with self.assertRaisesRegex(ValueError, 'near_exclusion_radius'):
                            module.load_recipe(path)

    def test_fov_must_fit_full_circle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'recipe.toml'
            original = recipe_text(directory)
            for fov in ('0.001', '120', '240', '360', '0', '-1', '360.001', 'nan', 'inf', 'true'):
                with self.subTest(fov=fov):
                    path.write_text(original.replace('[stimuli]', f'[stimuli]\nfov_deg = {fov}'))
                    if fov in ('0.001', '120', '240', '360'):
                        module.load_recipe(path)
                    else:
                        with self.assertRaisesRegex(ValueError, 'fov_deg'):
                            module.load_recipe(path)

    def test_training_seeds_must_start_above_distinct_held_out_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'recipe.toml'
            original = recipe_text(directory)
            for start in (0, 1999, 2000, 2500, 3000):
                with self.subTest(start=start):
                    path.write_text(original.replace('start_seed = 4000', f'start_seed = {start}'))
                    with self.assertRaisesRegex(ValueError, 'greater than both'):
                        module.load_recipe(path)
            path.write_text(original.replace('seed = 3000', 'seed = 2000'))
            with self.assertRaisesRegex(ValueError, 'must differ'):
                module.load_recipe(path)
            # Both held-out seed orderings are valid, even with many training sets.
            for validation, test in ((2000, 3000), (3000, 2000)):
                before_test, test_section = original.split('[test_set]', 1)
                text = (before_test.replace('seed = 2000', f'seed = {validation}')
                        + '[test_set]' + test_section.replace('seed = 3000', f'seed = {test}'))
                text = text.replace('start_seed = 4000', 'start_seed = 3001')
                text = text.replace('number_of_sets = 3', 'number_of_sets = 1000000')
                path.write_text(text)
                module.load_recipe(path)
            for increment in (0, -1):
                path.write_text(original.replace('seed_increment = 1', f'seed_increment = {increment}'))
                with self.assertRaisesRegex(ValueError, 'seed_increment must be a positive integer'):
                    module.load_recipe(path)

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

    def test_output_uses_compatibility_not_commit_equality(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'output'
            output.mkdir()
            recipe = {'network': {'output_path': str(output)}}
            checkpoints, final = module._initialize_output(recipe, 'translation_only', 'a' * 40)
            self.assertEqual(module._initialize_output(recipe, 'translation_only', 'b' * 40),
                             (checkpoints, final))
            manifest = json.loads((checkpoints.parent / 'run.json').read_text())
            self.assertEqual(manifest['origin_git_commit'], 'a' * 40)
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                module._initialize_output(recipe | {'changed': True}, 'translation_only', 'b' * 40)

    def test_generation_writes_only_dog_cache_and_reuses_exact_spec(self):
        stimuli = {'speed_range': [0.001, 0.006], 'heading_deg_range': [-180, 180],
                   'rot_deg_range': [0, 0], 'dog_bundle_size': 64}
        section = {'seed': 10, 'n_samples': 12, '_detector': DETECTOR}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'training'
            cache = path / 'parameters' / 'dog'

            def generate(**kwargs):
                import numpy as np
                (cache / 'train').mkdir(parents=True)
                (cache / 'dog.json').write_text(json.dumps(dict(
                    storage_format='dog-contrast-bundles-v2', storage_dtype='torch.float32',
                    finite_validated=True, n_samples=12, contrast_shape=[2, 31])))
                (cache / 'train' / 'bundles.json').write_text(json.dumps(
                    [[i, 'bundle-000000000.npy', i] for i in range(12)]))
                np.save(cache / 'train' / 'bundle-000000000.npy', np.zeros(12, dtype=[
                    ('dog_contrast', np.float32, (2, 31)), ('target', np.float32, (3,))]))
                return str(cache)

            with patch.object(module, 'generate_dog_dataset', side_effect=generate) as mocked:
                self.assertEqual(module._prepare_dataset(
                    path, 'train', section, stimuli, persistent=False), cache)
                self.assertFalse(list(path.rglob('*.pt')))
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
                torch.save(candidate_checkpoint(commit, epoch), candidates / f'epoch-{epoch:06d}.pt')

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

    def test_resume_second_set_reuses_data_and_completed_epoch(self):
        import train.train_flow_xt_net as training_engine
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'output').mkdir()
            (root / 'recipe').mkdir()
            path = root / 'recipe' / 'smoke.toml'
            text = recipe_text(directory)
            text = text.replace('number_of_sets = 3', 'number_of_sets = 2')
            text = text.replace('samples_per_set = 5000', 'samples_per_set = 2')
            text = text.replace('epochs_per_set = 10', 'epochs_per_set = 2')
            text = text.replace('n_samples = 1000', 'n_samples = 2')
            text = text.replace('dog_bundle_size = 64',
                                'dog_bundle_size = 2\nfov_deg = 3\nn_world_dots = 20\nx_supersampling = 4')
            path.write_text(text)
            save_latest = training_engine._save_latest

            def interrupt_after_checkpoint(checkpoint, *args):
                save_latest(checkpoint, *args)
                if checkpoint['training_set_index'] == 1 and checkpoint['epoch_within_set'] == 1:
                    raise RuntimeError('Simulated interruption in set 2')

            with patch.object(torch.cuda, 'is_available', return_value=False):
                with patch.object(training_engine, '_save_latest', side_effect=interrupt_after_checkpoint):
                    with self.assertRaisesRegex(RuntimeError, 'Simulated interruption'):
                        module.run_recipe(path, root=root, commit='c' * 40)
                training_path = root / 'stimuli' / 'training'
                state = json.loads((training_path / module.DATASET_STATE).read_text())
                self.assertEqual(state['spec']['seed'], 4001)
                checkpoints = root / 'output' / 'smoke' / 'checkpoints'
                completed_candidate = checkpoints / 'candidates' / 'epoch-000003.pt'
                saved_time = completed_candidate.stat().st_mtime_ns
                original_train = module.train_recipe_set

                def verify_reused_data(cfg, **kwargs):
                    self.assertEqual(cfg.training_set_index, 1)
                    self.assertEqual(json.loads((training_path / module.DATASET_STATE).read_text()), state)
                    self.assertEqual(Path(cfg.root_folder), training_path / state['cache'])
                    return original_train(cfg, **kwargs)

                with patch.object(module, 'generate_dog_dataset', wraps=module.generate_dog_dataset) as generate:
                    with patch.object(module, 'train_recipe_set', side_effect=verify_reused_data) as train:
                        module.run_recipe(path, root=root, commit='c' * 40)
                    self.assertEqual(train.call_count, 1)
                    self.assertEqual([call.kwargs['role'] for call in generate.call_args_list],
                                     ['validate', 'test'])
                latest = torch.load(checkpoints / 'latest.pt', weights_only=True)
                self.assertEqual(latest['global_epoch'], 4)
                self.assertEqual(completed_candidate.stat().st_mtime_ns, saved_time)
                self.assertFalse(training_path.exists())

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
            text += '\n[parallel_generation]\nenabled = true\nmin_threads = 1\nmax_threads = 2\n'
            path.write_text(text)
            module.run_recipe(path, root=root, commit='c' * 40)
            network = root / 'output' / 'smoke'
            self.assertTrue((network / 'run.json').is_file())
            self.assertEqual(len(list(root.rglob('source.tar.gz'))), 1)
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
