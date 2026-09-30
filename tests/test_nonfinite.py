import json
import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from recipe_helpers import module, recipe_text
import train.train_flow_xt_net as engine
from training_helpers import train_recipe_set
from stimuli.generate_flow_xt_data import generate_dog_dataset, StimConfig


BAD_VALUES = (float('nan'), float('inf'), float('-inf'))
METRICS = dict(n_samples=2, normalized_rmse=.1, translation_speed_mae=.1,
               translation_vector_tip_mae=.1, rotation_mae_deg=0., heading_mae_deg=1.)


class NonfiniteTests(unittest.TestCase):
    def test_checkpoint_loading_rejects_nonfinite_nested_tensors(self):
        from checkpoints import load_checkpoint
        from recipe_helpers import candidate_checkpoint
        commit = 'a' * 40
        candidate = candidate_checkpoint(commit) | {'model_state_dict': {'weight': torch.ones(2)}}
        resume = candidate | dict(kind='resume', optimizer_state_dict={'state': {0: {
            'step': torch.tensor(1.), 'exp_avg': torch.zeros(2), 'exp_avg_sq': torch.ones(2)}}},
            scaler_state_dict={}, scheduler_state_dict={'mode_worse': float('inf')}, settings={})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            torch.save(resume, path)
            load_checkpoint(path, resume=True, expected_commit=commit)
            for field in ('model_state_dict', 'optimizer_state_dict', 'scaler_state_dict', 'scheduler_state_dict'):
                for value in BAD_VALUES:
                    bad = copy.deepcopy(resume)
                    bad[field]['nested'] = [({'broken': torch.tensor([value])},)]
                    torch.save(bad, path)
                    for resumable in (False, True):
                        with self.subTest(field=field, value=value, resume=resumable):
                            with self.assertRaisesRegex(ValueError, f'Nonfinite tensor.*{field}.*broken'):
                                load_checkpoint(path, resume=resumable, expected_commit=commit)
            for value in BAD_VALUES:
                bad = copy.deepcopy(candidate)
                bad['model_state_dict']['weight'][0] = value
                torch.save(bad, path)
                with self.assertRaisesRegex(ValueError, 'Nonfinite tensor.*weight'):
                    load_checkpoint(path, expected_commit=commit)

    def test_tensor_health_accepts_finite_and_integer_tensors(self):
        from checkpoints import validate_tensor_health
        validate_tensor_health({'state': [torch.tensor([1, 2]), torch.tensor([True]),
                                         torch.tensor([], dtype=torch.float32), torch.tensor([1+2j])],
                                'mode_worse': float('inf')})
        with self.assertRaisesRegex(ValueError, 'Nonfinite tensor'):
            validate_tensor_health(torch.tensor([complex(0, float('inf'))]))

    def test_finite_loss_with_corrupt_final_update_retains_all_checkpoints(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory, patch.object(torch.cuda, 'is_available', return_value=False):
            root = Path(directory)
            cache = generate_dog_dataset(output_root=str(root / 'data'), n_samples=2, seed=4,
                speed_range=(.001, .002), heading_deg_range=(0, 90), rot_deg_range=(0, 0),
                role='train', dog_bundle_size=2,
                config=StimConfig(stim_wid_px=49, n_world_dots=10, x_supersampling=1))
            cfg = engine.RecipeTrainingSettings(root_folder=cache, checkpoint_dir=str(root / 'checkpoints'),
                git_commit='a' * 40, training_set_index=0, training_seed=4, epochs_per_set=1,
                shuffle_seed=99, batch_size=2, num_workers=0, allow_amp=False,
                mirror_train_stimuli=False, dx_values=(-1, 0, 1))
            train_recipe_set(cfg)
            train_recipe_set(replace(cfg, epochs_per_set=2))
            before = {p.relative_to(root): p.read_bytes() for p in (root / 'checkpoints').rglob('*.pt')}
            original_step = torch.optim.Adam.step
            for field in ('model_state_dict', 'optimizer_state_dict'):
                for value in BAD_VALUES:
                    def corrupt(optimizer, *args, **kwargs):
                        result = original_step(optimizer, *args, **kwargs)
                        parameter = optimizer.param_groups[0]['params'][0]
                        with torch.no_grad():
                            tensor = (parameter if field == 'model_state_dict'
                                      else optimizer.state[parameter]['exp_avg_sq'])
                            tensor.fill_(value)
                        return result
                    with self.subTest(field=field, value=value), \
                            patch.object(torch.optim.Adam, 'step', corrupt), \
                            patch.object(torch.optim.lr_scheduler.ReduceLROnPlateau, 'step') as scheduler:
                        with self.assertRaisesRegex(RuntimeError, f'Nonfinite tensor.*{field}.*global epoch 3'):
                            train_recipe_set(replace(cfg, epochs_per_set=3))
                        scheduler.assert_not_called()
                    after = {p.relative_to(root): p.read_bytes() for p in (root / 'checkpoints').rglob('*.pt')}
                    self.assertEqual(before, after)

    def test_recipe_rejects_nonfinite_scalars(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'recipe.toml'
            for value in ('nan', 'inf', '-inf'):
                with self.subTest(value=value):
                    path.write_text(recipe_text(directory).replace('initial = 0.001', f'initial = {value}'))
                    with self.assertRaisesRegex(ValueError, 'learning_rate.initial must be finite'):
                        module.load_recipe(path)

    def test_all_numeric_metrics_must_be_finite_but_heading_can_be_undefined(self):
        module._validate_metrics(METRICS | {'heading_mae_deg': None}, 'test')
        for key in METRICS.keys() - {'n_samples'}:
            for value in BAD_VALUES:
                with self.subTest(key=key, value=value):
                    with self.assertRaisesRegex(ValueError, 'must be finite'):
                        module._validate_metrics(METRICS | {key: value}, 'test')

    def test_json_failure_preserves_existing_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'results.json'
            module._atomic_json(path, METRICS)
            original = path.read_bytes()
            for value in BAD_VALUES:
                with self.assertRaises(ValueError):
                    module._atomic_json(path, METRICS | {'normalized_rmse': value})
                self.assertEqual(path.read_bytes(), original)

    def test_final_selection_rejects_nonfinite_saved_validation_loss(self):
        recipe = {'evaluation': {'batch_size': 2, 'num_workers': 0, 'shuffle': False}}
        commit = 'a' * 40
        for value in BAD_VALUES:
                with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    candidates = root / 'checkpoints' / 'candidates'
                    final = root / 'final'
                    candidates.mkdir(parents=True)
                    final.mkdir()
                    candidate = candidates / 'epoch-000001.pt'
                    from recipe_helpers import candidate_checkpoint
                    torch.save(candidate_checkpoint(commit) | {'validation_loss': value}, candidate)
                    latest = candidate_checkpoint(commit) | dict(kind='resume', optimizer_state_dict={},
                        scaler_state_dict={}, scheduler_state_dict={}, settings={})
                    torch.save(latest, root / 'checkpoints' / 'latest.pt')
                    with self.assertRaisesRegex(ValueError, 'best validation loss must be finite'):
                        module._select_final(root / 'checkpoints', final, recipe, commit)
                    self.assertFalse((final / 'model.pt').exists())
                    self.assertFalse((final / 'validation-results.json').exists())

    def test_nonfinite_epoch_preserves_checkpoint_and_scheduler(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory, patch.object(torch.cuda, 'is_available', return_value=False):
            root = Path(directory)
            cache = generate_dog_dataset(
                output_root=str(root / 'data'), n_samples=2, seed=4,
                speed_range=(.001, .002), heading_deg_range=(0, 90),
                rot_deg_range=(0, 0), role='train', dog_bundle_size=2,
                config=StimConfig(stim_wid_px=49, n_world_dots=10, x_supersampling=1))
            cfg = engine.RecipeTrainingSettings(
                root_folder=cache, checkpoint_dir=str(root / 'checkpoints'),
                git_commit='a' * 40, training_set_index=0, training_seed=4,
                epochs_per_set=1, shuffle_seed=99, batch_size=2, num_workers=0,
                allow_amp=False, mirror_train_stimuli=False, dx_values=(-1, 0, 1))
            latest = train_recipe_set(cfg)
            original = latest.read_bytes()
            loss = engine._normalized_motion_loss
            for value in BAD_VALUES:
                with self.subTest(value=value):
                    with patch.object(engine, '_normalized_motion_loss',
                                      side_effect=lambda *args: loss(*args) * value):
                        with patch.object(torch.optim.lr_scheduler.ReduceLROnPlateau, 'step') as step:
                            with self.assertRaisesRegex(RuntimeError, 'set 1, seed 4, epoch 2, global epoch 2'):
                                train_recipe_set(replace(cfg, epochs_per_set=2))
                            step.assert_not_called()
                    self.assertEqual(latest.read_bytes(), original)
                    self.assertFalse((root / 'checkpoints' / 'candidates' / 'epoch-000002.pt').exists())
