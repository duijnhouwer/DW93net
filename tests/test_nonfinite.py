import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_training_recipe import module, recipe_text
import train.train_flow_xt_net as engine
from stimuli.generate_flow_xt_data import generate_dog_dataset, StimConfig


BAD_VALUES = (float('nan'), float('inf'), float('-inf'))
METRICS = dict(n_samples=2, normalized_rmse=.1, translation_speed_mae=.1,
               translation_vector_tip_mae=.1, rotation_mae_deg=0., heading_mae_deg=1.)


class NonfiniteTests(unittest.TestCase):
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

    def test_fresh_and_cached_candidates_reject_nonfinite_results(self):
        recipe = {'evaluation': {'batch_size': 2, 'num_workers': 0, 'shuffle': False}}
        commit = 'a' * 40
        for cached in (False, True):
            for value in BAD_VALUES:
                with self.subTest(cached=cached, value=value), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    candidates = root / 'checkpoints' / 'candidates'
                    final = root / 'final'
                    candidates.mkdir(parents=True)
                    final.mkdir()
                    candidate = candidates / 'epoch-000001.pt'
                    torch.save({'git_commit': commit, 'model_state_dict': {}}, candidate)
                    invalid = METRICS | {'normalized_rmse': value}
                    if cached:
                        (final / 'validation-progress.json').write_text(json.dumps(dict(
                            version=2, evaluation_code=module.code_identity(module.TRAINING_SOURCES), selection_metric='normalized_rmse',
                            evaluation=recipe['evaluation'], candidates=[invalid | dict(
                                candidate=candidate.name, checkpoint_sha256=module._sha256(candidate))])))
                    with patch.object(module, 'evaluate_model', return_value=invalid) as evaluate:
                        with self.assertRaisesRegex(ValueError, 'normalized_rmse must be finite'):
                            module._select_final(root / 'checkpoints', final, root / 'validation', recipe, commit)
                        self.assertEqual(evaluate.call_count, 0 if cached else 1)
                    self.assertFalse((final / 'model.pt').exists())
                    self.assertFalse((final / 'validation-results.json').exists())

    def test_nonfinite_epoch_preserves_checkpoint_and_scheduler(self):
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
                mixed_precision=False, mirror_train_stimuli=False, dx_values=(-1, 0, 1))
            latest = engine.train_recipe_set(cfg)
            original = latest.read_bytes()
            loss = engine._normalized_motion_loss
            for value in BAD_VALUES:
                with self.subTest(value=value):
                    with patch.object(engine, '_normalized_motion_loss',
                                      side_effect=lambda *args: loss(*args) * value):
                        with patch.object(torch.optim.lr_scheduler.ReduceLROnPlateau, 'step') as step:
                            with self.assertRaisesRegex(RuntimeError, 'set 1, seed 4, epoch 2, global epoch 2'):
                                engine.train_recipe_set(replace(cfg, epochs_per_set=2))
                            step.assert_not_called()
                    self.assertEqual(latest.read_bytes(), original)
                    self.assertFalse((root / 'checkpoints' / 'candidates' / 'epoch-000002.pt').exists())
