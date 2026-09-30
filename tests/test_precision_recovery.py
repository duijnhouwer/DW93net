"""Numerical recovery must restore state, retry once, and stay in float32."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from checkpoints import load_checkpoint
from training_log import EventLog, training_log
from stimuli.generate_flow_xt_data import generate_dog_dataset, StimConfig
import train.train_flow_xt_net as engine
from training_helpers import train_recipe_set


class PrecisionRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cpu = patch.object(torch.cuda, 'is_available', return_value=False)
        self.cpu.start()
        self.addCleanup(self.cpu.stop)
        self.root = Path(self.directory.name)
        cache = generate_dog_dataset(output_root=str(self.root / 'data'), n_samples=4,
            seed=4, speed_range=(.001, .002), heading_deg_range=(0, 90), rot_deg_range=(0, 0),
            role='train', config=StimConfig(stim_wid_px=49, n_world_dots=10, x_supersampling=1))
        self.cfg = engine.RecipeTrainingSettings(root_folder=cache,
            checkpoint_dir=str(self.root / 'run'), git_commit='a' * 40,
            training_set_index=0, training_seed=4, epochs_per_set=1, shuffle_seed=99,
            batch_size=2, num_workers=0, allow_amp=True, allow_tp32=True,
            mirror_train_stimuli=False, dx_values=(-1, 0, 1))

    def read(self, path):
        return load_checkpoint(path, resume=True, expected_commit=self.cfg.git_commit)

    def assert_state_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_state_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_first_epoch_retries_from_initialization_and_matches_clean_float32(self):
        baseline = self.read(train_recipe_set(replace(self.cfg,
            checkpoint_dir=str(self.root / 'baseline'), allow_amp=False, allow_tp32=False)))
        real_loss = engine._normalized_motion_loss
        targets = []
        def fail_first_epoch(prediction, target, scales):
            targets.append(target.clone())
            loss = real_loss(prediction, target, scales)
            return loss * float('inf') if len(targets) <= 2 else loss
        with patch.object(engine, '_normalized_motion_loss', side_effect=fail_first_epoch):
            with training_log(self.root / 'training_log.jsonl'):
                actual = self.read(train_recipe_set(self.cfg))
        log = (self.root / 'training_log.jsonl').read_text()
        self.assertIn('Numerical failure details:', log)
        self.assertIn('AMP and TF32 OFF', log)
        self.assertIn('Healthy checkpoint saved:', log)
        self.assertEqual(len(targets), 4)
        for i in range(2):
            torch.testing.assert_close(targets[i], targets[i+2], atol=0, rtol=0)
        for key in ('model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict'):
            self.assert_state_equal(actual[key], baseline[key])
        self.assertFalse(actual['allow_amp'])
        self.assertFalse(actual['allow_tp32'])
        self.assertEqual(actual['scaler_state_dict'], {})
        self.assertEqual(actual['global_epoch'], 1)
        events = EventLog(self.root / 'training_log.jsonl', self.cfg.checkpoint_dir)
        self.assertEqual(events.events('precision_fallback')[-1]['global_epoch'], 1)
        completed = events.events('epoch_completed')[-1]
        self.assertEqual(completed['attempt'], 2)
        failed = events.events('numerical_failure')[-1]
        self.assertEqual(failed['attempt'], 1)
        self.assertGreater(failed['attempt_duration_s'], 0)
        self.assertGreater(completed['epoch_wall_duration_s'],
                           completed['training_duration_s'] + failed['attempt_duration_s'])
        # Recipe still permits reduced precision, but resume/next set must not.
        next_cfg = replace(self.cfg, training_set_index=1, training_seed=5)
        resumed = self.read(train_recipe_set(next_cfg))
        self.assertFalse(resumed['allow_amp'])
        self.assertFalse(resumed['allow_tp32'])

    def test_adam_corruption_rolls_back_and_then_stops_if_float32_also_fails(self):
        latest = train_recipe_set(self.cfg)
        before = {p: p.read_bytes() for p in Path(self.cfg.checkpoint_dir).rglob('*.pt')}
        original_step = torch.optim.Adam.step
        count = 0
        def corrupt(optimizer, *args, **kwargs):
            nonlocal count
            count += 1
            result = original_step(optimizer, *args, **kwargs)
            # Corrupt only the last update of each attempt; loss stays finite.
            if count % 2 == 0:
                parameter = optimizer.param_groups[0]['params'][0]
                optimizer.state[parameter]['exp_avg_sq'].fill_(float('inf'))
            return result
        with patch.object(torch.optim.Adam, 'step', corrupt):
            with self.assertRaisesRegex(engine.NumericalTrainingError, 'full float32 failed'):
                train_recipe_set(replace(self.cfg, epochs_per_set=2))
        self.assertEqual(count, 4)
        self.assertEqual(before, {p: p.read_bytes() for p in Path(self.cfg.checkpoint_dir).rglob('*.pt')})
        # A new invocation restores that same healthy checkpoint, stays float32.
        actual = self.read(train_recipe_set(replace(self.cfg, epochs_per_set=2)))
        baseline = self.read(train_recipe_set(replace(self.cfg, epochs_per_set=2,
            checkpoint_dir=str(self.root / 'baseline'), allow_amp=False, allow_tp32=False)))
        for key in ('model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict'):
            self.assert_state_equal(actual[key], baseline[key])
        self.assertFalse(actual['allow_amp'])
        self.assertFalse(actual['allow_tp32'])

    def test_interruption_during_retry_keeps_fallback_before_first_checkpoint(self):
        original = engine._normalized_motion_loss
        calls = 0
        def interrupt(prediction, target, scales):
            nonlocal calls
            calls += 1
            if calls > 2:
                raise OSError('simulated interruption')
            return original(prediction, target, scales) * float('inf')
        with patch.object(engine, '_normalized_motion_loss', side_effect=interrupt):
            with self.assertRaisesRegex(OSError, 'simulated interruption'):
                train_recipe_set(self.cfg)
        self.assertFalse((Path(self.cfg.checkpoint_dir) / 'latest.pt').exists())
        actual = self.read(train_recipe_set(self.cfg))
        self.assertFalse(actual['allow_amp'])
        self.assertFalse(actual['allow_tp32'])

    def test_unrelated_errors_do_not_retry(self):
        with patch.object(engine, '_normalized_motion_loss', side_effect=RuntimeError('unrelated')) as loss:
            with self.assertRaisesRegex(RuntimeError, 'unrelated'):
                train_recipe_set(self.cfg)
        self.assertEqual(loss.call_count, 1)
        self.assertFalse((Path(self.cfg.checkpoint_dir) / 'precision-fallback.json').exists())
        self.assertFalse(EventLog(self.root / 'training_log.jsonl',
                                 self.cfg.checkpoint_dir).events('precision_fallback'))

    def test_nonfinite_validation_retries_without_advancing_scheduler(self):
        from training_helpers import fixed_validation
        calls = []
        def validation(checkpoint):
            calls.append(checkpoint['allow_amp'])
            result = fixed_validation(checkpoint)
            if len(calls) == 1:
                result['normalized_mse'] = float('inf')
            return result
        latest = train_recipe_set(self.cfg, validate_epoch=validation)
        state = self.read(latest)
        self.assertEqual(calls, [True, False])
        self.assertEqual(state['scheduler_state_dict']['last_epoch'], 1)
        self.assertEqual(state['validation_loss'], 1.)

    def test_new_best_saved_between_periodic_snapshots(self):
        from training_helpers import fixed_validation
        values = iter([3., 1., 2.])
        def validation(checkpoint):
            return fixed_validation(checkpoint) | {'normalized_mse': next(values)}
        latest = train_recipe_set(replace(self.cfg, epochs_per_set=3, candidate_interval_epochs=10),
                                  validate_epoch=validation)
        self.assertEqual(self.read(latest)['best_candidate'], 'epoch-000002.pt')
        candidate = Path(self.cfg.checkpoint_dir) / 'candidates' / 'epoch-000002.pt'
        self.assertTrue(candidate.exists())
        self.assertEqual(torch.load(candidate, weights_only=True)['validation_loss'], 1.)
