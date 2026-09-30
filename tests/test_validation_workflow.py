"""Fixed per-epoch validation controls scheduling/selection, never the test set."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from mission_plan_helpers import module, mission_plan_text, TEST_RUN_NAME


class ValidationWorkflowTests(unittest.TestCase):
    def setUp(self):
        timestamp = patch('run_revision.commit_timestamp', return_value='20000101000000')
        timestamp.start()
        self.addCleanup(timestamp.stop)

    def test_validation_first_early_stop_best_checkpoint_test_last_and_resume(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'output').mkdir()
            (root / 'scripts/mission_control/mission_plan').mkdir(parents=True)
            path = root / 'scripts/mission_control/mission_plan' / 'smoke.toml'
            text = mission_plan_text(directory).replace('samples_per_set = 5000', 'samples_per_set = 2')
            text = text.replace('n_samples = 1000', 'n_samples = 2')
            text = text.replace('epochs_per_set = 10', 'epochs_per_set = 3')
            text = text.replace('dog_bundle_size = 64',
                'dog_bundle_size = 2\nfov_deg = 3\nn_world_dots = 20\nx_supersampling = 2')
            text = text.replace('patience_epochs = 10', 'patience_epochs = 0')
            text = text.replace('minimum = 0.000001', 'minimum = 0.0005')
            text = text.replace('candidate_interval_epochs = 1', 'candidate_interval_epochs = 10')
            path.write_text(text)
            text += '\n[parallel_generation]\nenabled = false\nmin_threads = 1\nmax_threads = 2\n'
            path.write_text(text)
            generation_threads = []
            real_generate = module.generate_dog_dataset
            def generate_with_thread_check(**kwargs):
                generation_threads.append((kwargs['role'], torch.get_num_threads()))
                return real_generate(**kwargs)
            previous_threads = torch.get_num_threads()
            evaluation_order = []
            real_evaluate = module.evaluate_checkpoint
            def evaluate(checkpoint, cache, role, evaluation, commit, **kwargs):
                evaluation_order.append((role, checkpoint['global_epoch']))
                if role == 'validate':
                    self.assertFalse((root / 'stimuli' / 'test').exists())
                result = real_evaluate(checkpoint, cache, role, evaluation, commit)
                if role == 'validate':
                    value = float(checkpoint['global_epoch'])  # Deliberately worsening validation.
                    result.update(normalized_mse=value, normalized_rmse=value**.5)
                return result
            with patch.object(torch.cuda, 'is_available', return_value=False), \
                    patch.object(module, 'evaluate_checkpoint', side_effect=evaluate), \
                    patch.object(module, 'generate_dog_dataset', side_effect=generate_with_thread_check) as generate:
                module.run_mission_plan(path, root=root, commit='c' * 40, require_clean=False)
            self.assertEqual([call.kwargs['role'] for call in generate.call_args_list],
                             ['validate', 'train', 'test', 'duffy_wurtz_1993'])
            self.assertEqual(generation_threads, [('validate', 2), ('train', 2), ('test', 2),
                                                  ('duffy_wurtz_1993', 2)])
            self.assertEqual(torch.get_num_threads(), previous_threads)
            self.assertEqual(evaluation_order, [('validate', 1), ('validate', 2), ('test', 1), ('duffy_wurtz_1993', 1)])
            network = root / 'output' / TEST_RUN_NAME
            checkpoints = network / 'checkpoints'
            latest = torch.load(checkpoints / 'latest.pt', weights_only=True)
            self.assertEqual(latest['global_epoch'], 2)
            self.assertEqual(latest['learning_rate'], .0005)
            self.assertEqual(latest['stop_reason'], 'learning_rate_minimum')
            self.assertEqual(latest['scheduler_state_dict']['best'], 1.)
            self.assertEqual(latest['best_candidate'], 'epoch-000001.pt')
            self.assertEqual(torch.load(network / 'final' / 'model.pt', weights_only=True)['global_epoch'], 1)
            validation_files = {p: p.read_bytes() for p in (root / 'stimuli' / 'validation').rglob('*') if p.is_file()}
            with patch.object(module, 'train_mission_plan_set', side_effect=AssertionError('must stay stopped')), \
                    patch.object(module, 'generate_dog_dataset', side_effect=AssertionError('must reuse data')), \
                    patch.object(torch.cuda, 'is_available', return_value=False):
                module.run_mission_plan(path, root=root, commit='c' * 40, require_clean=False)
            self.assertEqual(validation_files, {p: p.read_bytes() for p in validation_files})
            events = module.EventLog(root / 'output' / TEST_RUN_NAME / 'training_log.jsonl', str(checkpoints))
            self.assertEqual(len(events.events('validation_result')), 2)
            self.assertEqual(events.events('training_stopped')[-1]['reason'], 'learning_rate_minimum')

    def test_mission_plan_rejects_old_training_loss_monitor_and_invalid_minimum(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mission_plan.toml'
            for text, message in (
                (mission_plan_text(directory).replace('monitor = "validation_loss"', 'monitor = "training_loss"'), 'validation_loss'),
                (mission_plan_text(directory).replace('minimum = 0.000001', 'minimum = 0.001'), 'greater than minimum')):
                path.write_text(text)
                with self.assertRaisesRegex(ValueError, message):
                    module.load_mission_plan(path)
