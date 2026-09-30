"""Real stop/evaluate/resume transitions and the transparent-motion protocol."""
import copy
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from mission_plan_helpers import module, mission_plan_text, TEST_RUN_NAME
from evaluation import duffy_wurtz_summary, validate_inference_contract


class FinishTrainingTests(unittest.TestCase):
    def setUp(self):
        timestamp = patch('run_revision.commit_timestamp', return_value='20000101000000')
        timestamp.start()
        self.addCleanup(timestamp.stop)

    def tiny_mission_plan(self, directory):
        root = Path(directory)
        (root / 'output').mkdir()
        (root / 'scripts/mission_control/mission_plan').mkdir(parents=True)
        path = root / 'scripts/mission_control/mission_plan' / 'smoke.toml'
        text = mission_plan_text(directory).replace('samples_per_set = 5000', 'samples_per_set = 2')
        text = text.replace('n_samples = 1000', 'n_samples = 2')
        text = text.replace('dog_bundle_size = 64',
            'dog_bundle_size = 2\nfov_deg = 3\nn_world_dots = 20\nx_supersampling = 2')
        text = text.replace('candidate_interval_epochs = 1', 'candidate_interval_epochs = 20')
        path.write_text(text)
        return root, path

    def test_stop_evaluation_then_resume_training_immediately(self):
        import train.train_flow_xt_net as engine
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        original_handlers = {n: signal.getsignal(n) for n in (signal.SIGUSR1, signal.SIGUSR2)}
        with tempfile.TemporaryDirectory() as directory, patch.object(
                torch.cuda, 'is_available', return_value=False):
            root, path = self.tiny_mission_plan(directory)
            real_evaluate = module.evaluate_checkpoint
            real_save = engine._save_latest
            order = []

            def save(checkpoint, *args):
                real_save(checkpoint, *args)
                os.kill(os.getpid(), signal.SIGUSR1)

            def evaluate(checkpoint, cache, role, evaluation, commit, **kwargs):
                order.append((role, checkpoint['global_epoch']))
                if role == 'duffy_wurtz_1993':
                    os.kill(os.getpid(), signal.SIGUSR2)
                return real_evaluate(checkpoint, cache, role, evaluation, commit, **kwargs)

            with patch.object(engine, '_save_latest', side_effect=save), \
                    patch.object(module, 'evaluate_checkpoint', side_effect=evaluate):
                module.run_mission_plan(path, root=root, commit='c' * 40, require_clean=False)
            network = root / 'output' / TEST_RUN_NAME
            latest_path = network / 'checkpoints/latest.pt'
            self.assertEqual(torch.load(latest_path, weights_only=True)['global_epoch'], 1)
            self.assertFalse((network / 'final/duffy_wurtz_1993-results.json').exists())
            self.assertEqual(order, [('validate', 1), ('test', 1), ('duffy_wurtz_1993', 1)])
            for n, handler in original_handlers.items():
                self.assertEqual(signal.getsignal(n), handler)
            # Resume must train epoch 2 before any pending evaluation.
            order.clear()
            with patch.object(engine, '_save_latest', side_effect=save), \
                    patch.object(module, 'evaluate_checkpoint', side_effect=evaluate):
                module.run_mission_plan(path, root=root, commit='c' * 40, require_clean=False)
            self.assertEqual(order[0], ('validate', 2))
            latest = torch.load(latest_path, weights_only=True)
            self.assertEqual(latest['global_epoch'], 2)
            self.assertTrue(all(state['step'].item() == 2
                                for state in latest['optimizer_state_dict']['state'].values()))
            # Explicit evaluation never advances training.
            with patch.object(module, 'train_mission_plan_set', side_effect=AssertionError('no training')):
                module.run_mission_plan(path, root=root, commit='c' * 40, require_clean=False, mode='evaluate')
            self.assertEqual(torch.load(latest_path, weights_only=True)['global_epoch'], 2)
            self.assertTrue((network / 'final/duffy_wurtz_1993-results.json').is_file())

    def test_signal_before_first_epoch_still_produces_one_healthy_checkpoint(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory, patch.object(
                torch.cuda, 'is_available', return_value=False):
            root, path = self.tiny_mission_plan(directory)
            original = module.generate_dog_dataset
            generated = []

            def generate(**kwargs):
                generated.append((kwargs['role'], kwargs['config'].superimpose))
                if kwargs['role'] == 'validate':
                    os.kill(os.getpid(), signal.SIGUSR1)
                return original(**kwargs)

            with patch.object(module, 'generate_dog_dataset', side_effect=generate):
                module.run_mission_plan(path, root=root, commit='c' * 40, require_clean=False)
            self.assertEqual(generated, [('validate', False), ('duffy_wurtz_1993', True),
                                         ('train', False), ('test', False)])
            checkpoint = torch.load(root / 'output' / TEST_RUN_NAME / 'checkpoints/latest.pt', weights_only=True)
            self.assertEqual(checkpoint['global_epoch'], 1)

    def test_stop_mid_training_discards_epoch_and_skips_evaluation(self):
        import train.train_flow_xt_net as engine
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory, patch.object(
                torch.cuda, 'is_available', return_value=False):
            root, path = self.tiny_mission_plan(directory)
            real_save = engine._save_latest

            def save(checkpoint, *args):
                real_save(checkpoint, *args)
                os.kill(os.getpid(), signal.SIGUSR2)

            with patch.object(engine, '_save_latest', side_effect=save), \
                    patch.object(module, 'evaluate_model', side_effect=AssertionError('no final evaluation')):
                module.run_mission_plan(path, root=root, commit='c' * 40, require_clean=False)
            latest_path = root / 'output' / TEST_RUN_NAME / 'checkpoints/latest.pt'
            before = latest_path.read_bytes()
            real_train = module.train_mission_plan_set

            def train(settings, **kwargs):
                def epoch_started(epoch):
                    self.assertEqual(epoch, 2)
                    os.kill(os.getpid(), signal.SIGUSR2)
                kwargs['epoch_callback'] = epoch_started
                return real_train(settings, **kwargs)

            with patch.object(module, 'train_mission_plan_set', side_effect=train), \
                    patch.object(module, 'evaluate_checkpoint', side_effect=AssertionError('no evaluation')):
                module.run_mission_plan(path, root=root, commit='c' * 40, require_clean=False)
            self.assertEqual(latest_path.read_bytes(), before)
            events = module.EventLog(root / 'output' / TEST_RUN_NAME / 'training_log.jsonl', str(latest_path.parent))
            self.assertEqual(len(events.events('run_stopped')), 2)
            self.assertEqual(len(events.events('run_completed')), 0)

    def test_new_fixed_set_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root, path = self.tiny_mission_plan(directory)
            text = path.read_text()
            for changed in (text.replace('seed = 1000', 'seed = 2000'),
                            text.replace('seed = 1000', 'seed = 5000'),
                            text.replace('/stimuli/duffy_wurtz_1993_set', '/stimuli/test'),
                            text.replace('[duffy_wurtz_1993_set]', '[duffy_wurtz_set]')):
                path.write_text(changed)
                with self.assertRaises(ValueError):
                    module.load_mission_plan(path)

    def test_repeated_pause_resumes_latest_and_advances_sets(self):
        """Resume a partial set, then cross a set boundary, never rewind to best."""
        import train.train_flow_xt_net as engine
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory, patch.object(
                torch.cuda, 'is_available', return_value=False):
            root, path = self.tiny_mission_plan(directory)
            path.write_text(path.read_text().replace('epochs_per_set = 10', 'epochs_per_set = 2'))
            original = module.evaluate_checkpoint

            def evaluate(checkpoint, cache, role, evaluation, commit, **kwargs):
                result = original(checkpoint, cache, role, evaluation, commit, **kwargs)
                if role == 'validate':
                    # Best stays epoch 1, while training must progress to 2 and 3.
                    result['normalized_mse'] = float(checkpoint['global_epoch'])
                    result['normalized_rmse'] = result['normalized_mse'] ** .5
                return result

            real_save = engine._save_latest

            def save(checkpoint, *args):
                real_save(checkpoint, *args)
                os.kill(os.getpid(), signal.SIGUSR1)

            network = root / 'output' / TEST_RUN_NAME
            generation_roles = []
            with patch.object(module, 'evaluate_checkpoint', side_effect=evaluate), \
                    patch.object(engine, '_save_latest', side_effect=save):
                for epoch in (1, 2, 3):
                    with patch.object(module, 'generate_dog_dataset', wraps=module.generate_dog_dataset) as generate:
                        module.run_mission_plan(path, root=root, commit='c' * 40, require_clean=False)
                    generation_roles.append([c.kwargs['role'] for c in generate.call_args_list])
                    latest = torch.load(network / 'checkpoints/latest.pt', weights_only=True)
                    self.assertEqual(latest['global_epoch'], epoch)
                    self.assertEqual(latest['training_set_index'], (epoch - 1) // 2)
                    self.assertEqual(latest['epoch_within_set'], (epoch - 1) % 2 + 1)
                    self.assertEqual(latest['scheduler_state_dict']['last_epoch'], epoch)
                    self.assertTrue(all(state['step'].item() == epoch
                                        for state in latest['optimizer_state_dict']['state'].values()))
                    self.assertIsNone(latest['stop_reason'])
                    final = torch.load(network / 'final/model.pt', weights_only=True)
                    self.assertEqual(final['global_epoch'], 1)
            self.assertEqual(generation_roles, [
                ['validate', 'duffy_wurtz_1993', 'train', 'test'], [], ['train']])
            log = module.EventLog(root / 'output' / TEST_RUN_NAME / 'training_log.jsonl', str(network / 'checkpoints'))
            self.assertEqual([e['global_epoch'] for e in log.events('epoch_completed')], [1, 2, 3])

    def test_named_protocol_only_relaxes_superimposition(self):
        geometry = dict(superimpose=False, fov_deg=360, stim_wid_px=21600)
        checkpoint = {'data_contract': dict(schema='schema', boundary='circular',
                                            git_commit='c' * 40, geometry=geometry)}
        metadata = dict(storage_format='schema', boundary='circular', git_commit='c' * 40,
                        stimulus_generation=dict(role='duffy_wurtz_1993',
                                                 stimulus_config=geometry | {'superimpose': True}))
        validate_inference_contract(checkpoint, metadata, 'duffy_wurtz_1993')
        with self.assertRaises(ValueError):
            validate_inference_contract(checkpoint, metadata)
        for key, value in (('fov_deg', 120), ('stim_wid_px', 7200), ('superimpose', False)):
            altered = copy.deepcopy(metadata)
            altered['stimulus_generation']['stimulus_config'][key] = value
            with self.assertRaises(ValueError):
                validate_inference_contract(checkpoint, altered, 'duffy_wurtz_1993')
        for key, value in (('git_commit', 'd' * 40), ('boundary', 'valid')):
            with self.assertRaises(ValueError):
                validate_inference_contract(checkpoint, metadata | {key: value}, 'duffy_wurtz_1993')

    def test_overlay_direction_matches_rendered_motion(self):
        from stimuli.generate_flow_xt_data import generate_flow_xt_stim, StimConfig
        from unittest.mock import Mock
        rng = Mock()
        rng.uniform.return_value = np.array([[0., .5]])
        config = StimConfig(fov_deg=360, n_world_dots=1, x_supersampling=1)
        for rotation in (-.5, .5):
            frames = generate_flow_xt_stim([0, 0], rotation, config, rng).numpy()
            displacement = int(frames[1].argmax()) - int(frames[0].argmax())
            self.assertEqual(displacement, int(rotation * 60))

    def test_heading_bias_retinal_sign_and_undefined_heading(self):
        angles = np.radians([-10., 20., 0., 0.])
        pred = np.column_stack((np.sin(angles), np.cos(angles), [1., -1., 0., 1.]))
        pred[-1, :2] = 0
        target = np.array([[0., 1., -1.], [0., 1., 1.], [0., 1., 0.], [0., 1., -1.]])
        result = duffy_wurtz_summary([(pred, target)], [1., 1., 1.])
        groups = result['heading_response']
        self.assertAlmostEqual(groups['leftward']['heading_bias_deg'], -10.)
        self.assertAlmostEqual(groups['leftward']['overlay_aligned_heading_bias_deg'], 10.)
        self.assertAlmostEqual(groups['rightward']['overlay_aligned_heading_bias_deg'], 20.)
        self.assertEqual(groups['leftward']['n_samples'], 2)
        self.assertEqual(groups['leftward']['n_headings'], 1)
        self.assertIsNone(groups['stationary']['overlay_aligned_heading_bias_deg'])


if __name__ == '__main__':
    unittest.main()
