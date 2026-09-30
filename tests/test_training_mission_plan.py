import importlib.util
import json
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from mission_plan_helpers import module, mission_plan_text, DETECTOR, TEST_RUN_NAME, candidate_checkpoint


class TrainingMissionPlanTests(unittest.TestCase):
    def test_discovery_requires_one_tracked_plan_in_new_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            legacy = root / 'recipe'
            legacy.mkdir()
            (legacy / 'old.toml').write_text('version = 1\n')
            with self.assertRaisesRegex(ValueError, 'Expected exactly one'):
                module.discover_mission_plan(root)
            plans = root / 'scripts/mission_control/mission_plan'
            (plans / 'archive').mkdir(parents=True)
            (plans / 'archive/old.toml').write_text('version = 1\n')
            active = plans / 'active.toml'
            active.write_text('version = 1\n')
            with self.assertRaisesRegex(ValueError, 'must be tracked by Git'):
                module.discover_mission_plan(root)
            subprocess.run(['git', '-C', str(root), 'add', '.'], check=True)
            self.assertEqual(module.discover_mission_plan(root), active)
            (plans / 'second.toml').write_text('version = 1\n')
            with self.assertRaisesRegex(ValueError, 'Expected exactly one'):
                module.discover_mission_plan(root)

    def setUp(self):
        timestamp = patch('run_revision.commit_timestamp', return_value='20000101000000')
        timestamp.start()
        self.addCleanup(timestamp.stop)

    def test_final_test_cache_requires_exact_model_and_dataset_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test-results.json'
            metrics = dict(n_samples=2, normalized_rmse=1.0,
                           translation_speed_mae=1.0, translation_vector_tip_mae=1.0,
                           rotation_mae_deg=1.0, heading_mae_deg=None)
            identity = dict(model_sha256='a' * 64, dataset_sha256='b' * 64,
                            evaluation={'batch_size': 2}, evaluator_git_commit='c' * 40,
                            split='test')
            module._atomic_json(path, {'version': 1, 'identity': identity, 'metrics': metrics})
            self.assertEqual(module._cached_test_result(path, identity), metrics)
            for key in ('model_sha256', 'dataset_sha256'):
                changed = identity | {key: 'd' * 64}
                self.assertIsNone(module._cached_test_result(path, changed))
            module._atomic_json(path, metrics)  # Old, unidentified result format.
            self.assertIsNone(module._cached_test_result(path, identity))

    def test_dataset_hash_includes_tensor_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            (cache / 'test').mkdir()
            (cache / 'dog.json').write_bytes(b'metadata')
            bundle = cache / 'test' / 'bundle.npy'
            bundle.write_bytes(b'payload-one')
            original = module._dataset_sha256(cache, 'test')
            bundle.write_bytes(b'payload-two')
            self.assertNotEqual(module._dataset_sha256(cache, 'test'), original)

    def test_removed_test_staging_option_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mission_plan.toml'
            path.write_text(mission_plan_text(directory).replace(
                '[test_set]', f'[test_set]\nstaging_path = "{directory}/staged-test"'))
            with self.assertRaisesRegex(ValueError, 'Unknown test_set field.*staging_path'):
                module.load_mission_plan(path)

    def test_dog_order_option_is_boolean(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mission_plan.toml'
            for value in ('true', 'false', '1', '"false"'):
                path.write_text(mission_plan_text(directory).replace('[stimuli]',
                    f'[stimuli]\napply_dog_filter_after_downsample = {value}'))
                if value in ('true', 'false'):
                    self.assertEqual(module.load_mission_plan(path)['stimuli']['apply_dog_filter_after_downsample'],
                                     value == 'true')
                else:
                    with self.assertRaisesRegex(ValueError, 'apply_dog_filter_after_downsample'):
                        module.load_mission_plan(path)

    def test_precision_options_are_required_booleans(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mission_plan.toml'
            for key in ('allow_amp', 'allow_tp32'):
                for value in ('true', 'false', '1', '"true"'):
                    path.write_text(mission_plan_text(directory).replace(f'{key} = true', f'{key} = {value}'))
                    if value in ('true', 'false'):
                        self.assertEqual(module.load_mission_plan(path)['optimization'][key], value == 'true')
                    else:
                        with self.assertRaisesRegex(ValueError, key):
                            module.load_mission_plan(path)

    def test_removed_full_force_first_set_option_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mission_plan.toml'
            original = mission_plan_text(directory)
            for value in ('true', 'false', '1', '"true"'):
                with self.subTest(value=value):
                    path.write_text(original + '\n[parallel_generation]\n'
                                    f'full_force_first_set = {value}\n')
                    with self.assertRaisesRegex(ValueError, 'Unknown parallel_generation field.*full_force_first_set'):
                        module.load_mission_plan(path)

    def test_tf32_switch_controls_matmul_and_convolution(self):
        from precision import configure_tf32
        matmul = torch.backends.cuda.matmul.fp32_precision
        conv = torch.backends.cudnn.conv.fp32_precision
        try:
            for enabled, expected in ((False, 'ieee'), (True, 'tf32')):
                configure_tf32(enabled)
                self.assertEqual(torch.backends.cuda.matmul.fp32_precision, expected)
                self.assertEqual(torch.backends.cudnn.conv.fp32_precision, expected)
        finally:
            torch.backends.cuda.matmul.fp32_precision = matmul
            torch.backends.cudnn.conv.fp32_precision = conv

    def test_speed_bounds_must_be_representable_in_float32(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mission_plan.toml'
            for bounds in ('[1e-12, 2e-12]', '[1e-12, 1e-12]', '[1e-50, 1e-40]', '[1e38, 1e40]'):
                path.write_text(mission_plan_text(directory).replace(
                    'speed_range = [0.001, 0.006]', f'speed_range = {bounds}'))
                with self.subTest(bounds=bounds):
                    if '1e-12' in bounds:
                        module.load_mission_plan(path)
                    else:
                        with self.assertRaisesRegex(ValueError, 'float32'):
                            module.load_mission_plan(path)

    def test_zero_speed_mission_plan_is_valid_but_negative_speed_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mission_plan.toml'
            original = mission_plan_text(directory)
            path.write_text(original.replace('speed_range = [0.001, 0.006]',
                                             'speed_range = [0, 0]'))
            self.assertEqual(module.load_mission_plan(path)['stimuli']['speed_range'], [0, 0])
            path.write_text(original.replace('speed_range = [0.001, 0.006]',
                                             'speed_range = [-0.001, 0]'))
            with self.assertRaisesRegex(ValueError, 'nonnegative'):
                module.load_mission_plan(path)

    def test_near_exclusion_radius_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mission_plan.toml'
            for radius in ('0', '0.04', '0.999999', '-0.01', '1', '1.1', 'nan', 'inf', '-inf', 'true'):
                with self.subTest(radius=radius):
                    path.write_text(mission_plan_text(directory).replace(
                        '[stimuli]', f'[stimuli]\nnear_exclusion_radius = {radius}'))
                    if radius in ('0', '0.04', '0.999999'):
                        module.load_mission_plan(path)
                    else:
                        with self.assertRaisesRegex(ValueError, 'near_exclusion_radius'):
                            module.load_mission_plan(path)

    def test_fov_must_fit_full_circle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mission_plan.toml'
            original = mission_plan_text(directory)
            for fov in ('0.001', '120', '240', '360', '0', '-1', '360.001', 'nan', 'inf', 'true'):
                with self.subTest(fov=fov):
                    path.write_text(original.replace('[stimuli]', f'[stimuli]\nfov_deg = {fov}'))
                    if fov in ('0.001', '120', '240', '360'):
                        module.load_mission_plan(path)
                    else:
                        with self.assertRaisesRegex(ValueError, 'fov_deg'):
                            module.load_mission_plan(path)

    def test_training_seeds_must_start_above_distinct_held_out_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mission_plan.toml'
            original = mission_plan_text(directory)
            for start in (0, 1999, 2000, 2500, 3000):
                with self.subTest(start=start):
                    path.write_text(original.replace('start_seed = 4000', f'start_seed = {start}'))
                    with self.assertRaisesRegex(ValueError, 'greater than all'):
                        module.load_mission_plan(path)
            path.write_text(original.replace('seed = 3000', 'seed = 2000'))
            with self.assertRaisesRegex(ValueError, 'must differ'):
                module.load_mission_plan(path)
            # Both held-out seed orderings are valid, even with many training sets.
            for validation, test in ((2000, 3000), (3000, 2000)):
                before_test, test_section = original.split('[test_set]', 1)
                text = (before_test.replace('seed = 2000', f'seed = {validation}')
                        + '[test_set]' + test_section.replace('seed = 3000', f'seed = {test}'))
                text = text.replace('start_seed = 4000', 'start_seed = 3001')
                text = text.replace('number_of_sets = 3', 'number_of_sets = 1000000')
                path.write_text(text)
                module.load_mission_plan(path)
            for increment in (0, -1):
                path.write_text(original.replace('seed_increment = 1', f'seed_increment = {increment}'))
                with self.assertRaisesRegex(ValueError, 'seed_increment must be a positive integer'):
                    module.load_mission_plan(path)

    def test_loads_schema_and_rotation_range(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'translation_only.toml'
            path.write_text(mission_plan_text(directory))
            mission_plan = module.load_mission_plan(path)
            self.assertEqual(mission_plan['training_sets']['number_of_sets'], 3)
            self.assertEqual(mission_plan['stimuli']['dog_bundle_size'], 64)
            path.write_text(mission_plan_text(directory).replace(
                'rot_deg_range = [0, 0]', 'rot_deg_range = [-0.5, 0.5]'))
            mission_plan = module.load_mission_plan(path)
            self.assertEqual(mission_plan['stimuli']['rot_deg_range'], [-0.5, 0.5])

    def test_rejects_relative_or_nested_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'translation_only.toml'
            path.write_text(mission_plan_text(directory).replace(
                f'{directory}/stimuli/test', f'{directory}/stimuli/training/test'))
            with self.assertRaisesRegex(ValueError, 'separate, non-nested'):
                module.load_mission_plan(path)
            path.write_text(mission_plan_text(directory).replace(f'{directory}/output', 'relative'))
            with self.assertRaisesRegex(ValueError, 'absolute path'):
                module.load_mission_plan(path)

    def test_output_isolated_by_commit_and_reused_only_for_matching_mission_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'output'
            output.mkdir()
            mission_plan = {'network': {'output_path': str(output)}}
            checkpoints, final = module._initialize_output(mission_plan, 'translation_only', 'a' * 40)
            self.assertEqual(module._initialize_output(mission_plan, 'translation_only', 'a' * 40),
                             (checkpoints, final))
            manifest = json.loads((checkpoints.parent / 'run.json').read_text())
            self.assertEqual(manifest['git_commit'], 'a' * 40)
            other, _ = module._initialize_output(mission_plan, 'translation_only', 'b' * 40)
            self.assertNotEqual(other, checkpoints)
            self.assertEqual(checkpoints.parent.name, 'translation_only-20000101000000-' + 'a' * 40)
            self.assertEqual(json.loads((checkpoints.parent / 'run.json').read_text())['git_commit'], 'a' * 40)
            with self.assertRaisesRegex(ValueError, 'does not match'):
                module._initialize_output(mission_plan | {'changed': True}, 'translation_only', 'b' * 40)

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
                    storage_format='dog-contrast-bundles-v3', storage_dtype='torch.float32',
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
            with self.assertRaisesRegex(ValueError, 'differs from the mission plan'):
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

    def test_final_selection_uses_committed_best_not_last_or_orphan_candidate(self):
        commit = 'a' * 40
        mission_plan = {'evaluation': {'batch_size': 2, 'num_workers': 0, 'shuffle': False}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / 'checkpoints' / 'candidates'
            final = root / 'final'
            candidates.mkdir(parents=True)
            final.mkdir()
            for epoch in range(1, 4):
                torch.save(candidate_checkpoint(commit, epoch), candidates / f'epoch-{epoch:06d}.pt')

            latest = candidate_checkpoint(commit, 2) | dict(kind='resume',
                optimizer_state_dict={}, scaler_state_dict={}, scheduler_state_dict={}, settings={})
            torch.save(latest, root / 'checkpoints' / 'latest.pt')
            with patch.object(module, 'evaluate_model') as evaluate:
                module._select_final(root / 'checkpoints', final, mission_plan, commit)
                evaluate.assert_not_called()
            log = module.EventLog(root / 'training_log.jsonl', str(root / 'checkpoints'))
            progress = log.events('validation_completed')[-1]
            self.assertEqual(progress['selected'], 'epoch-000001.pt')
            self.assertEqual(torch.load(final / 'model.pt', weights_only=True)['global_epoch'], 1)

    def test_resume_second_set_reuses_data_and_completed_epoch(self):
        import train.train_flow_xt_net as training_engine
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'output').mkdir()
            (root / 'scripts/mission_control/mission_plan').mkdir(parents=True)
            path = root / 'scripts/mission_control/mission_plan' / 'smoke.toml'
            text = mission_plan_text(directory)
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
                        module.run_mission_plan(path, root=root, commit='c' * 40, require_clean=False)
                training_path = root / 'stimuli' / 'training'
                state = json.loads((training_path / module.DATASET_STATE).read_text())
                self.assertEqual(state['spec']['seed'], 4001)
                checkpoints = root / 'output' / TEST_RUN_NAME / 'checkpoints'
                completed_candidate = checkpoints / 'candidates' / 'epoch-000003.pt'
                saved_time = completed_candidate.stat().st_mtime_ns
                original_train = module.train_mission_plan_set

                def verify_reused_data(cfg, **kwargs):
                    self.assertEqual(cfg.training_set_index, 1)
                    self.assertEqual(json.loads((training_path / module.DATASET_STATE).read_text()), state)
                    self.assertEqual(Path(cfg.root_folder), training_path / state['cache'])
                    return original_train(cfg, **kwargs)

                with patch.object(module, 'generate_dog_dataset', wraps=module.generate_dog_dataset) as generate:
                    with patch.object(module, 'train_mission_plan_set', side_effect=verify_reused_data) as train:
                        module.run_mission_plan(path, root=root, commit='c' * 40, require_clean=False)
                    self.assertEqual(train.call_count, 1)
                    self.assertEqual([call.kwargs['role'] for call in generate.call_args_list],
                                     ['test'])
                latest = torch.load(checkpoints / 'latest.pt', weights_only=True)
                log = (root / 'output' / TEST_RUN_NAME / 'training_log.jsonl').read_text()
                self.assertEqual(log.count('Starting/resuming training mission plan:'), 2)
                self.assertIn('RuntimeError: Simulated interruption', log)
                self.assertIn('Training mission plan complete.', log)
                records = module.EventLog(root / 'output' / TEST_RUN_NAME / 'training_log.jsonl', str(checkpoints))
                self.assertEqual([e['result']['global_epoch'] for e in records.events('validation_result')],
                                 [1, 2, 3, 4])
                recovered = [e for e in records.events('epoch_completed') if e['global_epoch'] == 3]
                self.assertEqual(len(recovered), 1)
                self.assertTrue(recovered[0]['recovered_from_checkpoint'])
                self.assertEqual(latest['global_epoch'], 4)
                self.assertEqual(completed_candidate.stat().st_mtime_ns, saved_time)
                self.assertFalse(training_path.exists())

    def test_tiny_end_to_end_run_replaces_training_sets_and_selects_final(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'output').mkdir()
            (root / 'scripts/mission_control/mission_plan').mkdir(parents=True)
            path = root / 'scripts/mission_control/mission_plan' / 'smoke.toml'
            text = mission_plan_text(directory)
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
            module.run_mission_plan(path, root=root, commit='c' * 40, require_clean=False)
            network = root / 'output' / TEST_RUN_NAME
            log = (root / 'output' / TEST_RUN_NAME / 'training_log.jsonl').read_text()
            for event in ('global epoch 2', 'Validation performance:', 'Test performance:',
                          'normalized_rmse', 'Training precision:', 'Healthy checkpoint saved:',
                          'Background generation complete', 'Training mission plan complete.'):
                self.assertIn(event, log)
            self.assertTrue((network / 'run.json').is_file())
            self.assertEqual(len(list(root.rglob('source.tar.gz'))), 1)
            self.assertEqual(len(list((network / 'checkpoints' / 'candidates').glob('*.pt'))), 2)
            self.assertTrue((network / 'final' / 'model.pt').is_file())
            events = module.EventLog(root / 'output' / TEST_RUN_NAME / 'training_log.jsonl',
                                     str(network / 'checkpoints'))
            self.assertTrue(events.events('validation_completed'))
            self.assertTrue(events.events('test_result'))
            self.assertFalse((root / 'output' / 'training_log.jsonl').exists())
            self.assertEqual({event['run_id'] for event in events.read()},
                             {str(network / 'checkpoints')})
            self.assertEqual(events.events('training_stopped')[-1]['reason'], 'training_limit')
            for epoch in events.events('epoch_completed'):
                self.assertGreater(epoch['training_duration_s'], 0)
                self.assertGreaterEqual(epoch['epoch_wall_duration_s'], epoch['training_duration_s'])
                self.assertGreater(epoch['checkpoint_duration_s'], 0)
                self.assertEqual(epoch['presentations'], 2)
            measurement = events.events('test_result')[-1]['metrics']
            for key in ('speed_relative_mae_pct', 'vector_tip_relative_mae_pct',
                        'heading_relative_mae_pct', 'heading_cosine_error_pct',
                        'evaluation_duration_s', 'global_epoch', 'dataset', 'metric_definitions'):
                self.assertIn(key, measurement)
            for filename in ('validation-progress.json', 'validation-results.json'):
                self.assertFalse((network / 'final' / filename).exists())
            saved_test = json.loads((network / 'final' / 'test-results.json').read_text())
            self.assertEqual(saved_test['version'], 1)
            self.assertEqual(set(saved_test['identity']), {
                'model_sha256', 'dataset_sha256', 'evaluation', 'evaluator_git_commit', 'split'})
            for filename in ('precision-fallback.json', 'numerical-failure.json'):
                self.assertFalse((network / 'checkpoints' / filename).exists())
            self.assertFalse((root / 'stimuli' / 'training').exists())
            self.assertTrue((root / 'stimuli' / 'validation').exists())
            self.assertTrue((root / 'stimuli' / 'test' / module.DATASET_STATE).is_file())
            progress_spec = importlib.util.spec_from_file_location(
                'evaluate_progress', module.ROOT / 'scripts' / 'archive' / 'evaluate_progress.py')
            progress = importlib.util.module_from_spec(progress_spec)
            progress_spec.loader.exec_module(progress)
            test_path = root / 'stimuli' / 'test'
            state = json.loads((test_path / module.DATASET_STATE).read_text())
            report_path = root / 'reports'
            args = ['evaluate_progress', '--checkpoints', str(network / 'checkpoints'),
                    '--data', str(test_path / state['cache']), '--split', 'test',
                    '--samples', '0', '--no-plots', '--output', str(report_path)]
            with patch('sys.argv', args), patch.object(
                    progress, 'committed_revision', return_value='c' * 40):
                progress.main()
            report = next(report_path.rglob('progress.md')).read_text()
            self.assertIn('epoch-000001.pt', report)
            self.assertIn('epoch-000002.pt', report)
            measurements = events.events('progress_evaluation_result')
            self.assertEqual(len(measurements), 2)
            self.assertEqual(measurements[0]['split'], 'test')
            self.assertIn('breakdowns', measurements[0])
            self.assertFalse(list(report_path.rglob('*.json')))
            report_spec = importlib.util.spec_from_file_location(
                'report_training_log', module.ROOT / 'scripts/mission_control/backend/report_training_log.py')
            report_module = importlib.util.module_from_spec(report_spec)
            report_spec.loader.exec_module(report_module)
            # No data/checkpoints needed: the report consumes only event records.
            log_report = report_module.report(events.read(), str(network / 'checkpoints'))
            self.assertIn('Wall s incl. retries/save', log_report)
            self.assertIn('Cosine error %', log_report)
            self.assertIn('test / session', log_report)


if __name__ == '__main__':
    unittest.main()
