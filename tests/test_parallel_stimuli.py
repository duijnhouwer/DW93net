import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from test_training_recipe import module, recipe_text, DETECTOR
from parallel_stimuli import Prefetch, Throttle


class ParallelStimuliTests(unittest.TestCase):
    def test_full_force_first_set_uses_all_available_cpus_but_one_and_restores(self):
        with tempfile.TemporaryDirectory() as directory:
            recipe = dict(training_sets={'path': str(Path(directory) / 'train')},
                          optimization={'num_workers': 0},
                          parallel_generation={'enabled': True,
                                               'full_force_first_set': True})
            with patch('parallel_stimuli.os.sched_getaffinity', return_value=set(range(8))):
                prefetch = Prefetch(recipe)
                with (patch('torch.get_num_threads', return_value=3),
                      patch('torch.set_num_threads') as set_threads):
                    with prefetch.first_set_generation(0):
                        pass
                    self.assertEqual([call.args[0] for call in set_threads.call_args_list], [7, 3])
                    set_threads.reset_mock()
                    with prefetch.first_set_generation(1):
                        pass
                    set_threads.assert_not_called()
                    with self.assertRaisesRegex(RuntimeError, 'generation failed'):
                        with prefetch.first_set_generation(0):
                            raise RuntimeError('generation failed')
                    self.assertEqual([call.args[0] for call in set_threads.call_args_list], [7, 3])

    def test_interrupt_stops_real_worker_and_removes_partial_set(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'output').mkdir()
            path = Path(directory) / 'recipe.toml'
            path.write_text(recipe_text(directory) + '\n[parallel_generation]\nenabled = true\n')
            recipe = module.load_recipe(path)
            prefetch = Prefetch(recipe)
            try:
                prefetch.start(dict(seed=1001, n_samples=1000000, _detector=DETECTOR))
                process = prefetch.process
                deadline = time.monotonic() + 20
                while not (prefetch.stage / module.MANAGED_MARKER).exists():
                    self.assertIsNone(process.poll())
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(.02)
                prefetch.close()
                self.assertIsNotNone(process.poll())
                self.assertFalse(prefetch.stage.exists())
            finally:
                prefetch.close()

    def test_throttle_requires_sustained_pressure_and_recovers_slowly(self):
        controller = Throttle(1, 4)
        controller.threads = 4
        self.assertEqual(controller.observe(.5), 4)
        self.assertEqual(controller.observe(.5), 2)
        self.assertEqual(controller.observe(.5), 2)
        self.assertEqual(controller.observe(.5), 1)
        for _ in range(3):
            self.assertEqual(controller.observe(.01), 1)
        self.assertEqual(controller.observe(.01), 2)
        for _ in range(20):
            controller.observe(.01)
        self.assertEqual(controller.threads, 4)

    def test_real_child_matches_serial_and_reuses_completed_staging(self):
        self._assert_real_child_matches_serial(True)

    def test_high_resolution_dog_child_matches_serial(self):
        self._assert_real_child_matches_serial(False)

    def _assert_real_child_matches_serial(self, post):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'output').mkdir()
            path = root / 'recipe.toml'
            path.write_text(recipe_text(directory) + '''
[parallel_generation]
enabled = true
min_threads = 1
max_threads = 2
''')
            recipe = module.load_recipe(path)
            recipe['stimuli'].update(fov_deg=3, n_world_dots=20, x_supersampling=4,
                                    apply_dog_filter_post_downsample=post)
            section = dict(seed=1001, n_samples=4, _detector=DETECTOR)
            prefetch = Prefetch(recipe)
            try:
                prefetch.start(section)
                self.assertEqual(prefetch.process.wait(timeout=30), 0)
                prefetch.close()
                self.assertTrue((prefetch.stage / module.DATASET_STATE).is_file())
                recovered = Prefetch(recipe)
                destination = root / 'stimuli' / 'training'
                self.assertTrue(recovered.acquire(destination, section))
                state = json.loads((destination / module.DATASET_STATE).read_text())
                serial = module._prepare_dataset(root / 'serial', 'train', section,
                                                  recipe['stimuli'], persistent=False)
                staged = destination / state['cache']
                left = sorted((staged / 'train').glob('*.npy'))
                right = sorted((serial / 'train').glob('*.npy'))
                self.assertEqual(len(left), len(right))
                self.assertTrue(left)
                for a, b in zip(left, right):
                    np.testing.assert_array_equal(np.load(a), np.load(b))
            finally:
                prefetch.close()

    def test_failed_child_is_not_promoted_and_partial_data_is_cleaned(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as directory:
            recipe = dict(training_sets={'path': str(Path(directory) / 'train')},
                          optimization={'num_workers': 0}, parallel_generation={'enabled': True})
            prefetch = Prefetch(recipe)
            prefetch.stage.mkdir()
            (prefetch.stage / module.MANAGED_MARKER).touch()
            prefetch.owns_stage = True
            prefetch.process = Mock()
            prefetch.process.wait.return_value = 1
            prefetch.process.poll.return_value = 1
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                prefetch.acquire(Path(directory) / 'train', {})
            prefetch.close()
            self.assertFalse(prefetch.stage.exists())
