import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import torch

from data.generate_flow_xt_data import (
    StimConfig, generate_flow_xt_stim,
    generate_flow_xt_dataset,
)


class OneStepStimulusTests(unittest.TestCase):
    def test_dataset_saves_one_pair_per_file(self):
        for superimpose in (False, True):
            with self.subTest(superimpose=superimpose), tempfile.TemporaryDirectory() as root:
                folder = generate_flow_xt_dataset(
                    output_root=root, n_samples=2,
                    dx_range=(0, 0), dz_range=(0, 0), rot_deg_range=(0, 0),
                    split=(1, 0, 0), seed=42,
                    config=StimConfig(stim_wid_px=128, n_world_dots=120,
                                      x_supersampling=10, superimpose=superimpose))
                files = list(Path(folder).rglob('*.pt'))
                self.assertEqual(len(files), 2)
                for path in files:
                    pairs = torch.load(path, weights_only=True)
                    self.assertEqual(pairs.shape, (1, 2, 128))
                    torch.testing.assert_close(pairs[:, 0], pairs[:, 1], rtol=0, atol=0)
                    self.assertTrue(((pairs >= 0) & (pairs <= 1)).all())

    def test_zero_motion_preserves_dots_and_fractional_intensities(self):
        xt = generate_flow_xt_stim(
            cam_trans_xz=[0, 0], config=StimConfig(
                stim_wid_px=128, n_world_dots=120, x_supersampling=10),
            rng=np.random.default_rng(42))
        self.assertEqual(xt.shape, (2, 128))
        self.assertEqual(xt.dtype, torch.float32)
        torch.testing.assert_close(xt[0], xt[1], rtol=0, atol=0)
        self.assertTrue(((xt > 0) & (xt < 1)).any())

    def test_translation_then_rotation_matches_projected_angle(self):
        rng = Mock()
        rng.uniform.return_value = np.array([[0.1, 0.6]])
        config = StimConfig(stim_wid_px=1201, n_world_dots=1, x_supersampling=1)
        xt = generate_flow_xt_stim([0.02, 0.1], 5, config, rng=rng)
        # In this XZ convention, the dot's angle increases by cam_rot_deg.
        angles = [np.arctan2(0.1, 0.6), np.arctan2(0.08, 0.5) + np.deg2rad(5)]
        for frame, angle in zip(xt, angles):
            expected_pixel = int(np.floor((angle / np.deg2rad(60) + 1) * 600 - 0.5))
            self.assertEqual(frame.nonzero().flatten().tolist(), [expected_pixel])
        rng.uniform.assert_called_once_with(-1, 1, size=(1, 2))

    def test_departing_dot_is_not_wrapped_or_replaced(self):
        rng = Mock()
        rng.uniform.return_value = np.array([[0.0, 0.5]])
        xt = generate_flow_xt_stim(
            [0, 2], config=StimConfig(stim_wid_px=128, n_world_dots=1,
                                     x_supersampling=1), rng=rng)
        self.assertEqual(xt[0].sum().item(), 1)
        self.assertEqual(xt[1].sum().item(), 0)
        rng.uniform.assert_called_once_with(-1, 1, size=(1, 2))
