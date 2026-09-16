import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

from features.dog_motion import DoGFilter
from stimuli.generate_flow_xt_data import (
    StimConfig, generate_flow_xt_stim, generate_dog_dataset,
)


class OneStepStimulusTests(unittest.TestCase):
    def test_seed_reproduces_direct_dog_bundles_without_raw_files(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            outputs = []
            for bundle_size, root in enumerate((first, second), 1):
                outputs.append(generate_dog_dataset(
                    output_root=root, n_samples=3,
                    speed_range=(0.001, 0.006), heading_deg_range=(-180, 180),
                    rot_deg_range=(0, 0), role='train', seed=12345,
                    dog_bundle_size=bundle_size,
                    config=StimConfig(stim_wid_px=180, n_world_dots=20,
                                      x_supersampling=4)))
            first_cache, second_cache = map(Path, outputs)
            self.assertFalse(list(Path(first).rglob('*.pt')))
            self.assertFalse(list(Path(second).rglob('*.pt')))
            from stimuli.load_flow_xt_data import DoGDataset
            left = DoGDataset(first_cache / 'train')
            right = DoGDataset(second_cache / 'train')
            self.assertEqual(left.files, right.files)
            for index in range(len(left)):
                torch.testing.assert_close(left[index][0], right[index][0], rtol=0, atol=0)
                torch.testing.assert_close(left[index][1], right[index][1], rtol=0, atol=0)
                self.assertEqual(left[index][0].dtype, torch.float32)
            metadata = json.loads((first_cache / 'dog.json').read_text())
            self.assertEqual(metadata['storage_format'], 'dog-contrast-bundles-v1')

    def test_direct_dog_cache_matches_byte_decoded_filter(self):
        intensities = torch.linspace(0, 1, 98, dtype=torch.float32).reshape(2, 49)
        with tempfile.TemporaryDirectory() as root, patch(
                'stimuli.generate_flow_xt_data.generate_flow_xt_stim',
                return_value=intensities):
            cache = Path(generate_dog_dataset(
                output_root=root, n_samples=1, role='train', seed=42,
                dog_bundle_size=1,
                config=StimConfig(stim_wid_px=49, n_world_dots=4,
                                  x_supersampling=1)))
            from stimuli.load_flow_xt_data import DoGDataset
            saved, _ = DoGDataset(cache / 'train')[0]
        decoded = (intensities * 255).round().to(torch.uint8).float() / 255
        expected = DoGFilter()(decoded)[0, 0]
        torch.testing.assert_close(saved, expected, rtol=0, atol=0)

    def test_generation_writes_only_dog_bundles(self):
        for superimpose in (False, True):
            with self.subTest(superimpose=superimpose), tempfile.TemporaryDirectory() as root:
                cache = Path(generate_dog_dataset(
                    output_root=root, n_samples=2,
                    speed_range=(0.001, 0.002), heading_deg_range=(0, 1),
                    rot_deg_range=(0, 0), role='train', seed=42,
                    dog_bundle_size=2,
                    config=StimConfig(stim_wid_px=128, n_world_dots=120,
                                      x_supersampling=10, superimpose=superimpose)))
                self.assertFalse(list(Path(root).rglob('*.pt')))
                self.assertEqual(len(list((cache / 'train').glob('*.npy'))), 1)
                metadata = json.loads((cache / 'dog.json').read_text())['stimulus_generation']
                self.assertEqual(metadata['speed_range']['low_inclusive'], 0.001)
                self.assertEqual(metadata['speed_range']['high_exclusive'], 0.002)
                self.assertEqual(metadata['heading_deg_range']['convention'],
                                 '0=+Z_forward,-90=-X_left,+90=+X_right')
                self.assertEqual(metadata['role'], 'train')

    def test_polar_sampling_avoids_zero_speed_and_obeys_heading_convention(self):
        with tempfile.TemporaryDirectory() as root:
            cache = Path(generate_dog_dataset(
                output_root=root, n_samples=40, speed_range=(0.001, 0.015),
                heading_deg_range=(-180, 180), rot_deg_range=(0, 0),
                role='train', seed=17, dog_bundle_size=40,
                config=StimConfig(stim_wid_px=32, n_world_dots=4,
                                  x_supersampling=1)))
            self.assertIn('nspeed=1000000,15000000', cache.parent.name)
            self.assertIn('nheading=-180000000000,180000000000', cache.parent.name)
            stored = np.load(next((cache / 'train').glob('*.npy')))
            for dx, dz, _ in stored['target']:
                speed = np.hypot(dx, dz)
                heading = np.degrees(np.arctan2(dx, dz))
                self.assertGreaterEqual(speed, 0.001 - 1e-9)
                self.assertLess(speed, 0.015 + 1e-9)
                self.assertGreaterEqual(heading, -180)
                self.assertLess(heading, 180)

        cardinal = [(0, 0, 1), (-90, -1, 0), (90, 1, 0)]
        for heading, expected_dx, expected_dz in cardinal:
            radians = np.deg2rad(heading)
            self.assertAlmostEqual(np.sin(radians), expected_dx)
            self.assertAlmostEqual(np.cos(radians), expected_dz)

    def test_equal_range_endpoints_generate_fixed_values(self):
        with tempfile.TemporaryDirectory() as root:
            cache = Path(generate_dog_dataset(
                output_root=root, n_samples=3, speed_range=(0.005, 0.005),
                heading_deg_range=(-90, -90), rot_deg_range=(0, 0),
                role='train', seed=23, dog_bundle_size=3,
                config=StimConfig(stim_wid_px=32, n_world_dots=4,
                                  x_supersampling=1)))
            self.assertIn('nspeed=5000000,5000000', cache.parent.name)
            self.assertIn('nheading=-90000000000,-90000000000', cache.parent.name)
            targets = np.load(next((cache / 'train').glob('*.npy')))['target']
            expected = np.asarray([[-.005, 0, 0]] * 3, dtype=np.float32)
            np.testing.assert_array_equal(targets, expected)

            metadata = json.loads((cache / 'dog.json').read_text())['stimulus_generation']
            self.assertEqual(metadata['speed_range']['value'], 0.005)
            self.assertEqual(metadata['speed_range']['sampling'], 'fixed')
            self.assertEqual(metadata['heading_deg_range']['value'], -90)
            self.assertEqual(metadata['rot_deg_range']['value'], 0)

    def test_folder_polar_ranges_determine_component_ranges(self):
        from stimuli.load_flow_xt_data import (extract_ranges_from_foldername,
                                                stimulus_set_name)

        ranges = extract_ranges_from_foldername(
            'nspeed=5000000,5000000_'
            'nheading=-180000000000,180000000000_nrot=0,0')
        self.assertAlmostEqual(ranges['dx'], 0.01)
        self.assertAlmostEqual(ranges['dz'], 0.01)
        self.assertEqual(ranges['rot_deg'], 0)

        quadrant = extract_ranges_from_foldername(
            'nspeed=5000000,5000000_'
            'nheading=0,90000000000_nrot=-1000000000,1000000000')
        self.assertAlmostEqual(quadrant['dx'], 0.005)
        self.assertAlmostEqual(quadrant['dz'], 0.005)
        self.assertEqual(quadrant['rot_deg'], 2)

        with self.assertRaises(ValueError):
            extract_ranges_from_foldername(
                'nuspeed=5000000,5000000_nheading=0,0_nrot=0,0')
        self.assertEqual(stimulus_set_name('/tmp/parameters/dog'), 'parameters')

    def test_rejects_zero_speed_and_invalid_heading_ranges(self):
        config = StimConfig(stim_wid_px=32, n_world_dots=4, x_supersampling=1)
        with tempfile.TemporaryDirectory() as root:
            for speeds, headings in [((0, .01), (-180, 180)),
                                     (0, (-180, 180)),
                                     ((.001, .01), 0),
                                     ((.001, .01), (-181, 180)),
                                     ((.001, .01), (-180, 181))]:
                with self.subTest(speeds=speeds, headings=headings), self.assertRaises(ValueError):
                    generate_dog_dataset(
                        output_root=root, n_samples=1, speed_range=speeds,
                        heading_deg_range=headings, rot_deg_range=(0, 0),
                        role='train', seed=1, config=config)
            with self.assertRaisesRegex(ValueError, 'role must be'):
                generate_dog_dataset(
                    output_root=root, n_samples=1, speed_range=(.001, .01),
                    heading_deg_range=(-180, 180), rot_deg_range=(0, 0),
                    role='mixed', seed=1, config=config)

    def test_default_supersampling_aligns_with_byte_levels(self):
        config = StimConfig(stim_wid_px=128, n_world_dots=120)
        self.assertEqual(config.x_supersampling, 255)
        xt = generate_flow_xt_stim(config=config, rng=np.random.default_rng(42))
        encoded = (xt * 255).round().to(torch.uint8)
        # The rendered 1/255 levels survive storage, to float32 precision.
        torch.testing.assert_close(encoded.float() / 255, xt, rtol=0, atol=1e-7)
        self.assertTrue(((encoded > 0) & (encoded < 255)).any())

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
