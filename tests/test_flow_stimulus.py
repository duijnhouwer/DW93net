import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

from features.dog_motion import DoGFilter
from stimuli.generate_flow_xt_data import (
    StimConfig, generate_flow_xt_stim, generate_dog_dataset, sample_motion,
    _generate_flow_xt_samples,
)


class OneStepStimulusTests(unittest.TestCase):
    def test_dog_order_matches_direct_reference_and_preserves_labels(self):
        from dataclasses import replace
        from stimuli.load_flow_xt_data import DoGDataset
        from stimuli.metadata import data_contract
        torch.set_num_threads(1)
        config = StimConfig(stim_wid_px=49, n_world_dots=20, x_supersampling=7)
        with tempfile.TemporaryDirectory() as directory:
            datasets = []
            for post in (True, False):
                cfg = replace(config, apply_dog_filter_after_downsample=post)
                cache = generate_dog_dataset(output_root=str(Path(directory) / str(post)),
                    n_samples=2, seed=42, config=cfg, dog_bundle_size=2,
                    speed_range=(.001, .006), heading_deg_range=(-180, 180), rot_deg_range=(0, 0), role='train')
                datasets.append(DoGDataset(Path(cache) / 'train'))
                rng = np.random.default_rng(42)
                for i in range(2):
                    dx, dz, yaw = sample_motion(rng, (.001, .006), (-180, 180), (0, 0))
                    raw = generate_flow_xt_stim([dx, dz], yaw, cfg, rng, downsample=post)
                    factor = 1 if post else 7
                    expected = DoGFilter(sigma_center=factor, sigma_surround=3*factor)(raw)[0, 0]
                    if not post:
                        expected = expected.reshape(2, 31, 7).mean(dim=-1)
                    saved, label = datasets[-1][i]
                    torch.testing.assert_close(saved, expected, atol=1e-6, rtol=1e-5)
                    self.assertEqual(saved.shape, (2, 31))
                    self.assertEqual(saved.dtype, torch.float32)
            for i in range(2):
                torch.testing.assert_close(datasets[0][i][1], datasets[1][i][1], rtol=0, atol=0)
            self.assertNotEqual(data_contract(datasets[0].storage_metadata),
                                data_contract(datasets[1].storage_metadata))

    def test_fov_bounds(self):
        for fov in (0.001, 120, 240, 360):
            self.assertEqual(StimConfig(fov_deg=fov).fov_deg, fov)
        for fov in (0, -1, 360.001, float('nan'), float('inf'), float('-inf'), True):
            with self.subTest(fov=fov), self.assertRaisesRegex(ValueError, 'fov_deg'):
                StimConfig(fov_deg=fov)

    def test_wide_fov_uses_full_circle_angles(self):
        def render(angle_deg, fov_deg):
            angle = np.deg2rad(angle_deg)
            rng = Mock()
            rng.uniform.return_value = np.array([[np.sin(angle), np.cos(angle)]]) * .5
            return generate_flow_xt_stim(
                config=StimConfig(fov_deg=fov_deg, stim_wid_px=721,
                                  n_world_dots=1, x_supersampling=1),
                rng=rng)[0]

        # A 240-degree view includes dots behind the camera through +/-120
        # degrees, but not dots beyond that angular boundary.
        self.assertEqual(render(100, 240).sum().item(), 1)
        self.assertEqual(render(-100, 240).sum().item(), 1)
        self.assertEqual(render(130, 240).sum().item(), 0)
        self.assertEqual(render(-130, 240).sum().item(), 0)

        # A full-circle view has no excluded seam: directly behind and points
        # approaching it from either side all remain visible.
        self.assertEqual(render(180, 360).sum().item(), 1)
        self.assertEqual(render(-180, 360).sum().item(), 1)
        self.assertEqual(render(179, 360).sum().item(), 1)
        self.assertEqual(render(-179, 360).sum().item(), 1)

    def test_float32_labels_are_exact_renderer_inputs_and_replay_matches(self):
        for speeds in ((1e-12, 2e-12), (1e-12, 1e-12), (.001, .006)):
            with self.subTest(speeds=speeds):
                recorded, saved = [], []
                original = generate_flow_xt_stim
                def render(**kwargs):
                    recorded.append(tuple(kwargs['cam_trans_xz']) + (kwargs['cam_rot_deg'],))
                    return original(**kwargs)
                def consume(*args):
                    saved.append(np.asarray(args[-1], dtype=np.float32))
                config = StimConfig(stim_wid_px=49, n_world_dots=4, x_supersampling=1)
                with patch('stimuli.generate_flow_xt_data.generate_flow_xt_stim', side_effect=render):
                    _generate_flow_xt_samples(n_samples=3, seed=5, config=config,
                        speed_range=speeds, heading_deg_range=(-180, 180),
                        rot_deg_range=(-1e-12, 1e-12), sample_consumer=consume)
                rng = np.random.default_rng(5)
                for actual, stored in zip(recorded, saved):
                    self.assertEqual(actual, tuple(stored.astype(np.float64)))
                    self.assertTrue(np.any(stored[:2] != 0))
                    replay = sample_motion(rng, speeds, (-180, 180), (-1e-12, 1e-12))
                    self.assertEqual(replay, actual)
                    rng.uniform(-1, 1, size=(config.n_world_dots, 2))

    def test_float32_motion_rejects_underflow_overflow_and_nonfinite_values(self):
        for speed in (1e-50, 1e40, float('inf'), float('nan')):
            with self.subTest(speed=speed), self.assertRaisesRegex(ValueError, 'float32'):
                sample_motion(np.random.default_rng(0), (speed, speed), (0, 0), (0, 0))
        with self.assertRaisesRegex(ValueError, 'float32 targets'):
            sample_motion(np.random.default_rng(0), (.001, .001), (0, 0), (1e40, 1e40))
        for rotation in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                sample_motion(np.random.default_rng(0), (.001, .001), (0, 0), (rotation, rotation))

    def test_zero_speed_supports_rotation_only_motion(self):
        self.assertEqual(
            sample_motion(np.random.default_rng(0), (0, 0), (0, 0), (1, 1)),
            (0.0, 0.0, 1.0))
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            sample_motion(np.random.default_rng(0), (-.001, -.001), (0, 0), (1, 1))
        with tempfile.TemporaryDirectory() as root:
            cache = Path(generate_dog_dataset(
                output_root=root, n_samples=1, speed_range=(0, 0),
                heading_deg_range=(0, 0), rot_deg_range=(-1, 1), role='train', seed=42,
                dog_bundle_size=1,
                config=StimConfig(stim_wid_px=32, n_world_dots=4, x_supersampling=1)))
            from stimuli.metadata import motion_scales
            self.assertEqual(motion_scales(cache), (1.0, 1.0, 2.0))

    def test_near_exclusion_radius_bounds(self):
        for radius in (0, .04, .999999):
            self.assertEqual(StimConfig(near_exclusion_radius=radius).near_exclusion_radius, radius)
        for radius in (-.01, 1, 1.1, float('nan'), float('inf'), float('-inf'), True):
            with self.subTest(radius=radius), self.assertRaisesRegex(ValueError, 'near_exclusion_radius'):
                StimConfig(near_exclusion_radius=radius)

    def test_filter_uses_bounded_batches_including_partial_bundle(self):
        sizes = []
        original = DoGFilter.__call__

        def record(dog, frames):
            sizes.append(frames.shape[0])
            return original(dog, frames)

        with tempfile.TemporaryDirectory() as root, patch.object(DoGFilter, '__call__', record):
            cache = Path(generate_dog_dataset(
                output_root=root, n_samples=35, role='train', seed=42,
                dog_bundle_size=34, config=StimConfig(
                    stim_wid_px=49, n_world_dots=4, x_supersampling=1)))
            self.assertEqual(sizes, [16, 16, 2, 1])
            index = json.loads((cache / 'train' / 'bundles.json').read_text())
            self.assertEqual(len(index), 35)
            self.assertEqual([entry[2] for entry in index], list(range(34)) + [0])

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
            self.assertEqual(left.sample_ids, right.sample_ids)
            for index in range(len(left)):
                torch.testing.assert_close(left[index][0], right[index][0], rtol=0, atol=0)
                torch.testing.assert_close(left[index][1], right[index][1], rtol=0, atol=0)
                self.assertEqual(left[index][0].dtype, torch.float32)
            metadata = json.loads((first_cache / 'dog.json').read_text())
            self.assertEqual(metadata['storage_format'], 'dog-contrast-bundles-v2')

    def test_direct_dog_cache_preserves_float32_intensities(self):
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
        expected = DoGFilter()(intensities)[0, 0]
        torch.testing.assert_close(saved, expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(saved, DoGFilter()(decoded)[0, 0]))

    def test_supersampling_above_255_preserves_sub_byte_levels_through_generation(self):
        config = StimConfig(stim_wid_px=49, n_world_dots=20, x_supersampling=1024)
        observed = []
        render = generate_flow_xt_stim
        def record(**kwargs):
            frames = render(**kwargs)
            observed.append(frames)
            return frames
        received = []
        with patch('stimuli.generate_flow_xt_data.generate_flow_xt_stim', side_effect=record):
            _generate_flow_xt_samples(n_samples=2, seed=42, config=config,
                rot_deg_range=(0, 0), sample_consumer=lambda *args: received.append(args[-2]))
        for raw, frames in zip(observed, received):
            self.assertEqual(frames.dtype, torch.float32)
            torch.testing.assert_close(frames[0], raw, rtol=0, atol=0)
            quantized = (raw * 255).round() / 255
            self.assertTrue(torch.any(raw != quantized))

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
            targets = np.load(next((cache / 'train').glob('*.npy')))['target']
            expected = np.asarray([[-.005, 0, 0]] * 3, dtype=np.float32)
            np.testing.assert_array_equal(targets, expected)

            metadata = json.loads((cache / 'dog.json').read_text())['stimulus_generation']
            self.assertEqual(metadata['speed_range']['value'], 0.005)
            self.assertEqual(metadata['speed_range']['sampling'], 'fixed')
            self.assertEqual(metadata['heading_deg_range']['value'], -90)
            self.assertEqual(metadata['rot_deg_range']['value'], 0)

    def test_metadata_polar_ranges_determine_component_ranges(self):
        from stimuli.metadata import motion_ranges
        generation = dict(speed_range=dict(sampling='fixed', value=.005),
                          heading_deg_range=dict(sampling='uniform_half_open',
                                                 low_inclusive=-180, high_exclusive=180),
                          rot_deg_range=dict(sampling='fixed', value=0))
        ranges = motion_ranges(generation)
        self.assertAlmostEqual(ranges['dx'], .01)
        self.assertAlmostEqual(ranges['dz'], .01)
        self.assertEqual(ranges['rot_deg'], 0)
        generation['heading_deg_range'].update(low_inclusive=0, high_exclusive=90)
        quadrant = motion_ranges(generation)
        self.assertAlmostEqual(quadrant['dx'], .005)
        self.assertAlmostEqual(quadrant['dz'], .005)

    def test_rejects_negative_speed_and_invalid_heading_ranges(self):
        config = StimConfig(stim_wid_px=32, n_world_dots=4, x_supersampling=1)
        with tempfile.TemporaryDirectory() as root:
            for speeds, headings in [((-.001, .01), (-180, 180)),
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
