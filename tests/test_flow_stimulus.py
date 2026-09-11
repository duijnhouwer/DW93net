import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

from stimuli.generate_flow_xt_data import (
    StimConfig, generate_flow_xt_stim,
    generate_flow_xt_dataset, generate_and_precompute,
)


class OneStepStimulusTests(unittest.TestCase):
    def test_combined_generation_precomputes_sibling_cache(self):
        with tempfile.TemporaryDirectory() as root, patch(
                'stimuli.generate_flow_xt_data.precompute_dataset') as precompute:
            raw, products = generate_and_precompute(
                output_root=root, n_samples=1,
                speed_range=(0.005, 0.005), heading_deg_range=(0, 0),
                rot_deg_range=(0, 0), split=(1, 0, 0), seed=4,
                config=StimConfig(stim_wid_px=32, n_world_dots=4,
                                  x_supersampling=1))

            self.assertEqual(Path(raw).name, 'raw')
            expected = Path(raw).parent / 'reichardt'
            self.assertEqual(Path(products), expected)
            precompute.assert_called_once_with(Path(raw), expected)

    def test_dataset_saves_one_pair_per_file(self):
        for superimpose in (False, True):
            with self.subTest(superimpose=superimpose), tempfile.TemporaryDirectory() as root:
                folder = generate_flow_xt_dataset(
                    output_root=root, n_samples=2,
                    speed_range=(0.001, 0.002), heading_deg_range=(0, 1),
                    rot_deg_range=(0, 0),
                    split=(1, 0, 0), seed=42,
                    config=StimConfig(stim_wid_px=128, n_world_dots=120,
                                      x_supersampling=10, superimpose=superimpose))
                files = list(Path(folder).rglob('*.pt'))
                self.assertEqual(len(files), 2)
                for path in files:
                    pairs = torch.load(path, weights_only=True)
                    self.assertEqual(pairs.shape, (1, 2, 128))
                    self.assertEqual(pairs.dtype, torch.uint8)
                    self.assertTrue(((pairs > 0) & (pairs < 255)).any())
                metadata = json.loads((Path(folder) / 'generation.json').read_text())
                self.assertEqual(metadata['speed_range']['low_inclusive'], 0.001)
                self.assertEqual(metadata['speed_range']['high_exclusive'], 0.002)
                self.assertEqual(metadata['heading_deg_range']['convention'],
                                 '0=+Z_forward,-90=-X_left,+90=+X_right')

    def test_polar_sampling_avoids_zero_speed_and_obeys_heading_convention(self):
        from stimuli.load_flow_xt_data import parse_camera_motion_from_filename

        with tempfile.TemporaryDirectory() as root:
            folder = generate_flow_xt_dataset(
                output_root=root, n_samples=40, speed_range=(0.001, 0.015),
                heading_deg_range=(-180, 180), rot_deg_range=(0, 0),
                split=(1, 0, 0), seed=17,
                config=StimConfig(stim_wid_px=32, n_world_dots=4,
                                  x_supersampling=1))
            self.assertIn('nspeed=1000000,15000000', Path(folder).parent.name)
            self.assertIn('nheading=-180000000000,180000000000', Path(folder).parent.name)
            for path in Path(folder, 'train').glob('*.pt'):
                motion = parse_camera_motion_from_filename(path.name)
                speed = np.hypot(motion['dx'], motion['dz'])
                heading = np.degrees(np.arctan2(motion['dx'], motion['dz']))
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
        from stimuli.load_flow_xt_data import parse_camera_motion_from_filename

        with tempfile.TemporaryDirectory() as root:
            folder = generate_flow_xt_dataset(
                output_root=root, n_samples=3, speed_range=(0.005, 0.005),
                heading_deg_range=(-90, -90), rot_deg_range=(0, 0),
                split=(1, 0, 0), seed=23,
                config=StimConfig(stim_wid_px=32, n_world_dots=4,
                                  x_supersampling=1))
            self.assertIn('nspeed=5000000,5000000', Path(folder).parent.name)
            self.assertIn('nheading=-90000000000,-90000000000', Path(folder).parent.name)
            for path in Path(folder, 'train').glob('*.pt'):
                motion = parse_camera_motion_from_filename(path.name)
                self.assertEqual(motion['dx'], -0.005)
                self.assertEqual(motion['dz'], 0)
                self.assertEqual(motion['rot_deg'], 0)

            metadata = json.loads((Path(folder) / 'generation.json').read_text())
            self.assertEqual(metadata['speed_range']['value'], 0.005)
            self.assertEqual(metadata['speed_range']['sampling'], 'fixed')
            self.assertEqual(metadata['heading_deg_range']['value'], -90)
            self.assertEqual(metadata['rot_deg_range']['value'], 0)

    def test_folder_polar_ranges_determine_component_ranges(self):
        from stimuli.load_flow_xt_data import (extract_ranges_from_foldername,
                                            parse_camera_motion_from_filename)

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
        with self.assertRaises(ValueError):
            parse_camera_motion_from_filename(
                'flow_ndx=0_ndz=0_nrot=0_id=0.pt')

    def test_rejects_zero_speed_and_invalid_heading_ranges(self):
        config = StimConfig(stim_wid_px=32, n_world_dots=4, x_supersampling=1)
        with tempfile.TemporaryDirectory() as root:
            for speeds, headings in [((0, .01), (-180, 180)),
                                     (0, (-180, 180)),
                                     ((.001, .01), 0),
                                     ((.001, .01), (-181, 180)),
                                     ((.001, .01), (-180, 181))]:
                with self.subTest(speeds=speeds, headings=headings), self.assertRaises(ValueError):
                    generate_flow_xt_dataset(root, 1, speeds, headings, (0, 0),
                                             (1, 0, 0), 1, config)

    def test_default_supersampling_aligns_with_byte_levels(self):
        config = StimConfig(stim_wid_px=128, n_world_dots=120)
        self.assertEqual(config.x_supersampling, 255)
        xt = generate_flow_xt_stim(config=config, rng=np.random.default_rng(42))
        encoded = (xt * 255).round().to(torch.uint8)
        # The rendered 1/255 levels survive storage, to float32 precision.
        torch.testing.assert_close(encoded.float() / 255, xt, rtol=0, atol=1e-7)
        self.assertTrue(((encoded > 0) & (encoded < 255)).any())

    def test_viewer_uses_normalized_byte_intensities(self):
        from stimuli.show_flow_xt_data import show_xt_sample
        levels = torch.tensor([[[0, 1, 128, 255], [255, 128, 1, 0]]], dtype=torch.uint8)
        expected = levels.squeeze(0).float().numpy() / 255
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ndx=0_ndz=0_nrot=0_id=0.pt'
            torch.save(levels, path)
            with patch('stimuli.show_flow_xt_data.plt') as plot:
                show_xt_sample(directory, 0)
            np.testing.assert_allclose(plot.imshow.call_args.args[0], expected)
            self.assertEqual(plot.imshow.call_args.kwargs['vmax'], 1)

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
