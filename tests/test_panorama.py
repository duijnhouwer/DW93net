"""Seam conventions and circular processing, including persisted training state."""
import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch.nn import functional as F

from features.spatial import resolve_boundary, circular_pad
from features.dog_motion import DoGFilter, SupersampledDoG
from features.reichardt import ReichardtBank
from models.flatland_flow import FlatlandFlowNet, V1Motion
from stimuli.generate_flow_xt_data import StimConfig, generate_flow_xt_stim, generate_dog_dataset
from stimuli.load_flow_xt_data import DoGDataset
from stimuli.metadata import motion_scales
from stimuli.dataset_identity import dataset_spec, storage_manifest
from train.train_flow_xt_net import RecipeTrainingSettings, train_recipe_set
from checkpoints import load_checkpoint, validate_checkpoint
from evaluation import inference_batches, summary_metrics
from test_training_recipe import module, recipe_text, DETECTOR


class PanoramaTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_boundary_policy_and_unsupported_filter_order(self):
        for fov in (120, 349, 359, 359.999999):
            self.assertEqual(resolve_boundary(fov), 'valid')
            self.assertEqual(StimConfig(fov_deg=fov, apply_dog_filter_after_downsample=False).boundary, 'valid')
        self.assertEqual(StimConfig(fov_deg=360).boundary, 'circular')
        for fov in (0, 361, float('nan'), True):
            with self.assertRaises(ValueError):
                resolve_boundary(fov)
        with self.assertRaisesRegex(ValueError, 'after_downsample'):
            StimConfig(fov_deg=360, apply_dog_filter_after_downsample=False)
        with self.assertRaisesRegex(ValueError, 'after downsampling'):
            SupersampledDoG(DoGFilter(boundary='circular'), 64, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'recipe.toml'
            path.write_text(recipe_text(directory).replace('[stimuli]',
                '[stimuli]\nfov_deg = 360\napply_dog_filter_after_downsample = false'))
            with self.assertRaisesRegex(ValueError, 'after_downsample'):
                module.load_recipe(path)

    def render_dot(self, angle, rotation=0, factor=1):
        theta = np.deg2rad(angle)
        rng = Mock()
        rng.uniform.return_value = np.array([[.5 * np.sin(theta), .5 * np.cos(theta)]])
        return generate_flow_xt_stim([0, 0], rotation,
            StimConfig(fov_deg=360, stim_wid_px=360, n_world_dots=1, x_supersampling=factor), rng)

    def test_renderer_half_open_bins_and_local_seam_crossing(self):
        for angle in (-180, 180):
            frame = self.render_dot(angle)[0]
            self.assertEqual(frame.nonzero().flatten().tolist(), [0])
        self.assertEqual(self.render_dot(179.9)[0].nonzero().flatten().tolist(), [359])
        self.assertEqual(self.render_dot(-179.9)[0].nonzero().flatten().tolist(), [0])
        # Every half-open angular bin is present exactly once.
        bins = [self.render_dot(-179.5 + i)[0].argmax().item() for i in range(360)]
        self.assertEqual(bins, list(range(360)))
        pair = self.render_dot(179.5, rotation=1)
        self.assertEqual(pair.nonzero()[:, 1].tolist(), [359, 0])
        torch.testing.assert_close(pair[1], pair[0].roll(1))
        for factor in (2, 7, 255):
            pair = self.render_dot(180, factor=factor)
            self.assertAlmostEqual(pair[0].sum().item(), 1., places=6)
            self.assertGreater(pair[0, 0].item(), 0)
            self.assertGreater(pair[0, -1].item(), 0)
            torch.testing.assert_close(pair[0], pair[1])

    def test_padding_handles_multiple_wraps_and_gradients(self):
        x = torch.tensor([[[1., 2., 3.]]], requires_grad=True)
        padded = circular_pad(x, 4, 5)
        torch.testing.assert_close(padded, x.repeat(1, 1, 5)[..., 2:14])
        padded.sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 4))

    def test_raster_margins_fold_overlapping_footprints_before_downsampling(self):
        # The rearward dots overlap each other across both edges. Their union
        # must stay white rather than adding brightness or erasing either dot.
        angles = np.deg2rad([-170, 170, 0])
        positions = .5 * np.column_stack((np.sin(angles), np.cos(angles)))
        rng = Mock()
        rng.uniform.return_value = positions
        config = StimConfig(fov_deg=360, stim_wid_px=8, n_world_dots=3, x_supersampling=4)
        raw = generate_flow_xt_stim([0, 0], 0, config, rng, downsample=False)
        expected = torch.zeros(2, 32)
        expected[:, [29, 30, 31, 0, 1, 2, 14, 15, 16, 17]] = 1
        torch.testing.assert_close(raw, expected, rtol=0, atol=0)
        downsampled = generate_flow_xt_stim([0, 0], 0, config, rng)
        torch.testing.assert_close(downsampled, expected.reshape(2, 8, 4).mean(-1), rtol=0, atol=0)
        # When the core is only one output pixel wide, both folded margins
        # land in that same pixel and must still saturate at one.
        narrow = generate_flow_xt_stim([0, 0], 0, replace(config, stim_wid_px=1), rng)
        torch.testing.assert_close(narrow, torch.ones(2, 1), rtol=0, atol=0)

    def test_dog_roll_equivariance_and_tiled_reference(self):
        dog = DoGFilter(boundary='circular')
        x = torch.randn(2, 1, 2, 64)
        result = dog(x)
        self.assertEqual(result.shape, x.shape)
        for shift in (1, 19, -3):
            torch.testing.assert_close(dog(x.roll(shift, -1)), result.roll(shift, -1))
        impulse = torch.zeros(2, 64)
        impulse[:, 0] = 1
        torch.testing.assert_close(dog(impulse.roll(32, -1)), dog(impulse).roll(32, -1))
        valid = DoGFilter()
        reference = valid(x.repeat(1, 1, 1, 3))[..., 64-dog.radius:128-dog.radius]
        torch.testing.assert_close(result, reference)
        self.assertEqual(valid(x).shape[-1], 46)
        torch.testing.assert_close(dog(torch.ones(2, 3)), torch.zeros(1, 1, 2, 3), atol=1e-6, rtol=0)

    def test_reichardt_crosses_seam_in_both_directions(self):
        bank = ReichardtBank([-1, 0, 1, 9], boundary='circular')
        for first, second, channel in ((7, 0, 2), (0, 7, 0)):
            x = torch.zeros(2, 8)
            x[0, first], x[1, second] = 2, 3
            result = bank(x)
            self.assertEqual(result.shape, (4, 1, 8))
            self.assertEqual(result[channel, 0, second], 6)
        x = torch.randn(2, 1, 2, 8)
        expected = torch.stack([x[:, 0, 0].roll(dx, -1) * x[:, 0, 1]
                                for dx in bank.dx_values], dim=1).unsqueeze(2)
        torch.testing.assert_close(bank(x), expected)
        torch.testing.assert_close(bank(x.roll(3, -1)), bank(x).roll(3, -1))

    def test_local_model_operators_match_periodic_reference(self):
        x = torch.randn(1, 3, 128)
        model = FlatlandFlowNet([-1, 0, 1], boundary='circular')
        for actual, rolled in zip(model.v1(x), model.v1(x.roll(1, -1))):
            torch.testing.assert_close(rolled, actual.roll(1, -1))
        pool = model.mt[0]
        torch.testing.assert_close(pool(x.roll(8, -1)), pool(x).roll(1, -1))
        reference = F.avg_pool1d(x.repeat(1, 1, 3)[..., 128-15:256+15], 31, stride=8)
        torch.testing.assert_close(pool(x), reference)
        for conv in (model.mt[3], model.mt[5]):
            narrow = torch.randn(1, 96, 16)
            torch.testing.assert_close(conv(narrow.roll(1, -1)), conv(narrow).roll(1, -1))
            padding = conv.padding[0]
            tiled = narrow.repeat(1, 1, 7)[..., 48-padding:64+padding]
            reference = F.conv1d(tiled, conv.weight, conv.bias, dilation=conv.dilation)
            torch.testing.assert_close(conv(narrow), reference)
        y = model(x.unsqueeze(2))
        self.assertEqual(y.shape, (1, 3))
        y.square().sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        # Valid operators are still the original PyTorch modules with identical numerics.
        valid = V1Motion()
        torch.testing.assert_close(valid.center(x), F.avg_pool1d(x, 15, 1, 7, count_include_pad=False))
        torch.testing.assert_close(valid.surround(x), F.avg_pool1d(x, 61, 1, 30, count_include_pad=False))

    def test_production_recipe_and_archive(self):
        root = Path(__file__).resolve().parents[1]
        # Git membership is covered by recipe-runner tests; this file may be untracked while developing.
        with patch.object(module.subprocess, 'check_output', return_value=b''):
            active = module.discover_recipe(root)
        self.assertEqual(active.stem, 'trans_plus_rot_panoramic')
        recipe = module.load_recipe(active)
        self.assertEqual(StimConfig(fov_deg=recipe['stimuli']['fov_deg']).stim_wid_px, 21600)
        for key, value in [('speed_range', [.001, .006]), ('heading_deg_range', [-180, 180]),
                           ('rot_deg_range', [-.5, .5])]:
            self.assertEqual(recipe['stimuli'][key], value)
        # Historical recipe is preserved; it predates the required Duffy-Wurtz section.
        archived = module.tomllib.loads((root / 'recipe/archive/rot_only.toml').read_text())
        self.assertEqual(recipe['detector'], archived['detector'])
        self.assertEqual(recipe['training_sets'], archived['training_sets'])

    def test_generation_training_resume_evaluation_and_metadata_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commit = 'c' * 40
            config = StimConfig(fov_deg=360, stim_wid_px=128, n_world_dots=20, x_supersampling=3)
            caches = {}
            for role in ('train', 'validate', 'test'):
                caches[role] = Path(generate_dog_dataset(output_root=root / role, n_samples=2,
                    role=role, seed=42, config=config, dog_bundle_size=2,
                    speed_range=(.001, .006), heading_deg_range=(-180, 180), rot_deg_range=(-.5, .5),
                    provenance={'git_commit': commit}))
            validation = DoGDataset(caches['validate'] / 'validate')
            def validate(checkpoint):
                return summary_metrics(inference_batches(checkpoint, validation, device='cpu', batch_size=2),
                                       motion_scales(caches['validate']))
            cfg = RecipeTrainingSettings(root_folder=str(caches['train']), checkpoint_dir=str(root / 'checkpoints'),
                git_commit=commit, training_set_index=0, training_seed=42, epochs_per_set=1, shuffle_seed=7,
                batch_size=2, num_workers=0, allow_amp=False, allow_tp32=False, dx_values=(-2, 0, 2))
            with patch.object(torch.cuda, 'is_available', return_value=False):
                train_recipe_set(cfg, validate_epoch=validate)
                train_recipe_set(replace(cfg, epochs_per_set=2), validate_epoch=validate)
            checkpoint = load_checkpoint(root / 'checkpoints/latest.pt', expected_commit=commit, resume=True)
            self.assertEqual(checkpoint['global_epoch'], 2)
            self.assertEqual(checkpoint['data_contract']['boundary'], 'circular')
            self.assertEqual(checkpoint['detector']['correlation']['boundary'], 'circular')
            obsolete = copy.deepcopy(checkpoint)
            obsolete['schema'] = 'dw93net-checkpoint-v3'
            with self.assertRaisesRegex(ValueError, 'regenerate'):
                validate_checkpoint(obsolete, expected_commit=commit)
            mismatched = copy.deepcopy(checkpoint)
            mismatched['data_contract']['boundary'] = 'valid'
            torch.save(mismatched, root / 'checkpoints/latest.pt')
            with patch.object(torch.cuda, 'is_available', return_value=False), self.assertRaisesRegex(ValueError, 'geometry'):
                train_recipe_set(replace(cfg, epochs_per_set=3), validate_epoch=validate)
            torch.save(checkpoint, root / 'checkpoints/latest.pt')
            dataset = DoGDataset(caches['test'] / 'test')
            self.assertEqual(dataset[0][0].shape, (2, 128))
            self.assertEqual(storage_manifest(caches['test'], 'test', 2)['metadata']['boundary'], 'circular')
            result = summary_metrics(inference_batches(checkpoint, dataset, device='cpu', batch_size=2),
                                     motion_scales(caches['test']))
            self.assertEqual(result['n_samples'], 2)
            self.assertTrue(np.isfinite(result['normalized_mse']))
            corrupted = copy.deepcopy(checkpoint)
            corrupted['detector']['correlation']['boundary'] = 'valid'
            with self.assertRaisesRegex(ValueError, 'correlation'):
                list(inference_batches(corrupted, dataset, device='cpu', batch_size=2))
            path = caches['test'] / 'dog.json'
            original = json.loads(path.read_text())
            for location, key, value in [((), 'storage_format', 'dog-contrast-bundles-v2'),
                    ((), 'format', 'dog-signed-contrast-v1'), ((), 'boundary', 'valid'),
                    (('dog',), 'boundary', 'valid'),
                    (('stimulus_generation', 'stimulus_config'), 'boundary', 'valid'),
                    ((), 'contrast_shape', [2, 110])]:
                changed = copy.deepcopy(original)
                target = changed
                for part in location:
                    target = target[part]
                target[key] = value
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    DoGDataset(caches['test'] / 'test')
            path.write_text(json.dumps(original))
            section = {'seed': 42, 'n_samples': 2, '_detector': DETECTOR, '_git_commit': commit}
            self.assertEqual(dataset_spec('test', section, {'fov_deg': 360})['boundary'], 'circular')
            self.assertEqual(dataset_spec('test', section, {'fov_deg': 359})['boundary'], 'valid')
