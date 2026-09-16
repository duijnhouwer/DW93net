from pathlib import Path
import tempfile
import unittest

import torch

from features.reichardt import ReichardtBank
from stimuli.load_flow_xt_data import MirroredDoGDataset, make_flow_loader
from stimuli.generate_flow_xt_data import generate_dog_dataset, StimConfig
from train.train_flow_xt_net import RecipeTrainingSettings, train_recipe_set


class ReichardtTests(unittest.TestCase):
    def test_mirrored_dog_reverses_space_dx_and_yaw(self):
        contrast = torch.arange(2 * 5, dtype=torch.float32).reshape(2, 5)
        target = torch.tensor([2.0, 3.0, 0.5])
        mirrored, mirrored_target = MirroredDoGDataset([(contrast, target)])[0]
        torch.testing.assert_close(mirrored, contrast.flip(-1))
        torch.testing.assert_close(mirrored_target, torch.tensor([-2.0, 3.0, -0.5]))
        torch.testing.assert_close(target, torch.tensor([2.0, 3.0, 0.5]))

    def test_mirrored_dog_reverses_online_correlation_displacements(self):
        torch.manual_seed(9)
        contrast = torch.randn(2, 31)
        mirrored, _ = MirroredDoGDataset([(contrast, torch.zeros(3))])[0]
        bank = ReichardtBank([-3, 0, 3])
        ordinary = bank(contrast)
        reflected = bank(mirrored)
        torch.testing.assert_close(reflected, ordinary.flip(0, 2))

    def test_default_bank_detects_fast_endpoints(self):
        bank = ReichardtBank()
        self.assertEqual(bank.dx_values, tuple(range(-60, 61)))
        for displacement in (-60, 60):
            xt = torch.zeros(2, 7200)
            xt[0, 3600 - displacement] = 1
            xt[1, 3600] = 1
            products = bank(xt)
            self.assertEqual(products.shape, (121, 1, 7080))
            self.assertEqual(products[bank.dx_values.index(displacement), 0, 3540], 1)
            self.assertEqual(products.sum(), 1)

    def test_background_loader_matches_serial_across_epochs(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(generate_dog_dataset(
                output_root=directory, n_samples=9,
                speed_range=(.001, .002), heading_deg_range=(0, 90),
                rot_deg_range=(0, 0), role='train', seed=7,
                dog_bundle_size=4,
                config=StimConfig(stim_wid_px=49, n_world_dots=10,
                                  x_supersampling=1)))
            from stimuli.load_flow_xt_data import DoGDataset
            dataset = DoGDataset(cache / 'train')
            serial = list(make_flow_loader(dataset, 4, 'cpu', num_workers=0))
            loader = make_flow_loader(dataset, 4, 'cpu')
            for _ in range(2):
                actual = list(loader)
                self.assertEqual(len(actual), 3)
                for (x, y), (expected_x, expected_y) in zip(actual, serial):
                    torch.testing.assert_close(x, expected_x, rtol=0, atol=0)
                    torch.testing.assert_close(y, expected_y, rtol=0, atol=0)
            del loader

    def test_scalar_reference_and_alignment(self):
        xt = torch.arange(2 * 2 * 11, dtype=torch.float32).reshape(2, 1, 2, 11) / 100
        bank = ReichardtBank([2, -3, 0])
        result = bank(xt)
        self.assertEqual(result.shape, (2, 3, 1, 6))
        for batch in range(2):
            for channel, dx in enumerate(bank.dx_values):
                for x in range(6):
                    self.assertEqual(
                        result[batch, channel, 0, x],
                        xt[batch, 0, 0, x + 2 - dx] * xt[batch, 0, 1, x + 2])
        torch.testing.assert_close(bank(xt[0]), result[0])
        torch.testing.assert_close(bank(xt[0, 0]), result[0])

    def test_direction_and_fractional_product(self):
        bank = ReichardtBank([-2, 2])
        for dx in (-2, 2):
            xt = torch.zeros(2, 10)
            xt[0, 4] = .5
            xt[1, 4 + dx] = .25
            responses = bank(xt)
            self.assertEqual(responses.sum().item(), .125)
            channel = bank.dx_values.index(dx)
            self.assertEqual(responses[channel].sum().item(), .125)

    def test_no_wraparound(self):
        xt = torch.zeros(2, 7)
        xt[0, -1] = xt[1, 0] = 1
        self.assertEqual(ReichardtBank([-1, 1])(xt).sum().item(), 0)

    def test_validation(self):
        for dx in ([], [1, 1], [1.5], [True]):
            with self.assertRaises(ValueError):
                ReichardtBank(dx)
        bank = ReichardtBank([-2, 2])
        invalid = [
            torch.zeros(1, 10), torch.zeros(3, 10), torch.zeros(2, 4),
            torch.ones(2, 5, 10), torch.full((2, 10), float('nan')),
            torch.ones(2, 10, dtype=torch.int64),
        ]
        for xt in invalid:
            with self.assertRaises(ValueError):
                bank(xt)

    def test_recipe_training_continues_optimizer_across_seeded_sets(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = Path(generate_dog_dataset(
                output_root=str(root), n_samples=2, speed_range=(1.0, 2.0),
                heading_deg_range=(0, 90), rot_deg_range=(0, 0), role='train',
                seed=4, dog_bundle_size=2,
                config=StimConfig(stim_wid_px=49, n_world_dots=10,
                                  x_supersampling=1)))
            checkpoints = root / 'checkpoints'
            common = dict(
                root_folder=str(cache), checkpoint_dir=str(checkpoints),
                git_commit='a' * 40, epochs_per_set=1, shuffle_seed=99,
                batch_size=2, num_workers=0, profile_batches=0,
                mirror_train_stimuli=False, dx_values=(-1, 1))
            train_recipe_set(RecipeTrainingSettings(
                training_set_index=0, training_seed=1000, **common))
            train_recipe_set(RecipeTrainingSettings(
                training_set_index=1, training_seed=1001, **common))
            latest = torch.load(checkpoints / 'latest.pt', map_location='cpu',
                                weights_only=True)
            self.assertEqual(latest['training_set_index'], 1)
            self.assertEqual(latest['training_seed'], 1001)
            self.assertEqual(latest['epoch_within_set'], 1)
            self.assertEqual(latest['global_epoch'], 2)
            self.assertEqual(latest['scheduler_state_dict']['last_epoch'], 2)
            candidates = sorted((checkpoints / 'candidates').glob('epoch-*.pt'))
            self.assertEqual([path.name for path in candidates],
                             ['epoch-000001.pt', 'epoch-000002.pt'])
            candidate = torch.load(candidates[-1], map_location='cpu',
                                   weights_only=True)
            self.assertNotIn('optimizer_state_dict', candidate)
            self.assertEqual(candidate['model_state_dict']['decoder.weight'].shape[0], 3)


if __name__ == '__main__':
    unittest.main()
