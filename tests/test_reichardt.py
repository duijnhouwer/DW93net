import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from features.reichardt import ReichardtBank
from features.dog_motion import DoGMotionBank
from stimuli.precompute_reichardt import precompute_dataset
from stimuli.load_flow_xt_data import (FlowXtDataset, ReichardtDataset, MirroredReichardtDataset,
                                       make_flow_loader)
from stimuli.stimulus_io import decode_stimulus
from models.pure_translation import PureTranslationNet
from train.train_flow_xt_net import Settings, train
from train.test_flow_xt_net import run_flow_xt_model_on_test_data


class ReichardtTests(unittest.TestCase):
    def test_sharded_half_cache_matches_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'raw'
            for split in ('train', 'validate'):
                (source / split).mkdir(parents=True)
                for i in range(3):
                    torch.save(torch.randint(0, 256, (1, 2, 49), dtype=torch.uint8),
                               source / split / f'ndx={i}_ndz=0_nrot=0_id={i}.pt')
            bank = DoGMotionBank([-1, 1])
            precompute_dataset(source, root / 'legacy', bank)
            metadata = precompute_dataset(source, root / 'sharded', bank, shard_size=2)
            self.assertTrue(metadata['finite_validated'])
            for split in ('train', 'validate'):
                legacy = ReichardtDataset(root / 'legacy' / split, keep_half=True)
                sharded = ReichardtDataset(root / 'sharded' / split, keep_half=True)
                self.assertEqual(legacy.files, sharded.files)
                self.assertEqual(len(list((root / 'sharded' / split).glob('*.npy'))), 2)
                for i in (2, 0, 1, 2):
                    x, y = sharded[i]
                    self.assertEqual(x.dtype, torch.float16)
                    torch.testing.assert_close(x, legacy[i][0], rtol=0, atol=0)
                    torch.testing.assert_close(y, legacy[i][1], rtol=0, atol=0)

    def test_nonfinite_cache_is_rejected_at_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'raw' / 'train').mkdir(parents=True)
            torch.save(torch.zeros((1, 2, 49), dtype=torch.uint8),
                       root / 'raw' / 'train' / 'ndx=0_ndz=0_nrot=0_id=0.pt')
            with patch.object(DoGMotionBank, '__call__', return_value=torch.full((4, 1, 7), 1e10)):
                with self.assertRaisesRegex(ValueError, 'Storage dtype overflow'):
                    precompute_dataset(root / 'raw', root / 'cache', DoGMotionBank([0]), shard_size=2)
            self.assertFalse((root / 'cache' / 'reichardt.json').exists())

    def test_mirrored_reichardt_reverses_space_displacement_dx_and_yaw(self):
        dx_values = (-1, 0, 1)
        responses = torch.arange(4 * 3 * 1 * 5, dtype=torch.float32).reshape(12, 1, 5)
        target = torch.tensor([2.0, 3.0, 0.5])
        mirrored, mirrored_target = MirroredReichardtDataset([(responses, target)], dx_values)[0]
        expected = responses.reshape(4, 3, 1, 5).flip(1, 3).reshape_as(responses)
        torch.testing.assert_close(mirrored, expected)
        torch.testing.assert_close(mirrored_target, torch.tensor([-2.0, 3.0, -0.5]))
        torch.testing.assert_close(target, torch.tensor([2.0, 3.0, 0.5]))
        restored, restored_target = MirroredReichardtDataset([(mirrored, mirrored_target)], dx_values)[0]
        torch.testing.assert_close(restored, responses)
        torch.testing.assert_close(restored_target, target)
    def test_default_bank_detects_fast_endpoints(self):
        bank = ReichardtBank()
        self.assertEqual(bank.dx_values, tuple(range(-60, 61)))
        for displacement in [-60, 60]:
            xt = torch.zeros(2, 7200)
            xt[0, 3600 - displacement] = 1
            xt[1, 3600] = 1
            products = bank(xt)
            self.assertEqual(products.shape, (121, 1, 7080))
            self.assertEqual(products[bank.dx_values.index(displacement), 0, 3540], 1)
            self.assertEqual(products.sum(), 1)

    def test_dataset_preserves_fractional_intensities(self):
        from stimuli.generate_flow_xt_data import generate_flow_xt_dataset, StimConfig
        intensities = torch.tensor([[0., .1, .4, .8, 1.]]).repeat(2, 1)
        for superimpose in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                with patch('stimuli.generate_flow_xt_data.generate_flow_xt_stim', return_value=intensities):
                    folder = generate_flow_xt_dataset(
                        output_root=directory, n_samples=1, split=(1, 0, 0), seed=42,
                        config=StimConfig(superimpose=superimpose))
                saved = torch.load(next((Path(folder) / 'train').glob('*.pt')), weights_only=True)
                levels = [0, 51, 204, 255, 255] if superimpose else [0, 26, 102, 204, 255]
                expected = torch.tensor(levels, dtype=torch.uint8).repeat(1, 2, 1)
                torch.testing.assert_close(saved, expected, rtol=0, atol=0)
                self.assertEqual(saved.dtype, torch.uint8)
                decoded, _ = FlowXtDataset(Path(folder) / 'train')[0]
                original = (intensities * (2 if superimpose else 1)).clamp(0, 1)[None]
                torch.testing.assert_close(decoded, original, rtol=0, atol=0.5/255 + 1e-7)

    def test_byte_grayscale_loading(self):
        levels = torch.arange(256, dtype=torch.uint8).repeat(1, 2, 1)
        expected = torch.arange(256, dtype=torch.float32).repeat(1, 2, 1) / 255
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = 'ndx=1000000000_ndz=-2000000000_nrot=3000000000_id=0.pt'
            torch.save(levels, root / name)
            decoded, label = FlowXtDataset(root)[0]
            self.assertEqual(decoded.dtype, torch.float32)
            torch.testing.assert_close(label, torch.tensor([1., -2., 3.]), rtol=0, atol=0)
            torch.testing.assert_close(decoded, expected, rtol=0, atol=0)
            for invalid in (expected, expected.half()):
                torch.save(invalid, root / name)
                with self.assertRaises(ValueError):
                    FlowXtDataset(root)[0]
            torch.testing.assert_close(decode_stimulus(levels)[..., 1], torch.full((1, 2), 1/255))

    def test_byte_preprocessing_scales_before_products(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'raw'
            (source / 'train').mkdir(parents=True)
            generation = {"format": "polar-translation-stimuli-v1", "seed": 7}
            (source / 'generation.json').write_text(json.dumps(generation))
            filename = 'ndx=0_ndz=0_nrot=0_id=0.pt'
            levels = torch.tensor([[[0, 1, 128, 255], [255, 1, 64, 128]]], dtype=torch.uint8).repeat(1, 1, 10)
            torch.save(levels, source / 'train' / filename)
            bank = DoGMotionBank([0])
            # Scalar DoG/product reference is tested separately in test_dog_motion.
            reference = bank(levels.float() / 255)
            cache = root / 'cache'
            metadata = precompute_dataset(source, cache, bank)
            saved = torch.load(cache / 'train' / filename, weights_only=True)
            self.assertEqual(saved.dtype, torch.float16)
            self.assertEqual(metadata['storage_dtype'], 'torch.float16')
            self.assertEqual(metadata['stimulus_generation'], generation)
            torch.testing.assert_close(saved, reference.half())
            self.assertGreater(saved.max().item(), 0)
            decoded, _ = ReichardtDataset(cache / 'train')[0]
            torch.testing.assert_close(decoded, saved.float(), rtol=0, atol=0)
            for invalid in (saved.float(), saved.to(torch.uint8)):
                torch.save(invalid, cache / 'train' / filename)
                with self.assertRaises(ValueError):
                    ReichardtDataset(cache / 'train')[0]

    def test_background_loader_matches_serial_across_epochs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'train').mkdir()
            (root / 'reichardt.json').write_text(json.dumps(DoGMotionBank([0]).metadata))
            for i in range(9):
                torch.save(torch.full((4, 1, 7), i/10, dtype=torch.float16),
                           root / 'train' / f'ndx={i}_ndz=0_nrot=0_id={i}.pt')
            dataset = ReichardtDataset(root / 'train')
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
        for b in range(2):
            for c, dx in enumerate(bank.dx_values):
                for t in range(1):
                    for x in range(6):
                        self.assertEqual(result[b, c, t, x],
                                         xt[b, 0, 0, x+2-dx] * xt[b, 0, 1, x+2])
        torch.testing.assert_close(bank(xt[0]), result[0])
        torch.testing.assert_close(bank(xt[0, 0]), result[0])

    def test_direction_and_fractional_product(self):
        bank = ReichardtBank([-2, 2])
        for dx in [-2, 2]:
            xt = torch.zeros(2, 10)
            xt[0, 4] = .5
            xt[1, 4+dx] = .25
            responses = bank(xt)
            self.assertEqual(responses.sum().item(), .125)
            c = bank.dx_values.index(dx)
            self.assertEqual(responses[c].sum().item(), .125)
            xt[0, 4] = 0
            self.assertEqual(bank(xt).sum().item(), 0)
            xt[0, 4] = .5
            xt[1, 4+dx] = 0
            self.assertEqual(bank(xt).sum().item(), 0)

    def test_no_wraparound(self):
        xt = torch.zeros(2, 7)
        xt[0, -1] = xt[1, 0] = 1
        self.assertEqual(ReichardtBank([-1, 1])(xt).sum().item(), 0)

    def test_validation(self):
        for dx in ([], [1, 1], [1.5], [True]):
            with self.assertRaises(ValueError):
                ReichardtBank(dx)
        bank = ReichardtBank([-2, 2])
        for xt in [torch.zeros(1, 10), torch.zeros(3, 10), torch.zeros(2, 4), torch.ones(2, 5, 10),
                   torch.full((2, 10), float('nan')), torch.ones(2, 10, dtype=torch.int64)]:
            with self.assertRaises(ValueError):
                bank(xt)

    def test_cache_training_resume_and_evaluation(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'raw'
            destination = (root /
                           'nspeed=1000000000,1000000000_nheading=0,90000000000_nrot=0,0' / 'reichardt')
            for split in ['train', 'validate', 'test']:
                (source / split).mkdir(parents=True)
                for i in range(2):
                    torch.save(torch.randint(0, 256, (1, 2, 49), dtype=torch.uint8), source / split / f'ndx={i*1000000000}_ndz=0_nrot=0_id={i}.pt')
            bank = DoGMotionBank([-1, 1])
            precompute_dataset(source, destination, bank)
            dataset = ReichardtDataset(destination / 'train')
            maps, label = dataset[0]
            raw = torch.load(source / 'train' / dataset.files[0], weights_only=True)
            torch.testing.assert_close(maps, bank(decode_stimulus(raw)).half().float())
            self.assertEqual(maps.dtype, torch.float32)
            self.assertEqual(label.tolist(), [0, 0, 0])
            with self.assertRaises(ValueError):
                ReichardtDataset(destination / 'train', DoGMotionBank([1, -1]).metadata)
            with self.assertRaises(FileExistsError):
                precompute_dataset(source, destination, bank)
            model = PureTranslationNet(dataset.metadata["dx_values"])
            self.assertFalse(hasattr(model, 'pre'))
            output = model(torch.stack([maps, maps]))
            output.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            cfg = Settings(root_folder=str(destination), checkpoints_dir=str(root / 'checkpoints'),
                           num_epochs=1, batch_size=2)
            train(cfg)
            self.assertFalse((root / 'checkpoints' / destination.parent.name / 'epoch_000001.pt').exists())
            cfg.num_epochs = 2
            train(cfg)
            checkpoint = root / 'checkpoints' / destination.parent.name / '_latest.pt'
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            self.assertEqual(saved['scheduler_state_dict']['last_epoch'], 2)
            self.assertEqual(saved['scheduler_state_dict']['_last_lr'],
                             [saved['optimizer_state_dict']['param_groups'][0]['lr']])
            result = run_flow_xt_model_on_test_data(str(destination), str(checkpoint), 2)
            self.assertEqual(result['preds'].shape, (2, 2))
            self.assertTrue(torch.isfinite(torch.from_numpy(result['preds'])).all())
            metadata_path = destination / 'reichardt.json'
            metadata = json.loads(metadata_path.read_text())
            metadata['dx_values'].reverse()
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                train(cfg)


if __name__ == '__main__':
    unittest.main()
