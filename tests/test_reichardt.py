import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from features.reichardt import ReichardtBank
from data.precompute_reichardt import precompute_dataset
from data.load_flow_xt_data import ReichardtDataset
from models.flow_xt_net import FlowXTNet
from train.train_flow_xt_net import Settings, train
from train.test_flow_xt_net import run_flow_xt_model_on_test_data


class ReichardtTests(unittest.TestCase):
    def test_dataset_preserves_fractional_intensities(self):
        from data.generate_flow_xt_data import generate_flow_xt_dataset, StimConfig
        intensities = torch.tensor([[0., .1, .4, .8, 1.]])
        for superimpose in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                with patch('data.generate_flow_xt_data.generate_flow_xt_stim', return_value=intensities):
                    folder = generate_flow_xt_dataset(
                        output_root=directory, n_samples=1, split=(1, 0, 0), seed=42,
                        config=StimConfig(superimpose=superimpose))
                saved = torch.load(next((Path(folder) / 'train').glob('*.pt')), weights_only=True)
                expected = (intensities * (2 if superimpose else 1)).clamp(0, 1)[None]
                torch.testing.assert_close(saved, expected)
                self.assertEqual(saved.dtype, torch.float32)

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
            destination = root / 'cache' / 'nux=0,1000000000_nuz=0,0_nrot=0,0'
            for split in ['train', 'validate', 'test']:
                (source / split).mkdir(parents=True)
                for i in range(2):
                    torch.save(torch.rand(1, 2, 9), source / split / f'flow_nux={i*1000000000}_nuz=0_nrot=0_id={i}.pt')
            bank = ReichardtBank([-1, 1])
            precompute_dataset(source, destination, bank, torch.float16)
            dataset = ReichardtDataset(destination / 'train')
            maps, label = dataset[0]
            raw = torch.load(source / 'train' / dataset.files[0], weights_only=True)
            torch.testing.assert_close(maps, bank(raw).half().float())
            self.assertEqual(maps.dtype, torch.float32)
            self.assertEqual(label.tolist(), [0, 0, 0])
            with self.assertRaises(ValueError):
                ReichardtDataset(destination / 'train', ReichardtBank([1, -1]).metadata)
            with self.assertRaises(FileExistsError):
                precompute_dataset(source, destination, bank)
            model = FlowXTNet(dataset.n_channels)
            self.assertFalse(hasattr(model, 'pre'))
            output = model(torch.stack([maps, maps]))
            output.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            cfg = Settings(root_folder=str(destination), checkpoints_dir=str(root / 'checkpoints'),
                           num_epochs=1, batch_size=2)
            train(cfg)
            cfg.num_epochs = 2
            train(cfg)
            checkpoint = root / 'checkpoints' / destination.name / '_latest.pt'
            result = run_flow_xt_model_on_test_data(str(destination), str(checkpoint), 2)
            self.assertEqual(result['preds'].shape, (2, 3))
            self.assertTrue(torch.isfinite(torch.from_numpy(result['preds'])).all())
            metadata_path = destination / 'reichardt.json'
            metadata = json.loads(metadata_path.read_text())
            metadata['dx_values'].reverse()
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                train(cfg)


if __name__ == '__main__':
    unittest.main()
