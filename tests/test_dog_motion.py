import hashlib
import json
import math
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import torch

from features.dog_motion import DoGMotionBank
from features.reichardt import ReichardtBank
from models.pure_translation import PureTranslationNet
from stimuli.preprocessing_provenance import write_provenance, repository_state
from stimuli.precompute_reichardt import precompute_dataset


class DoGMotionTests(unittest.TestCase):
    def test_independent_scalar_reference_and_channel_alignment(self):
        bank = DoGMotionBank([2, -3, 0])
        torch.manual_seed(5)
        xt = torch.rand(2, 1, 2, 49)
        kernels = []
        for sigma in [1., 3.]:
            weights = [math.exp(-.5 * (i / sigma)**2) for i in range(-9, 10)]
            kernels.append([w / sum(weights) for w in weights])
        reference = torch.zeros(2, 12, 1, 26)
        for b in range(2):
            contrast = [[sum((kernels[0][k] - kernels[1][k]) * float(xt[b, 0, t, x+k])
                             for k in range(19)) for x in range(31)] for t in range(2)]
            for polarity, (sign0, sign1) in enumerate([(1, 1), (-1, -1), (1, -1), (-1, 1)]):
                for c, dx in enumerate(bank.dx_values):
                    for x in range(26):
                        reference[b, polarity*3+c, 0, x] = (max(0, sign0*contrast[0][x+2-dx])
                                                            * max(0, sign1*contrast[1][x+2]))
        torch.testing.assert_close(bank(xt), reference, atol=1e-7, rtol=1e-4)
        torch.testing.assert_close(bank(xt[0, 0]), reference[0], atol=1e-7, rtol=1e-4)

    def test_uniform_and_contrast_inversion(self):
        bank = DoGMotionBank([-1, 0, 1])
        self.assertLess(bank(torch.ones(2, 61)).abs().max().item(), 1e-12)
        x = torch.rand(2, 61)
        original, inverted = bank(x), bank(1-x)
        torch.testing.assert_close(original[:3], inverted[3:6], atol=1e-7, rtol=1e-4)
        torch.testing.assert_close(original[3:6], inverted[:3], atol=1e-7, rtol=1e-4)
        torch.testing.assert_close(original[6:9], inverted[9:12], atol=1e-7, rtol=1e-4)
        torch.testing.assert_close(original[9:12], inverted[6:9], atol=1e-7, rtol=1e-4)

    def test_full_width_fast_displacement_peaks(self):
        bank = DoGMotionBank()
        for dx in [-60, 60]:
            x = torch.zeros(2, 7200)
            x[0, 3600-dx] = x[1, 3600] = 1
            products = bank(x)
            self.assertEqual(products.shape, (484, 1, 7062))
            for block in products.chunk(4)[:2]:
                self.assertEqual(block.sum(dim=(1, 2)).argmax().item(), dx+60)
            self.assertTrue((products >= 0).all())

    def test_cross_subtraction_matches_signed_products_and_reverses_opponency(self):
        bank = DoGMotionBank([-4, 4])
        model = PureTranslationNet(bank.dx_values)
        x = torch.zeros(2, 101)
        x[0, 45] = x[1, 49] = 1
        products = bank(x).unsqueeze(0).squeeze(2)
        signed = model.signed_correlation(products)
        direct = ReichardtBank(bank.dx_values)(bank.filter_frames(x)).squeeze(2)
        torch.testing.assert_close(signed, direct, atol=1e-7, rtol=1e-4)
        # Invert only the second frame: same-polarity and cross-polarity evidence
        # exchange, so the signed response changes sign before rectification.
        reversed_x = x.clone()
        reversed_x[1] = 1 - reversed_x[1]
        reversed_products = bank(reversed_x).unsqueeze(0).squeeze(2)
        reversed_signed = model.signed_correlation(reversed_products)
        torch.testing.assert_close(reversed_signed, -signed, atol=1e-7, rtol=1e-4)
        for ordinary, reversed_response in zip(model.v1(signed), model.v1(reversed_signed)):
            rates = model.mt_opponency(ordinary)
            reversed_rates = model.mt_opponency(reversed_response)
            torch.testing.assert_close(reversed_rates, rates.flip(1), atol=1e-6, rtol=1e-4)
            self.assertGreater(rates[:, 1].sum().item(), rates[:, 0].sum().item())

    def test_metadata_and_width_validation(self):
        bank = DoGMotionBank()
        self.assertEqual(DoGMotionBank.from_metadata(bank.metadata).metadata, bank.metadata)
        changed = bank.metadata
        changed['polarity_order'].reverse()
        with self.assertRaises(ValueError):
            DoGMotionBank.from_metadata(changed)
        with self.assertRaises(ValueError):
            bank(torch.zeros(2, 138))
        for center, surround in [(0, 3), (3, 1), (1, float('inf'))]:
            with self.assertRaises(ValueError):
                DoGMotionBank(sigma_center=center, sigma_surround=surround)


class ProvenanceTests(unittest.TestCase):
    def test_recovers_dirty_worktree_without_git_history(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / 'repo'
            repo.mkdir()
            def git(*args):
                return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.DEVNULL)
            git('init')
            (repo / 'code.py').write_text('original\n')
            (repo / 'deleted.py').write_text('delete me\n')
            (repo / 'binary.bin').write_bytes(b'\x00original')
            git('add', '.')
            git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-m', 'base')
            (repo / 'code.py').write_text('staged\n')
            git('add', 'code.py')
            (repo / 'code.py').write_text('final unstaged\n')
            (repo / 'deleted.py').unlink()
            (repo / 'binary.bin').write_bytes(b'\x00changed')
            (repo / 'new.py').write_text('untracked source\n')
            output = repo / 'cache'
            output.mkdir()
            info, state = write_provenance(output, repo)
            self.assertEqual(info['git_commit'], git('rev-parse', 'HEAD').decode().strip())
            self.assertTrue(info['tracked_changes'])
            self.assertEqual(state, repository_state(repo, output))
            provenance = output / 'provenance'
            self.assertIn(b'GIT binary patch', (provenance / 'working-tree.patch').read_bytes())
            restored = Path(directory) / 'restored'
            with tarfile.open(provenance / 'source.tar.gz') as archive:
                archive.extractall(restored, filter='data')
            self.assertEqual((restored / 'new.py').read_text(), 'untracked source\n')
            self.assertEqual((restored / 'code.py').read_text(), 'final unstaged\n')
            self.assertFalse((restored / 'deleted.py').exists())
            for name, entry in json.loads((provenance / 'source-manifest.json').read_text()).items():
                self.assertEqual(hashlib.sha256((restored / name).read_bytes()).hexdigest(), entry['sha256'])
            (repo / 'new.py').write_text('changed during run\n')
            self.assertNotEqual(state, repository_state(repo, output))

    def test_source_changes_leave_cache_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'raw'
            source.mkdir()
            torch.save(torch.zeros(2, 150, dtype=torch.uint8), source / 'sample.pt')
            output = Path(directory) / 'cache'
            with patch('stimuli.precompute_reichardt.repository_state', return_value=None):
                with self.assertRaisesRegex(RuntimeError, 'Source changed'):
                    precompute_dataset(source, output)
            self.assertFalse((output / 'reichardt.json').exists())


if __name__ == '__main__':
    unittest.main()
