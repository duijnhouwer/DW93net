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

from features.dog_motion import DoGFilter
from stimuli.generate_flow_xt_data import StimConfig, generate_dog_dataset
from stimuli.preprocessing_provenance import write_provenance, repository_state


class DoGFilterTests(unittest.TestCase):
    def test_independent_scalar_reference(self):
        dog = DoGFilter()
        torch.manual_seed(5)
        xt = torch.rand(2, 1, 2, 49)
        kernels = []
        for sigma in (1., 3.):
            weights = [math.exp(-.5 * (i / sigma) ** 2) for i in range(-9, 10)]
            kernels.append([weight / sum(weights) for weight in weights])
        reference = torch.empty(2, 1, 2, 31)
        for batch in range(2):
            for frame in range(2):
                for x in range(31):
                    reference[batch, 0, frame, x] = sum(
                        (kernels[0][k] - kernels[1][k])
                        * float(xt[batch, 0, frame, x + k])
                        for k in range(19))
        torch.testing.assert_close(dog(xt), reference, atol=1e-7, rtol=1e-4)
        torch.testing.assert_close(dog(xt[0, 0])[0], reference[0], atol=1e-7, rtol=1e-4)

    def test_uniform_and_contrast_inversion(self):
        dog = DoGFilter()
        self.assertLess(dog(torch.ones(2, 61)).abs().max().item(), 1e-6)
        x = torch.rand(2, 61)
        torch.testing.assert_close(dog(1 - x), -dog(x), atol=1e-6, rtol=1e-4)

    def test_metadata_and_width_validation(self):
        dog = DoGFilter()
        self.assertEqual(DoGFilter.from_metadata(dog.metadata).metadata, dog.metadata)
        changed = dog.metadata
        changed['signed_contrast'] = 'wrong'
        with self.assertRaises(ValueError):
            DoGFilter.from_metadata(changed)
        with self.assertRaises(ValueError):
            dog(torch.zeros(2, 18))
        for center, surround in ((0, 3), (3, 1), (1, float('inf'))):
            with self.assertRaises(ValueError):
                DoGFilter(sigma_center=center, sigma_surround=surround)


class ProvenanceTests(unittest.TestCase):
    def test_recovers_dirty_worktree_without_git_history(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / 'repo'
            repo.mkdir()

            def git(*args):
                return subprocess.check_output(
                    ['git', '-C', str(repo), *args], stderr=subprocess.DEVNULL)

            git('init')
            (repo / 'code.py').write_text('original\n')
            (repo / 'deleted.py').write_text('delete me\n')
            (repo / 'binary.bin').write_bytes(b'\x00original')
            git('add', '.')
            git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                'commit', '-m', 'base')
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
            for name, entry in json.loads(
                    (provenance / 'source-manifest.json').read_text()).items():
                self.assertEqual(
                    hashlib.sha256((restored / name).read_bytes()).hexdigest(),
                    entry['sha256'])
            (repo / 'new.py').write_text('changed during run\n')
            self.assertNotEqual(state, repository_state(repo, output))

    def test_source_changes_leave_cache_incomplete(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                'stimuli.generate_flow_xt_data.repository_state', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'Source changed'):
                generate_dog_dataset(
                    output_root=directory, n_samples=1,
                    speed_range=(.001, .001), heading_deg_range=(0, 0),
                    rot_deg_range=(0, 0), role='train', seed=1,
                    config=StimConfig(stim_wid_px=49, n_world_dots=4,
                                      x_supersampling=1))
            self.assertFalse(list(Path(directory).rglob('dog.json')))


if __name__ == '__main__':
    unittest.main()
