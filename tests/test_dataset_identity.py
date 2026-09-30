import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mission_plan_helpers import module, DETECTOR
from stimuli.dataset_identity import storage_manifest


class DatasetIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'managed'
        self.cache = self.path / 'dog'
        self.folder = self.cache / 'test'
        self.section = dict(seed=4, n_samples=2, _detector=DETECTOR)
        self.stimuli = dict(speed_range=[.001, .002], heading_deg_range=[0, 90],
                            rot_deg_range=[0, 0], dog_bundle_size=2)
        self.generator = patch.object(module, 'generate_dog_dataset', side_effect=self.generate).start()
        self.addCleanup(patch.stopall)
        self.prepare()

    def generate(self, **kwargs):
        self.folder.mkdir(parents=True)
        (self.cache / 'dog.json').write_text(json.dumps(dict(
            storage_format='dog-contrast-bundles-v3', storage_dtype='torch.float32',
            finite_validated=True, n_samples=2, contrast_shape=[2, 31])))
        self.index = [[0, 'bundle.npy', 0], [1, 'bundle.npy', 1]]
        self.write_index()
        np.save(self.folder / 'bundle.npy', np.zeros(2, dtype=[
            ('dog_contrast', np.float32, (2, 31)), ('target', np.float32, (3,))]))
        return self.cache

    def write_index(self):
        (self.folder / 'bundles.json').write_text(json.dumps(self.index))

    def prepare(self, persistent=True):
        return module._prepare_dataset(self.path, 'test', self.section,
                                       self.stimuli, persistent=persistent)

    def test_identity_and_reuse(self):
        self.assertEqual(self.prepare(), self.cache)
        self.assertEqual(self.generator.call_count, 1)
        with patch('stimuli.dataset_identity.revision', return_value='b' * 40):
            with self.assertRaisesRegex(ValueError, 'Persistent'):
                self.prepare()
            self.assertTrue((self.folder / 'bundle.npy').exists())
            self.prepare(persistent=False)
        self.assertEqual(self.generator.call_count, 2)

    def test_obsolete_cache_rejected_without_removing_artifacts(self):
        metadata_path = self.cache / 'dog.json'
        metadata = json.loads(metadata_path.read_text())
        metadata['storage_format'] = 'dog-contrast-bundles-v2'
        metadata_path.write_text(json.dumps(metadata))
        for persistent in (True, False):
            with self.assertRaisesRegex(ValueError, 'regenerate'):
                self.prepare(persistent=persistent)
            self.assertTrue((self.folder / 'bundle.npy').exists())
        self.assertEqual(self.generator.call_count, 1)

    def test_missing_bundle_rejected_and_managed_cache_regenerated(self):
        (self.folder / 'bundle.npy').unlink()
        with self.assertRaisesRegex(ValueError, 'Persistent'):
            self.prepare()
        self.prepare(persistent=False)
        self.assertEqual(self.generator.call_count, 2)

    def test_truncated_bundle_rejected(self):
        with (self.folder / 'bundle.npy').open('r+b') as stream:
            stream.truncate(150)
        with self.assertRaisesRegex(ValueError, 'Persistent'):
            self.prepare()

    def test_duplicate_out_of_range_and_missing_index_rows(self):
        for index in ([[0, 'bundle.npy', 0], [1, 'bundle.npy', 0]],
                      [[0, 'bundle.npy', 0], [1, 'bundle.npy', 2]],
                      [[0, 'bundle.npy', 0]]):
            with self.subTest(index=index):
                self.index = index
                self.write_index()
                with self.assertRaises(ValueError):
                    storage_manifest(self.cache, 'test', 2)

    def test_wrong_dtype_and_extra_bundle_rejected(self):
        np.save(self.folder / 'extra.npy', np.zeros(2))
        with self.assertRaises(ValueError):
            storage_manifest(self.cache, 'test', 2)
        (self.folder / 'extra.npy').unlink()
        np.save(self.folder / 'bundle.npy', np.zeros(2))
        with self.assertRaises(ValueError):
            storage_manifest(self.cache, 'test', 2)

    def test_metadata_changes_and_legacy_state_not_reused(self):
        state_path = self.path / module.DATASET_STATE
        state = json.loads(state_path.read_text())
        del state['storage']
        state_path.write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, 'Persistent'):
            self.prepare()
        self.prepare(persistent=False)
        metadata_path = self.cache / 'dog.json'
        metadata = json.loads(metadata_path.read_text())
        metadata['changed'] = True
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'Persistent'):
            self.prepare()
