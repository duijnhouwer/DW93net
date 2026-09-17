"""Integration checks for scientific identity, provenance, and current-only schemas."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from checkpoints import validate_checkpoint, load_checkpoint
from evaluation import inference_batches, metrics, summary_metrics
from stimuli.dataset_identity import dataset_spec
from stimuli.generate_flow_xt_data import generate_dog_dataset, StimConfig
from stimuli.load_flow_xt_data import DoGDataset
from stimuli.metadata import motion_scales
from stimuli.preprocessing_provenance import capture_provenance
from train.train_flow_xt_net import train_recipe_set, RecipeTrainingSettings
from test_training_recipe import module, DETECTOR


class CompatibilityTests(unittest.TestCase):
    def test_documentation_is_not_a_compatibility_dependency(self):
        import checkpoints
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in checkpoints.TRAINING_SOURCES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('implementation')
            with patch.object(checkpoints, 'ROOT', root):
                original = checkpoints.compatibility()
                (root / 'README.md').write_text('new documentation')
                self.assertEqual(original, checkpoints.compatibility())
                (root / 'src/models/pure_translation.py').write_text('changed model')
                self.assertNotEqual(original, checkpoints.compatibility())

    def test_online_bank_changes_do_not_invalidate_dog_cache(self):
        section = dict(seed=1, n_samples=2, _detector=DETECTOR)
        original = dataset_spec('train', section, {})
        changed = section | {'_detector': DETECTOR | {'displacement_max_px': 100}}
        self.assertEqual(original, dataset_spec('train', changed, {}))
        changed['_detector']['dog_sigma_center_px'] = 2.
        self.assertNotEqual(original, dataset_spec('train', changed, {}))

    def test_snapshot_is_captured_once_and_referenced(self):
        from stimuli import preprocessing_provenance as provenance
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(provenance, 'write_provenance', wraps=provenance.write_provenance) as capture:
                source = capture_provenance(root / 'run', module.ROOT)
                self.assertEqual(source, capture_provenance(root / 'run', module.ROOT))
                for index in range(2):
                    cache = Path(generate_dog_dataset(output_root=str(root / str(index)),
                        provenance=source, n_samples=1, seed=index, role='train',
                        rot_deg_range=(0, 0), config=StimConfig(
                            stim_wid_px=49, n_world_dots=4, x_supersampling=1)))
                    self.assertEqual(json.loads((cache / 'dog.json').read_text())['software'], source)
                self.assertEqual(capture.call_count, 1)
            self.assertEqual(len(list(root.rglob('source.tar.gz'))), 1)

    def test_renaming_commit_changes_shared_evaluation_and_strict_schema(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        with tempfile.TemporaryDirectory() as directory, patch.object(torch.cuda, 'is_available', return_value=False):
            root = Path(directory)
            cache = Path(generate_dog_dataset(output_root=str(root / 'data'),
                n_samples=3, seed=4, role='train', rot_deg_range=(0, 0),
                speed_range=(.001, .002), config=StimConfig(
                    stim_wid_px=49, n_world_dots=4, x_supersampling=1)))
            scales = motion_scales(cache)
            renamed = cache.with_name('anything_at_all')
            cache.rename(renamed)
            self.assertEqual(motion_scales(renamed), scales)
            cfg = RecipeTrainingSettings(root_folder=str(renamed), checkpoint_dir=str(root / 'checkpoints'),
                git_commit='a' * 40, training_set_index=0, training_seed=4, epochs_per_set=1,
                shuffle_seed=9, num_workers=0, batch_size=2, mixed_precision=False,
                mirror_train_stimuli=False, dx_values=(-1, 0, 1))
            latest = train_recipe_set(cfg)
            checkpoint = load_checkpoint(latest, resume=True)
            # Git is provenance, not the compatibility decision.
            train_recipe_set(replace(cfg, git_commit='b' * 40))
            result = module.evaluate_model(latest, renamed, 'train',
                                           dict(batch_size=2, num_workers=0), 'b' * 40)
            self.assertEqual(result['checkpoint_git_commit'], 'a' * 40)
            self.assertEqual(result['evaluator_git_commit'], 'b' * 40)
            dataset = DoGDataset(renamed / 'train')
            batches = list(inference_batches(checkpoint, dataset, device='cpu', batch_size=2))
            summary = summary_metrics(iter(batches), scales)
            prediction = np.concatenate([p for p, _ in batches])
            target = np.concatenate([t for _, t in batches])
            detailed = metrics(prediction[:, :2], target[:, :2])
            self.assertAlmostEqual(summary['heading_mae_deg'], detailed['heading_mae_deg'])
            self.assertAlmostEqual(summary['translation_speed_mae'], detailed['speed_mae'])
            self.assertAlmostEqual(summary['translation_vector_tip_mae'], detailed['vector_tip_mae'])
            for field in ('scaler_state_dict', 'compatibility', 'schema', 'data_contract'):
                invalid = copy.deepcopy(checkpoint)
                del invalid[field]
                with self.subTest(field=field), self.assertRaises(ValueError):
                    validate_checkpoint(invalid, resume=True)
            for field in ('model_schema', 'data_schema', 'inference_code'):
                invalid = copy.deepcopy(checkpoint)
                invalid['compatibility'][field] = 'changed'
                with self.subTest(field=field), self.assertRaises(ValueError):
                    validate_checkpoint(invalid)
            invalid = copy.deepcopy(checkpoint)
            invalid['compatibility']['training_code'] = {}
            validate_checkpoint(invalid)  # Inference is independent of optimizer code.
            with self.assertRaises(ValueError):
                validate_checkpoint(invalid, resume=True)
            invalid = copy.deepcopy(checkpoint)
            invalid['data_contract']['geometry']['fov_deg'] = 99
            with self.assertRaisesRegex(ValueError, 'geometry'):
                list(inference_batches(invalid, dataset, device='cpu', batch_size=2))
