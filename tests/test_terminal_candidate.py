from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from stimuli.generate_flow_xt_data import generate_dog_dataset, StimConfig
from train.train_flow_xt_net import RecipeTrainingSettings, train_recipe_set


class TerminalCandidateTests(unittest.TestCase):
    def test_terminal_candidate_preserves_global_cadence_and_resume(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        with tempfile.TemporaryDirectory() as directory, patch.object(
                torch.cuda, 'is_available', return_value=False):
            root = Path(directory)
            cache = generate_dog_dataset(
                output_root=str(root / 'data'), n_samples=2, seed=4,
                speed_range=(.001, .002), heading_deg_range=(0, 90),
                rot_deg_range=(0, 0), role='train', dog_bundle_size=2,
                config=StimConfig(stim_wid_px=49, n_world_dots=10, x_supersampling=1))
            # Four total epochs: shorter than interval 10, non-divisible by 3,
            # and exactly divisible by 2.
            for interval, expected in ((10, [4]), (3, [3, 4]), (2, [2, 4])):
                with self.subTest(interval=interval):
                    checkpoints = root / str(interval)
                    cfg = RecipeTrainingSettings(
                        root_folder=cache, checkpoint_dir=str(checkpoints),
                        git_commit='a' * 40, training_set_index=0, training_seed=4,
                        epochs_per_set=2, shuffle_seed=99, batch_size=2, num_workers=0,
                        allow_amp=False, mirror_train_stimuli=False,
                        dx_values=(-1, 0, 1), candidate_interval_epochs=interval,
                        is_final_training_set=False)
                    train_recipe_set(cfg)
                    candidates = checkpoints / 'candidates'
                    self.assertEqual(len(list(candidates.glob('*.pt'))), int(interval == 2))
                    final_cfg = replace(cfg, training_set_index=1, training_seed=5,
                                        is_final_training_set=True)
                    latest = train_recipe_set(final_cfg)
                    self.assertEqual(sorted(p.name for p in candidates.glob('*.pt')),
                                     [f'epoch-{epoch:06d}.pt' for epoch in expected])
                    terminal = candidates / 'epoch-000004.pt'
                    candidate = torch.load(terminal, weights_only=True)
                    checkpoint = torch.load(latest, weights_only=True)
                    self.assertEqual(candidate['global_epoch'], 4)
                    for key, tensor in candidate['model_state_dict'].items():
                        self.assertTrue(torch.equal(tensor, checkpoint['model_state_dict'][key]))
                    saved_time = terminal.stat().st_mtime_ns
                    train_recipe_set(final_cfg)
                    self.assertEqual(terminal.stat().st_mtime_ns, saved_time)
