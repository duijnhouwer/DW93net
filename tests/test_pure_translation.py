"""Contract checks for the translation-only baseline, not accuracy claims."""
import unittest
from unittest.mock import patch

import torch

from stimuli.metadata import require_pure_translation_root
import json
import tempfile
from pathlib import Path
from stimuli.generate_flow_xt_data import generate_dog_dataset, StimConfig
from models.pure_translation import PureTranslationNet, LocalMotionOpponency, V1Motion


class PureTranslationTests(unittest.TestCase):
    def test_opponency_uses_span_labels_and_preserves_stationary(self):
        opponent = LocalMotionOpponency([2, 0, -2])
        x = torch.tensor([[[3., 1.], [4., 5.], [1., 3.]]])
        torch.testing.assert_close(opponent(x), torch.tensor([[[2., 0.], [4., 5.], [0., 2.]]]))
        with self.assertRaisesRegex(ValueError, 'paired'):
            LocalMotionOpponency([-2, 1])

    def test_local_opponency_precedes_spatial_integration(self):
        opponent = LocalMotionOpponency([-1, 1])
        balanced = torch.ones(1, 2, 100)
        self.assertEqual(opponent(balanced).sum().item(), 0.)
        separated = torch.zeros_like(balanced)
        separated[:, 0, :50] = 1
        separated[:, 1, 50:] = 1
        self.assertTrue((opponent(separated).mean(dim=-1) > 0).all())

    def test_surround_suppresses_local_response(self):
        v1 = V1Motion()
        isolated = torch.zeros(1, 1, 121)
        isolated[:, :, 53:68] = 1
        surrounded = torch.ones_like(isolated)
        center_a, contrast_a = v1(isolated)
        center_b, contrast_b = v1(surrounded)
        torch.testing.assert_close(center_a[:, :, 60], center_b[:, :, 60])
        self.assertGreater(contrast_a[0, 0, 60].item(), contrast_b[0, 0, 60].item())

    def test_yaw_constraints_come_from_metadata_not_folder_name(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(generate_dog_dataset(output_root=directory, n_samples=1,
                role='train', seed=1, rot_deg_range=(0, 0),
                config=StimConfig(stim_wid_px=49, n_world_dots=4, x_supersampling=1)))
            renamed = cache.with_name('arbitrary_name_nrot=1,1')
            cache.rename(renamed)
            require_pure_translation_root(renamed)
            path = renamed / 'dog.json'
            metadata = json.loads(path.read_text())
            metadata['stimulus_generation']['rot_deg_range']['value'] = 1
            path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'requires rot_deg=0'):
                require_pure_translation_root(renamed)

    def test_full_resolution_backward_and_roundtrip(self):
        torch.manual_seed(10)
        model = PureTranslationNet().eval()
        x = torch.rand(1, 121, 1, 7062)
        prediction = model(x)
        self.assertEqual(prediction.shape, (1, 3))
        prediction.square().sum().backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        restored = PureTranslationNet().eval()
        restored.load_state_dict(model.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(restored(x), prediction)

    def test_reject_wrong_input_axes(self):
        model = PureTranslationNet([-1, 0, 1])
        for shape in [(1, 3, 2, 10), (1, 2, 1, 10), (1, 3, 10), (1, 3, 1, 0)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.zeros(shape))


if __name__ == '__main__':
    unittest.main()
