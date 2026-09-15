import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('evaluate_progress', Path(__file__).resolve().parents[1] / 'scripts/evaluate_progress.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class EvaluationMetricsTests(unittest.TestCase):
    def test_identical_and_speed_only_error(self):
        m = module.metrics([[0, 2], [2, 0]], [[0, 1], [1, 0]])
        self.assertEqual(m['heading_mae_deg'], 0)
        self.assertEqual(m['heading_relative_mae_pct'], 0)
        self.assertEqual(m['speed_mae'], 1)
        self.assertEqual(m['speed_mae_ci95_low'], 1)
        self.assertEqual(m['speed_mae_ci95_high'], 1)
        self.assertEqual(m['speed_relative_mae_pct'], 100)
        self.assertEqual(m['vector_tip_relative_mae_pct'], 100)
        self.assertEqual(m['heading_cosine_similarity'], 1)
        self.assertEqual(m['speed_rmse'], 1)
        self.assertEqual(m['vector_tip_mae'], 1)
        self.assertEqual(m['vector_rmse'], 1)

    def test_wrap_boundary(self):
        np = module.np
        angles = np.deg2rad([179, -179])
        vectors = np.column_stack([np.sin(angles), np.cos(angles)])
        m = module.metrics(vectors[:1], vectors[1:])
        self.assertAlmostEqual(m['heading_mae_deg'], 2)
        self.assertAlmostEqual(m['heading_relative_mae_pct'], 2 / 180 * 100)

    def test_zero_prediction_is_not_success(self):
        m = module.metrics([[0, 0]], [[0, 1]])
        self.assertIsNone(m['heading_mae_deg'])
        self.assertEqual(m['heading_within_15_pct'], 0)
        self.assertEqual(m['undefined_heading_count'], 1)

    def test_heading_bins_are_centred_on_cardinal_directions(self):
        np = module.np
        angles = np.deg2rad([0, 90, 180, -90])
        target = np.column_stack([np.sin(angles), np.cos(angles)])
        bins = module.binned_metrics(target, target, dimension='heading', bin_counts=[4])['4']
        self.assertEqual([bin_data['label'] for bin_data in bins],
                         ['Front', 'Right', 'Back', 'Left'])
        self.assertEqual([bin_data['metrics']['n'] for bin_data in bins], [1, 1, 1, 1])

    def test_speed_bins_cover_edges_without_dropping_maximum(self):
        target = [[0, 1], [0, 2], [0, 3]]
        bins = module.binned_metrics(target, target, dimension='speed', bin_counts=[3])['3']
        self.assertEqual(sum(bin_data['metrics']['n'] for bin_data in bins), 3)
        self.assertTrue(bins[-1]['upper_inclusive'])

    def test_more_bins_than_samples_yield_empty_strata(self):
        bins = module.binned_metrics([[0, 1]], [[0, 1]], dimension='speed', bin_counts=[3])['3']
        self.assertEqual(sum(bin_data['metrics']['n'] for bin_data in bins), 1)
        self.assertTrue(any(bin_data['metrics']['n'] == 0 for bin_data in bins))


if __name__ == '__main__':
    unittest.main()
