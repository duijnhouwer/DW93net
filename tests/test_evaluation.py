import unittest

import numpy as np
import evaluation as module


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
        self.assertEqual(m['heading_cosine_error_pct'], 0)
        self.assertEqual(m['speed_rmse'], 1)
        self.assertEqual(m['vector_tip_mae'], 1)
        self.assertEqual(m['vector_rmse'], 1)

    def test_wrap_boundary(self):
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

    def test_streamed_percentages_and_cosines_match_detailed_metrics(self):
        from evaluation import summary_metrics
        target = np.array([[0., 1., 0.], [0., 2., 0.], [0., 3., 0.]])
        pred = np.array([[0., 1., 0.], [4., 0., 0.], [0., -3., 0.]])
        summary = summary_metrics([(pred[:1], target[:1]), (pred[1:], target[1:])], (1, 1, 1))
        detailed = module.metrics(pred[:, :2], target[:, :2])
        for key in ('heading_mae_deg', 'heading_relative_mae_pct', 'heading_cosine_similarity',
                    'heading_cosine_error_pct', 'speed_relative_mae_pct', 'vector_tip_relative_mae_pct',
                    'heading_within_15_pct', 'speed_rmse', 'vector_rmse', 'undefined_heading_count'):
            self.assertAlmostEqual(summary[key], detailed[key], msg=key)
        self.assertEqual(summary['heading_mae_deg'], 90)
        self.assertEqual(summary['heading_relative_mae_pct'], 50)
        self.assertEqual(summary['heading_cosine_error_pct'], 50)
        self.assertEqual(summary['heading_cosine_similarity'], 0)

    def test_zero_truth_is_counted_as_undefined_not_infinite_percentage(self):
        from evaluation import summary_metrics
        import json
        pred = np.array([[1., 0., 0.]])
        target = np.zeros((1, 3))
        summary = summary_metrics([(pred, target)], (1, 1, 1))
        detailed = module.metrics(pred[:, :2], target[:, :2])
        for value in (summary, detailed):
            self.assertEqual(value['undefined_relative_error_count'], 1)
            self.assertIsNone(value['speed_relative_mae_pct'])
            self.assertIsNone(value['vector_tip_relative_mae_pct'])
            self.assertIsNone(value['heading_cosine_error_pct'])
            json.dumps(value, allow_nan=False)

    def test_heading_bins_are_centred_on_cardinal_directions(self):
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
