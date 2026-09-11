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
        self.assertEqual(m['speed_rmse'], 1)
        self.assertEqual(m['vector_rmse'], 1)

    def test_wrap_boundary(self):
        np = module.np
        angles = np.deg2rad([179, -179])
        vectors = np.column_stack([np.sin(angles), np.cos(angles)])
        self.assertAlmostEqual(module.metrics(vectors[:1], vectors[1:])['heading_mae_deg'], 2)

    def test_zero_prediction_is_not_success(self):
        m = module.metrics([[0, 0]], [[0, 1]])
        self.assertIsNone(m['heading_mae_deg'])
        self.assertEqual(m['heading_within_15_pct'], 0)
        self.assertEqual(m['undefined_heading_count'], 1)


if __name__ == '__main__':
    unittest.main()
