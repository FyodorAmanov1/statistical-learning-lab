import unittest
import numpy as np
import lab


class LearningTests(unittest.TestCase):
    def test_disjoint_ordered_partitions(self):
        train, validation, test = lab.chronological_split(100, gap=3)
        self.assertLess(train[-1] + 3, validation[0])
        self.assertLess(validation[-1] + 3, test[0])
        self.assertEqual(len(set(train) & set(test)), 0)

    def test_known_ridge_solution(self):
        np.testing.assert_allclose(lab.ridge_fit(np.eye(2), np.array([2, 4]), alpha=1), [1, 2])

    def test_ols_singular(self):
        weights = lab.ridge_fit(np.ones((5, 2)), np.ones(5), alpha=0)
        np.testing.assert_allclose(np.ones((5, 2)) @ weights, np.ones(5))

    def test_known_stationarity_condition(self):
        x, y = lab.synthetic_series(200)
        weights = lab.ridge_fit(x, y, alpha=2)
        np.testing.assert_allclose(x.T @ (x @ weights - y) + 2 * weights, np.zeros(3), atol=1e-10)


if __name__ == "__main__":
    unittest.main()
