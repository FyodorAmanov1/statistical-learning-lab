import unittest
from unittest.mock import patch

import numpy as np

import lab


class ScalerTests(unittest.TestCase):
    def test_known_population_statistics_and_bias_passthrough(self):
        training = np.array([[1, 10, 1], [3, 14, 1]], dtype=int)
        before = training.copy()
        scaler = lab.TrainingFittedScaler(columns=(0, 1))
        transformed = scaler.fit_transform(training)
        np.testing.assert_array_equal(scaler.mean_, [2, 12])
        np.testing.assert_array_equal(scaler.scale_, [1, 2])
        np.testing.assert_array_equal(transformed, [[-1, -1, 1], [1, 1, 1]])
        np.testing.assert_array_equal(training, before)

    def test_transform_uses_frozen_training_statistics_and_copies(self):
        scaler = lab.TrainingFittedScaler().fit([[1, 10], [3, 14]])
        mean, scale = scaler.mean_.copy(), scaler.scale_.copy()
        holdout = np.array([[1002.0, -988.0]])
        before = holdout.copy()
        result = scaler.transform(holdout)
        np.testing.assert_array_equal(result, [[1000, -500]])
        result[:] = 0
        np.testing.assert_array_equal(holdout, before)
        np.testing.assert_array_equal(scaler.mean_, mean)
        np.testing.assert_array_equal(scaler.scale_, scale)

    def test_constant_column_and_single_training_row(self):
        scaler = lab.TrainingFittedScaler().fit([[2, 9], [2, 11]])
        np.testing.assert_array_equal(scaler.scale_, [1, 1])
        np.testing.assert_array_equal(scaler.transform([[3, 12]]), [[1, 2]])
        np.testing.assert_array_equal(
            lab.TrainingFittedScaler().fit_transform([[4, 7]]), [[0, 0]])

    def test_training_input_mutation_does_not_change_fitted_statistics(self):
        training = np.array([[1.0, 10.0], [3.0, 14.0]])
        scaler = lab.TrainingFittedScaler().fit(training)
        training[:] = -999
        np.testing.assert_array_equal(scaler.mean_, [2, 12])
        np.testing.assert_array_equal(scaler.scale_, [1, 2])

    def test_rejects_unfitted_invalid_and_mismatched_inputs(self):
        with self.assertRaises(ValueError):
            lab.TrainingFittedScaler().transform([[1]])
        for invalid in ([], [1, 2], np.empty((0, 2)), [[np.nan]],
                        [[np.inf]], np.empty((2, 0))):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                lab.TrainingFittedScaler().fit(invalid)
        for columns in ((), (0, 0), (-1,), (2,), (0.5,), (True,)):
            with self.subTest(columns=columns), self.assertRaises(ValueError):
                lab.TrainingFittedScaler(columns).fit([[1, 2]])
        scaler = lab.TrainingFittedScaler().fit([[1, 2], [3, 4]])
        for invalid in ([[1]], [[1, np.nan]], [1, 2]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                scaler.transform(invalid)
        self.assertEqual(scaler.transform(np.empty((0, 2))).shape, (0, 2))


class TrainingIsolationTests(unittest.TestCase):
    def test_actual_experiment_invariant_to_holdout_features_or_targets(self):
        # Exercise the public experiment rather than just the audit's summary.
        original_x, original_y = lab.synthetic_series()
        _, _, test = lab.chronological_split(len(original_y))
        baseline = lab.experiment()
        rng = np.random.default_rng(31)
        for change in ("features", "targets", "both"):
            with self.subTest(change=change):
                x, y = original_x.copy(), original_y.copy()
                if change in ("features", "both"):
                    x[test, :2] = rng.normal(500, 100, size=(len(test), 2))
                if change in ("targets", "both"):
                    y[test] = rng.normal(-200, 50, size=len(test))
                with patch.object(lab, "synthetic_series", return_value=(x, y)):
                    changed = lab.experiment()
                self.assertEqual(changed["preprocessing"], baseline["preprocessing"])
                self.assertEqual(changed["metrics"]["selected_alpha"],
                                 baseline["metrics"]["selected_alpha"])
                self.assertEqual(changed["metrics"]["validation_mse"],
                                 baseline["metrics"]["validation_mse"])
                self.assertNotEqual(changed["metrics"]["holdout_mse"],
                                    baseline["metrics"]["holdout_mse"])
                np.testing.assert_array_equal(original_x[:test[0]], x[:test[0]])
                np.testing.assert_array_equal(original_y[:test[0]], y[:test[0]])

    def test_all_nontraining_features_cannot_change_fixed_alpha_fit(self):
        # Validation may legitimately select alpha; at a fixed alpha it must
        # not affect learned scaling or coefficients, just like test/gap rows.
        for seed in (7, 31, 1729):
            with self.subTest(seed=seed):
                x, y = lab.synthetic_series(200, seed=seed)
                train, _, _ = lab.chronological_split(len(y))
                changed = x.copy()
                changed[len(train):, :2] = 1e6
                a = lab.TrainingFittedScaler((0, 1)).fit(x[train])
                b = lab.TrainingFittedScaler((0, 1)).fit(changed[train])
                np.testing.assert_array_equal(a.mean_, b.mean_)
                np.testing.assert_array_equal(a.scale_, b.scale_)
                np.testing.assert_array_equal(a.transform(x[train]), b.transform(changed[train]))
                for alpha in (0.0, 1.0, 100.0):
                    np.testing.assert_array_equal(
                        lab.ridge_fit(a.transform(x[train]), y[train], alpha),
                        lab.ridge_fit(b.transform(changed[train]), y[train], alpha))

    def test_audit_has_nonzero_intervention_and_sensitive_leaky_control(self):
        x, y = lab.synthetic_series()
        before_x, before_y = x.copy(), y.copy()
        partitions = lab.chronological_split(len(y))
        audit = lab.training_only_audit(x, y, *partitions)
        for key in ("scaler_mean_max_abs_change", "scaler_scale_max_abs_change",
                    "transformed_training_max_abs_change", "coefficient_max_abs_change"):
            self.assertEqual(audit[key], 0.0)
        self.assertTrue(audit["selected_alpha_unchanged"])
        self.assertTrue(audit["validation_losses_unchanged"])
        for key in ("holdout_feature_max_abs_change", "holdout_target_max_abs_change",
                    "holdout_prediction_max_abs_change"):
            self.assertGreater(audit[key], 0)
        for key in ("scaler_mean_max_abs_change", "scaler_scale_max_abs_change",
                    "coefficient_max_abs_change"):
            self.assertGreater(audit["leaky_control"][key], 0)
        np.testing.assert_array_equal(x, before_x)
        np.testing.assert_array_equal(y, before_y)

    def test_refactor_matches_inline_baseline(self):
        x, y = lab.synthetic_series()
        train, validation, _ = lab.chronological_split(len(y))
        inline = x.copy()
        inline[:, :2] = (x[:, :2] - x[train, :2].mean(axis=0)) / x[train, :2].std(axis=0)
        scaler, weights, alpha, _ = lab.fit_training_model(
            x[train], y[train], x[validation], y[validation])
        np.testing.assert_allclose(scaler.transform(x), inline, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(weights, lab.ridge_fit(inline[train], y[train], alpha),
                                   rtol=1e-13, atol=1e-13)


if __name__ == "__main__":
    unittest.main()
