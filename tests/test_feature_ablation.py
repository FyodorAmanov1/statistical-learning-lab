"""Feature removals and the separate fitting/selection/calibration boundaries."""
import json
import unittest
from unittest.mock import patch

import numpy as np

from src import studio


class FeatureAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = studio.generate(seed=200)
        cls.record = studio.feature_ablation_record(*cls.data[:3])
        cls.protocol = studio.feature_ablation_protocol((200, 201, 202))

    def test_groups_are_disjoint_and_removed_not_zeroed(self):
        self.assertEqual(studio.FEATURE_GROUPS,
                         {"Latest shock": (0,), "Smoothed magnitude": (1, 2, 4),
                          "Signed returns": (3, 5)})
        x, y, time, _ = self.data
        original = studio.ridge_fit
        calls = []

        def audit(a, b, alpha):
            name = list(studio.ABLATION_COLUMNS)[len(calls) // 4]
            columns = studio.ABLATION_COLUMNS[name]
            np.testing.assert_array_equal(a, x[time < 1000][:, columns])
            np.testing.assert_array_equal(b, y[time < 1000])
            calls.append(alpha)
            return original(a, b, alpha)

        with patch.object(studio, "ridge_fit", audit):
            record = studio.feature_ablation_record(x, y, time)
        self.assertEqual(calls, list(studio.ABLATION_ALPHAS) * 4)
        self.assertEqual(record['counts'],
                         dict(train=979, validation=199, calibration=249, test=549))
        self.assertEqual([len(s['columns']) for s in record['selection'].values()], [6, 5, 3, 4])

    def test_validation_scores_select_both_levels_and_ties(self):
        x, y, time, _ = self.data
        training = time < 1000
        validation = (time >= 1001) & (time < 1200)
        for name, state in self.record['selection'].items():
            cols = state['columns']
            for alpha, score in state['validation_mse'].items():
                predict = studio.ridge_fit(x[training][:, cols], y[training], float(alpha))
                expected = np.mean((predict(x[validation][:, cols]) - y[validation])**2)
                self.assertEqual(score, expected)
            best = min(state['validation_mse'], key=state['validation_mse'].get)
            self.assertEqual(state['selected_alpha'], float(best))
        self.assertEqual(self.record['selected_variant'], min(
            self.record['selection'], key=lambda name:
            self.record['selection'][name]['selected_validation_mse']))
        models, chosen = studio.fit_ablation_models(np.zeros((10, 6)), np.zeros(10),
                                                    np.zeros((5, 6)), np.zeros(5))
        self.assertEqual(chosen, 'Full ridge')
        self.assertTrue(all(m['selected_alpha'] == 0.01 for m in models.values()))

    def test_test_feature_and_label_mutations_leave_selection_and_radii_frozen(self):
        x, y, time, _ = self.data
        for feature_change, label_change in ((True, False), (False, True), (True, True)):
            changed_x, changed_y = x.copy(), y.copy()
            if feature_change:
                changed_x[time >= 1451] = 100 + 7 * changed_x[time >= 1451]
            if label_change:
                changed_y[time >= 1451] += 1000
            other = studio.feature_ablation_record(changed_x, changed_y, time)
            self.assertEqual(other['selected_variant'], self.record['selected_variant'])
            self.assertEqual(other['selection'], self.record['selection'])
            for name in self.record['models']:
                self.assertEqual(other['models'][name]['interval_radius'],
                                 self.record['models'][name]['interval_radius'])
            self.assertNotEqual(other['models']['Full ridge']['mae'],
                                self.record['models']['Full ridge']['mae'])

    def test_calibration_only_changes_intervals_not_selection_or_point_errors(self):
        x, y, time, _ = self.data
        changed = y.copy()
        changed[(time >= 1201) & (time < 1450)] += 1000
        other = studio.feature_ablation_record(x, changed, time)
        self.assertEqual(other['selection'], self.record['selection'])
        self.assertEqual(other['selected_variant'], self.record['selected_variant'])
        for name, scores in self.record['models'].items():
            for measure in ('mae', 'rmse', 'qlike', 'floor_fraction'):
                self.assertEqual(other['models'][name][measure], scores[measure])
            self.assertNotEqual(other['models'][name]['interval_radius'], scores['interval_radius'])

    def test_validation_mutation_is_a_sensitive_selection_control(self):
        x, y, time, _ = self.data
        changed = y.copy()
        changed[(time >= 1001) & (time < 1200)] += 1000
        other = studio.feature_ablation_record(x, changed, time)
        self.assertNotEqual(other['selection'], self.record['selection'])

    def test_exact_boundaries_and_excluded_rows_have_no_influence(self):
        x, y, time, _ = self.data
        self.assertEqual(self.record['target_ranges'], {
            'train': [21, 999], 'validation': [1001, 1199],
            'calibration': [1201, 1449], 'test': [1451, 1999]})
        self.assertEqual(self.record['excluded_target_times'], [1000, 1200, 1450])
        changed_x, changed_y = x.copy(), y.copy()
        excluded = np.isin(time, [1000, 1200, 1450])
        changed_x[excluded] += 1000
        changed_y[excluded] += 1000
        self.assertEqual(studio.feature_ablation_record(changed_x, changed_y, time), self.record)

    def test_rejects_ambiguous_histories_and_empty_fixed_windows(self):
        x, y, time, _ = self.data
        bad_time = time.copy()
        bad_time[1] = bad_time[0]
        fractional = time.astype(float)
        fractional[0] += 0.5
        nonfinite = x.copy()
        nonfinite[0, 0] = np.nan
        for a, b, t in ((x[:, :5], y, time), (x, y[:-1], time),
                        (x, y, time[:-1]), (x, y, bad_time), (x, y, fractional),
                        (x[::-1], y[::-1], time[::-1]),
                        (x[::-1], y[::-1], time[::-1].astype(np.uint64)),
                        (nonfinite, y, time), (x[:100], y[:100], time[:100])):
            with self.subTest(shape=a.shape, times=t[:2]):
                with self.assertRaises(ValueError):
                    studio.feature_ablation_record(a, b, t)

    def test_training_mean_is_feature_free_and_training_only(self):
        x, y, time, _ = self.data
        truth = y[time >= 1451]
        constant = y[time < 1000].mean()
        baseline = self.record['models']['Training mean']
        self.assertEqual(baseline['rmse'], np.sqrt(np.mean((truth - constant)**2)))
        changed_x = 1000 + 7*x
        other = studio.feature_ablation_record(changed_x, y, time)
        self.assertEqual(other['models']['Training mean'], baseline)

    def test_exported_isolation_audit_measures_invariance_and_sensitivity(self):
        audit = self.protocol['isolation_audit']
        self.assertEqual(audit['seed'], 200)
        checks = audit['checks']
        for name, check in checks.items():
            self.assertTrue(check['selection_unchanged'], name)
        for name in ('test_features', 'test_targets'):
            self.assertEqual(checks[name]['max_radius_change'], 0)
            self.assertGreater(checks[name]['max_test_rmse_change'], 0)
        self.assertGreater(checks['calibration_targets']['max_radius_change'], 0)
        self.assertEqual(checks['calibration_targets']['max_test_rmse_change'], 0)
        self.assertTrue(checks['excluded_rows']['entire_record_unchanged'])
        cases = studio.feature_ablation_cases(self.protocol)
        self.assertTrue(all(any('Recorded isolation audit' in m for m in case['method'])
                            for case in cases))

    def test_calibration_rank_and_selected_variant_alias(self):
        x, y, time, _ = self.data
        for name, state in self.record['selection'].items():
            cols = state['columns']
            predict = studio.ridge_fit(x[time < 1000][:, cols], y[time < 1000], state['selected_alpha'])
            cal = (time >= 1201) & (time < 1450)
            expected = np.sort(abs(y[cal] - predict(x[cal][:, cols])))[224]
            self.assertEqual(self.record['models'][name]['interval_radius'], expected)
        self.assertEqual(self.record['models']['Validation selected'],
                         self.record['models'][self.record['selected_variant']])

    def test_paired_summaries_and_prebreak_invariance(self):
        p = self.protocol
        for a, b in zip(p['records'][::2], p['records'][1::2]):
            self.assertEqual(a['seed'], b['seed'])
            self.assertEqual(a['selection'], b['selection'])
            self.assertEqual(a['selected_variant'], b['selected_variant'])
            for name in a['models']:
                self.assertEqual(a['models'][name]['interval_radius'], b['models'][name]['interval_radius'])
        for regime, summary in p['regimes'].items():
            runs = [r for r in p['records'] if r['regime'] == regime]
            self.assertEqual(sum(summary['selection_counts'].values()), 3)
            for name in summary['models']:
                for measure, s in summary['models'][name].items():
                    values = [r['models'][name][measure] for r in runs]
                    self.assertAlmostEqual(s['mean'], np.mean(values))
                    self.assertAlmostEqual(s['sd'], np.std(values, ddof=1))
                if name == 'Full ridge':
                    continue
                delta = np.array([r['models'][name]['rmse'] - r['models']['Full ridge']['rmse'] for r in runs])
                s = summary['paired_differences'][name + ' minus Full ridge']['rmse']
                self.assertAlmostEqual(s['mean'], delta.mean())
                self.assertEqual(s['lower_count'], int(np.sum(delta < 0)))
        json.dumps(p, allow_nan=False)
        for seeds in ((1,), (1, 1), (-1, 2), (True, 2), (1.5, 2)):
            with self.assertRaises(ValueError):
                studio.feature_ablation_protocol(seeds)

    def test_studio_exports_comparisons_and_aligned_charts(self):
        cases = studio.feature_ablation_cases(self.protocol)
        self.assertEqual([c['id'] for c in cases], ['ablation-stationary', 'ablation-shift'])
        for case in cases:
            self.assertTrue(case['method'])
            self.assertTrue(case['limitations'])
            self.assertEqual(len(case['table']['rows']), 50)
            for row in case['table']['rows']:
                self.assertEqual(len(row), len(case['table']['columns']))
            for chart in case['charts']:
                for series in chart['series']:
                    self.assertEqual(len(series['values']), len(chart['x']))


if __name__ == '__main__':
    unittest.main()
