"""Independent checks of seed aggregation and causal forecasting boundaries."""
import json
import unittest
from unittest.mock import patch

import numpy as np

from src import studio


class RepeatedForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.legacy = studio.experiment(include_repeated=False)
        cls.protocol = studio.repeated_seed_protocol((100, 101, 102))

    def test_legacy_scores_and_partition_sizes(self):
        stable, stress = self.legacy['cases']
        self.assertEqual([c['id'] for c in self.legacy['cases']], ['stationary', 'shift'])
        self.assertAlmostEqual(stable['table']['rows'][2][1], 0.403031, places=6)
        self.assertAlmostEqual(stress['table']['rows'][2][1], 9.190811689009081)
        self.assertAlmostEqual(stress['table']['rows'][4][2], 18.33977582813258)
        self.assertEqual(stable['protocol_record']['counts'],
                         dict(train=979, validation=199, calibration=249, test=549))

    def test_independent_histories_with_paired_prebreak_data(self):
        a = studio.generate(seed=100)
        b = studio.generate(shift=True, seed=100)
        c = studio.generate(seed=101)
        before = a[2] < 1450
        np.testing.assert_array_equal(a[0][before], b[0][before])
        np.testing.assert_array_equal(a[1][before], b[1][before])
        self.assertFalse(np.array_equal(a[1], c[1]))
        # Longer generation cannot rewrite previously observed features/targets.
        short = studio.generate(seed=100, n=1500)
        np.testing.assert_array_equal(a[0][:len(short[0])], short[0])
        np.testing.assert_array_equal(a[1][:len(short[1])], short[1])

    def test_summary_known_values_and_validation(self):
        s = studio.seed_summary([1, 2, 3])
        self.assertEqual(s['mean'], 2)
        self.assertEqual(s['sd'], 1)
        self.assertAlmostEqual(s['mcse_mean'], 1 / np.sqrt(3))
        self.assertAlmostEqual(s['p10'], 1.2)
        self.assertAlmostEqual(s['p90'], 2.8)
        for values in ([1], [1, np.nan], [[1, 2]]):
            with self.assertRaises(ValueError):
                studio.seed_summary(values)
        for seeds in ((1,), (1, 1), (1, -2), (True, 2), (1.5, 2)):
            with self.assertRaises(ValueError):
                studio.repeated_seed_protocol(seeds)

    def test_aggregation_is_equal_weight_and_paired(self):
        p = self.protocol
        self.assertEqual(len(p['records']), 6)
        for regime, summary in p['regimes'].items():
            runs = [r for r in p['records'] if r['regime'] == regime]
            for model, measures in summary['models'].items():
                for loss, s in measures.items():
                    values = np.array([r['models'][model][loss] for r in runs])
                    self.assertAlmostEqual(s['mean'], values.mean())
                    self.assertAlmostEqual(s['sd'], values.std(ddof=1))
            delta = np.array([r['models']['Expanding ridge']['rmse'] -
                              r['models']['Ridge']['rmse'] for r in runs])
            paired = summary['paired_differences']['Expanding ridge minus Ridge']['rmse']
            self.assertAlmostEqual(paired['mean'], delta.mean())
            self.assertEqual(paired['wins'], int(np.sum(delta < 0)))
            self.assertEqual(sum(summary['alpha_counts'].values()), 3)
        for a, b in zip(p['records'][::2], p['records'][1::2]):
            self.assertEqual(a['selected_alpha'], b['selected_alpha'])
            self.assertEqual(a['validation_mse'], b['validation_mse'])
            self.assertEqual(a['interval_radius'], b['interval_radius'])
        json.dumps(p, allow_nan=False)

    def test_holdout_mutation_cannot_change_selection_or_calibration(self):
        original = studio.generate
        def changed(shift=False, seed=73, n=2000):
            x, y, time, variance = original(shift, seed, n)
            x[time >= 1451] += 100
            y[time >= 1451] += 200
            return x, y, time, variance
        with patch.object(studio, 'generate', changed):
            mutated = studio.experiment(include_repeated=False)
        for a, b in zip(self.legacy['cases'], mutated['cases']):
            for field in ('selected_alpha', 'validation_mse', 'interval_radius'):
                self.assertEqual(a['protocol_record'][field], b['protocol_record'][field])
            self.assertNotEqual(a['table']['rows'], b['table']['rows'])

    def test_all_refits_observe_only_labels_before_block(self):
        original_generate, original_fit = studio.generate, studio.ridge_fit
        active = {}
        def capture(shift=False, seed=73, n=2000):
            data = original_generate(shift, seed, n)
            active['data'] = data
            active['calls'] = 0
            return data
        def audited_fit(x, y, alpha):
            full_x, full_y, time, _ = active['data']
            call = active['calls']
            active['calls'] += 1
            if call < 5:  # Four candidates, then final frozen fit.
                mask = time < 1000
            else:
                now = 1451 + (call - 5) * 60
                mask = time < now
            np.testing.assert_array_equal(x, full_x[mask])
            np.testing.assert_array_equal(y, full_y[mask])
            return original_fit(x, y, alpha)
        with patch.object(studio, 'generate', capture), patch.object(studio, 'ridge_fit', audited_fit):
            studio.experiment(seed=100, include_repeated=False)
        self.assertEqual(active['calls'], 15)

    def test_studio_cases_expose_all_scores_and_coverage(self):
        cases = studio.repeated_seed_cases(self.protocol)
        self.assertEqual([c['id'] for c in cases], ['repeated-stationary', 'repeated-shift'])
        for case in cases:
            self.assertEqual(len(case['table']['rows']), 26)
            self.assertEqual(len(case['table']['columns']), 7)
            for chart in case['charts']:
                for series in chart['series']:
                    self.assertEqual(len(series['values']), len(chart['x']))
            self.assertTrue(case['method'])
            self.assertTrue(case['limitations'])


if __name__ == '__main__':
    unittest.main()
