# Applied case study: RegimeLab

## Research question

Build causal forecasting and uncertainty tools that remain interpretable under regime changes, using fundamental ML and useful integrations.

## Reproduce

`python src/studio.py --out docs/study.json`

## Stable regime

Predict next-day realized variance from six causal return features. Compare a small neural network, regularized regression, persistence and an expanding-window adaptation under a controlled volatility shock.

### Method

Seed 73: Gaussian innovations with v[t]=ω+α r[t−1]²+β v[t−1]. Baseline (ω,α,β)=(0.025,0.10,0.84); from day 1450 the stress case uses (0.12,0.16,0.84), deliberately α+β=1. Features at close t: squared return, 5/20-day variance, return, 20-day mean absolute return and 5-day mean return. Label: r² at t+1.

Training targets <1000; validation 1001–1199 chooses ridge penalty 0.01; calibration 1201–1449 sets a finite-sample absolute-residual quantile; test starts 1451. Scaling is fitted on training only. Embargoed boundaries prevent target overlap.

A 6→12→1 tanh MLP trains for 350 full-batch epochs on log1p variance, with fixed settings and training-only normalization. No claim that this architecture is optimal is made.

Frozen ridge interval radius = 0.860936; lower endpoint clipped at zero. Expanding ridge refits every 60 targets, uses only earlier observed labels and keeps the previously selected penalty. It does not reuse the frozen interval as an adaptive guarantee.

### Measured comparison

| Model | Test MAE | Test RMSE | QLIKE (lower is better) |
| --- | --- | --- | --- |
| Training mean | 0.400502 | 0.61512 | 0.130759 |
| Lagged square | 0.531393 | 0.842062 | 305.556 |
| Ridge | 0.403031 | 0.607059 | 0.0950162 |
| Nonlinear MLP | 0.376228 | 0.611575 | 0.128499 |
| Expanding ridge | 0.403785 | 0.603005 | 0.0880208 |

### Limits

- Synthetic heteroscedastic data tests a known mechanism; it is not a historical market backtest or evidence of alpha. Squared returns are noisy variance proxies.
- The post-break stress has unit variance persistence and no finite stationary unconditional variance. It is a deliberately nonstationary stress, not a stable high-volatility regime.
- Serial dependence and deliberate distribution shift violate exchangeability. Split-conformal intervals here are diagnostics; 90% empirical coverage is not guaranteed under these conditions.
- Mean absolute error, RMSE and QLIKE can rank models differently. QLIKE uses a 10⁻⁵ positivity floor, disclosed because near-zero forecasts can dominate it.
- Only one process seed is shown. Hyperparameters are selected without test data, but broad model selection uncertainty and transaction costs are not quantified.

## Volatility regime break

Predict next-day realized variance from six causal return features. Compare a small neural network, regularized regression, persistence and an expanding-window adaptation under a controlled volatility shock.

### Method

Seed 73: Gaussian innovations with v[t]=ω+α r[t−1]²+β v[t−1]. Baseline (ω,α,β)=(0.025,0.10,0.84); from day 1450 the stress case uses (0.12,0.16,0.84), deliberately α+β=1. Features at close t: squared return, 5/20-day variance, return, 20-day mean absolute return and 5-day mean return. Label: r² at t+1.

Training targets <1000; validation 1001–1199 chooses ridge penalty 0.01; calibration 1201–1449 sets a finite-sample absolute-residual quantile; test starts 1451. Scaling is fitted on training only. Embargoed boundaries prevent target overlap.

A 6→12→1 tanh MLP trains for 350 full-batch epochs on log1p variance, with fixed settings and training-only normalization. No claim that this architecture is optimal is made.

Frozen ridge interval radius = 0.860936; lower endpoint clipped at zero. Expanding ridge refits every 60 targets, uses only earlier observed labels and keeps the previously selected penalty. It does not reuse the frozen interval as an adaptive guarantee.

### Measured comparison

| Model | Test MAE | Test RMSE | QLIKE (lower is better) |
| --- | --- | --- | --- |
| Training mean | 10.0136 | 21.6421 | 25.7154 |
| Lagged square | 13.0142 | 24.5081 | 440.175 |
| Ridge | 9.19081 | 19.813 | 36624.1 |
| Nonlinear MLP | 9.85223 | 21.5022 | 17.8067 |
| Expanding ridge | 9.83873 | 18.3398 | 3.18396 |

### Limits

- Synthetic heteroscedastic data tests a known mechanism; it is not a historical market backtest or evidence of alpha. Squared returns are noisy variance proxies.
- The post-break stress has unit variance persistence and no finite stationary unconditional variance. It is a deliberately nonstationary stress, not a stable high-volatility regime.
- Serial dependence and deliberate distribution shift violate exchangeability. Split-conformal intervals here are diagnostics; 90% empirical coverage is not guaranteed under these conditions.
- Mean absolute error, RMSE and QLIKE can rank models differently. QLIKE uses a 10⁻⁵ positivity floor, disclosed because near-zero forecasts can dominate it.
- Only one process seed is shown. Hyperparameters are selected without test data, but broad model selection uncertainty and transaction costs are not quantified.

## Interpretation

The neural model improves stable-regime MAE on this seed, but model rankings depend on the loss. The stressed process deliberately has unit variance persistence. Frozen interval coverage deteriorates sharply, and expanding refits improve some scores while worsening MAE. These outcomes demonstrate why adaptation and uncertainty claims require separate evaluation.

## Concepts to explain in an interview

Explain the objective and units, where each baseline obtains its information, what was fixed before evaluation, how costs and uncertainty were measured, and one failure the model cannot capture. Reproduce the data before presenting a number; distinguish an implementation demonstration from a validated research conclusion.

## Next work

- Completed repeated-seed evaluation: see [ML02](ML02.md) and the two repeated-seed studio cases. Further generator and shift mechanisms remain open.
- Add feature ablations and compare likelihood-aware training objectives without tuning on the final holdout.
- Provide a validated user-CSV forecasting adapter and compare the same causal protocol on licensed public data.

## Repeated-seed evidence

The original seed-73 trajectories above are preserved. Repeating their full selection, calibration and causal evaluation procedure on process seeds 100-119 gives 20 independent stable/stress pairs. Neural initialization remains fixed at 53; ridge alpha is validation-selected per dataset and frozen for its test and expanding refits.

Stable mean interval coverage is 88.77% (seed SD 7.50 pp; p10-p90 83.64-95.46%), versus 18.39% (SD 9.70 pp; p10-p90 9.49-29.29%) under stress. Expanding ridge improves stress RMSE in 16/20 seeds but worsens MAE in all 20. Extreme stress paths dominate some mean errors and QLIKE; none are excluded.

The studio now exposes both ensemble cases with per-seed charts and downloadable paired summaries. [The repeated-seed report](ML02.md) explains actual results, the independent sampling unit, selection boundaries, variability measures, and the limitations of this synthetic experiment.
