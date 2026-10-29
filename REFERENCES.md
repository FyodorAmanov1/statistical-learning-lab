# Primary references and implementation boundaries

- [A gentle introduction to conformal prediction](https://arxiv.org/abs/2107.07511)
- [Forecasting: principles and practice · time-series cross-validation](https://otexts.com/fpp3/tscv.html)

The implementation, data provenance and scope limits are documented in [the technical case study](reports/applied-study.md). Citation identifies an algorithmic foundation, not equivalence with the full paper or a claim of hardware performance.

## Repeated-seed forecasting

- Demsar (2006), [Statistical Comparisons of Classifiers over Multiple Data Sets](https://www.jmlr.org/papers/volume7/demsar06a/demsar06a.pdf). Supports the dataset-level comparison perspective. Here comparisons are descriptive paired differences on synthetic regression histories; no classification significance test is claimed.
- Barber, Candes, Ramdas and Tibshirani (2023), [Conformal prediction beyond exchangeability](https://arxiv.org/abs/2202.13415). Explains the scope of conformal coverage under violations of exchangeability. Its extensions are not implemented here; our serially dependent/shifted bands are empirical diagnostics.

[ML02](reports/ML02.md) defines the seed-level SD, Monte Carlo standard error and empirical quantile conventions. Independent units are process histories; stable/stress pairs share innovations. Repetition does not validate the synthetic generator as a market model.
