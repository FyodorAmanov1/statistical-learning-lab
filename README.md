# Statistical Learning Lab

Statistical learning fundamentals, chronological validation, and distribution shift.

## Current implementation

Closed-form ridge regression, chronological train/validation/test partitions with a gap, training-only preprocessing, and a prescribed distribution shift.

## Reproduce

Python 3.12 and CPU execution. From this repository:

    python -m pip install -r requirements.txt
    python -m unittest discover -s tests -v
    python lab.py --output results/latest.json

The measured artifact is [results/latest.json](results/latest.json). Data provenance and scientific limitations are recorded alongside the metrics. Primary references and implementation boundaries are in [REFERENCES.md](REFERENCES.md).

## ML01: training-only preprocessing audit

`TrainingFittedScaler` in `lab.py` exposes `fit`, `transform`, and `fit_transform`.
Fit it on training rows only; reuse the same object to transform validation and
test data. It uses population standard deviations (`ddof=0`), maps zero scales
to one, rejects nonfinite inputs and mismatched feature counts, and returns
copies. The experiment uses `columns=(0, 1)` to preserve the explicit constant
in column 2; by default the scaler standardizes every column.

```python
from lab import TrainingFittedScaler

scaler = TrainingFittedScaler(columns=(0, 1)).fit(x_train)
scaled_train = scaler.transform(x_train)
scaled_validation = scaler.transform(x_validation)
scaled_test = scaler.transform(x_test)
```

The existing CLI now also records fitted statistics, coefficients, and a holdout
mutation audit. `fit_training_model` accepts only training and validation data:
training fits the scaler and ridge coefficients, while validation selects alpha.
The audit changes test features and labels after generating the synthetic series.
Its deliberately incorrect full-data scaler is a sensitivity control and is
never used to select the reported model. Tests also mutate test features and
labels separately through the public `experiment()` function.

Measured training-statistic and coefficient changes were exactly zero; the leaky
control changed coefficients. Ridge still had worse holdout MSE (1.392) than the
training-mean baseline (1.204). See [reports/ML01.md](reports/ML01.md) for methods,
actual measurements, limitations, and interview concepts.

## Research process

This is an AI-assisted research portfolio developed through bounded daily milestones. GPT-6.1 Sol at medium reasoning assists with implementation. Independent numerical checks and CI guard the main branch. Reports distinguish implemented behavior, exploratory results, and future work. Automated commits do not establish original research or personal mastery.

Each completed milestone adds a report under reports/ explaining the question, method, executable evidence, limitations, and concepts to understand for an interview. Negative results are retained.

The [portfolio controller](https://github.com/FyodorAmanov1/research-portfolio) contains the backlog and runtime policy. Initial baselines are runnable starting points; the full roadmap is not yet implemented.
