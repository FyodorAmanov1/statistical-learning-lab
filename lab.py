"""Ridge regression and chronological validation on synthetic autoregressive data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def chronological_split(n, train_fraction=0.6, validation_fraction=0.2, gap=1):
    if n < 10 or gap < 0 or not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 - train_fraction:
        raise ValueError("Invalid split")
    train_end = int(n * train_fraction)
    validation_end = int(n * (train_fraction + validation_fraction))
    if train_end + gap >= validation_end or validation_end + gap >= n:
        raise ValueError("Gap leaves an empty partition")
    return np.arange(train_end), np.arange(train_end + gap, validation_end), np.arange(validation_end + gap, n)


def synthetic_series(n=1500, seed=1729):
    rng = np.random.default_rng(seed)
    values = np.zeros(n + 1)
    noise = rng.normal(size=n)
    for t in range(n):
        phi = 0.8 if t < 1200 else 0.3
        values[t + 1] = phi * values[t] + noise[t]
    # Features at t predict t+1. No target or future-derived feature is included.
    x = np.column_stack([values[:-1], np.sin(values[:-1]), np.ones(n)])
    return x, values[1:]


def ridge_fit(x, y, alpha=1.0):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 2 or y.shape != (len(x),) or alpha < 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Invalid regression data")
    # All coefficients, including the explicit constant feature, are regularized.
    if alpha == 0:
        return np.linalg.lstsq(x, y, rcond=None)[0]
    return np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ y)


def experiment():
    x, y = synthetic_series()
    train, validation, test = chronological_split(len(y))
    mean = x[train, :2].mean(axis=0)
    scale = x[train, :2].std(axis=0)
    x[:, :2] = (x[:, :2] - mean) / scale
    validation_losses = {}
    for alpha in (0.0, 0.1, 1.0, 10.0, 100.0):
        weights = ridge_fit(x[train], y[train], alpha)
        validation_losses[alpha] = float(np.mean((x[validation] @ weights - y[validation])**2))
    chosen = min(validation_losses, key=validation_losses.get)
    # Holdout is evaluated only after selection; preprocessing and coefficients stay training-only.
    weights = ridge_fit(x[train], y[train], chosen)
    test_mse = float(np.mean((x[test] @ weights - y[test])**2))
    baseline_mse = float(np.mean((y[test] - y[train].mean())**2))
    return {"project": "statistical-learning-lab", "seed": 1729,
            "metrics": {"selected_alpha": chosen, "validation_mse": validation_losses,
                        "holdout_mse": test_mse, "training_mean_baseline_mse": baseline_mse,
                        "train_count": len(train), "validation_count": len(validation), "test_count": len(test)},
            "provenance": {"data": "Synthetic AR(1) series with a prescribed coefficient shift", "numpy": np.__version__},
            "limitations": ["Single synthetic series and a single prescribed shift; not a financial forecasting result.",
                            "A one-row gap handles these one-step labels, not arbitrary overlapping targets.",
                            "Repeated model development can overfit a fixed holdout; fresh confirmation is a later milestone."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(experiment(), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
    else:
        print(result)
