"""Ridge regression and chronological validation on synthetic autoregressive data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


class TrainingFittedScaler:
    """Fit population mean/std on training rows; transform without refitting.

    By default all columns are standardized. Pass explicit column indices to
    preserve features such as an intercept. Constant selected columns use scale
    one and are centered. Inputs and learned statistics are never mutated by
    transform. The caller must supply training rows only to fit.
    """

    def __init__(self, columns=None):
        self.columns = None if columns is None else tuple(columns)

    @staticmethod
    def _matrix(x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] == 0 or not np.isfinite(x).all():
            raise ValueError("Expected a finite two-dimensional feature matrix")
        return x

    def fit(self, x):
        x = self._matrix(x)
        if len(x) == 0:
            raise ValueError("Cannot fit an empty training matrix")
        columns = tuple(range(x.shape[1])) if self.columns is None else self.columns
        if (not columns or any(isinstance(c, (bool, np.bool_)) or
                               not isinstance(c, (int, np.integer)) or
                               c < 0 or c >= x.shape[1] for c in columns) or
                len(set(columns)) != len(columns)):
            raise ValueError("Columns must be distinct valid feature indices")
        mean = x[:, columns].mean(axis=0)
        scale = x[:, columns].std(axis=0, ddof=0)
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("Training statistics overflowed")
        self.n_features_in_ = x.shape[1]
        self.columns_ = columns
        self.mean_ = mean
        self.scale_ = np.where(scale == 0, 1.0, scale)
        return self

    def transform(self, x):
        if not hasattr(self, "mean_"):
            raise ValueError("Fit the scaler on training data before transform")
        x = self._matrix(x)
        if x.shape[1] != self.n_features_in_:
            raise ValueError("Feature count differs from fitted training data")
        result = x.copy()
        result[:, self.columns_] = (x[:, self.columns_] - self.mean_) / self.scale_
        return result

    def fit_transform(self, x):
        return self.fit(x).transform(x)


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


def fit_training_model(x_train, y_train, x_validation, y_validation):
    """Fit preprocessing/coefficients on training; select alpha on validation.

    The three-column design retains its explicit constant feature. No holdout
    argument is accepted, so holdout values cannot enter fitting or selection.
    """
    scaler = TrainingFittedScaler(columns=(0, 1)).fit(x_train)
    scaled_train = scaler.transform(x_train)
    scaled_validation = scaler.transform(x_validation)
    validation_losses = {}
    for alpha in (0.0, 0.1, 1.0, 10.0, 100.0):
        weights = ridge_fit(scaled_train, y_train, alpha)
        validation_losses[alpha] = float(np.mean((scaled_validation @ weights - y_validation)**2))
    chosen = min(validation_losses, key=validation_losses.get)
    weights = ridge_fit(scaled_train, y_train, chosen)
    return scaler, weights, chosen, validation_losses


def training_only_audit(x, y, train, validation, test):
    """Controlled intervention on held-out features and labels, not new data."""
    changed_x, changed_y = x.copy(), y.copy()
    changed_x[test, 0] = 1000.0 + 10.0 * x[test, 0]
    changed_x[test, 1] = -500.0 + 20.0 * x[test, 1]
    changed_y[test] = 100.0 - 7.0 * y[test]
    original = fit_training_model(x[train], y[train], x[validation], y[validation])
    changed = fit_training_model(changed_x[train], changed_y[train],
                                 changed_x[validation], changed_y[validation])
    scaler, weights, alpha, losses = original
    other_scaler, other_weights, other_alpha, other_losses = changed
    # Exercise inference on both holdouts before comparing the fitted state.
    predictions = scaler.transform(x[test]) @ weights
    changed_predictions = other_scaler.transform(changed_x[test]) @ other_weights

    def max_delta(a, b):
        return float(np.max(np.abs(a - b)))

    # Deliberately incorrect full-data preprocessing: a sensitivity control.
    leaky = TrainingFittedScaler(columns=(0, 1)).fit(x)
    changed_leaky = TrainingFittedScaler(columns=(0, 1)).fit(changed_x)
    leaky_weights = ridge_fit(leaky.transform(x[train]), y[train], alpha)
    changed_leaky_weights = ridge_fit(changed_leaky.transform(x[train]), y[train], alpha)
    return {
        "intervention": "Only test rows: x0=1000+10*x0, x1=-500+20*x1, y=100-7*y; constant unchanged",
        "holdout_feature_max_abs_change": max_delta(x[test], changed_x[test]),
        "holdout_target_max_abs_change": max_delta(y[test], changed_y[test]),
        "scaler_mean_max_abs_change": max_delta(scaler.mean_, other_scaler.mean_),
        "scaler_scale_max_abs_change": max_delta(scaler.scale_, other_scaler.scale_),
        "transformed_training_max_abs_change": max_delta(scaler.transform(x[train]), other_scaler.transform(changed_x[train])),
        "coefficient_max_abs_change": max_delta(weights, other_weights),
        "selected_alpha_unchanged": alpha == other_alpha,
        "validation_losses_unchanged": losses == other_losses,
        "holdout_prediction_max_abs_change": max_delta(predictions, changed_predictions),
        "original_holdout_mse": float(np.mean((predictions - y[test])**2)),
        "changed_holdout_mse": float(np.mean((changed_predictions - changed_y[test])**2)),
        "leaky_control": {
            "description": "Incorrect full-data scaler, same training labels and fixed original selected alpha",
            "scaler_mean_max_abs_change": max_delta(leaky.mean_, changed_leaky.mean_),
            "scaler_scale_max_abs_change": max_delta(leaky.scale_, changed_leaky.scale_),
            "coefficient_max_abs_change": max_delta(leaky_weights, changed_leaky_weights),
        },
    }


def experiment():
    x, y = synthetic_series()
    train, validation, test = chronological_split(len(y))
    scaler, weights, chosen, validation_losses = fit_training_model(
        x[train], y[train], x[validation], y[validation])
    # Holdout is evaluated only after selection; preprocessing and coefficients stay training-only.
    test_mse = float(np.mean((scaler.transform(x[test]) @ weights - y[test])**2))
    baseline_mse = float(np.mean((y[test] - y[train].mean())**2))
    from src.studio import repeated_seed_protocol
    return {"project": "statistical-learning-lab", "seed": 1729,
            "repeated_seed_protocol": repeated_seed_protocol(),
            "metrics": {"selected_alpha": chosen, "validation_mse": validation_losses,
                        "holdout_mse": test_mse, "training_mean_baseline_mse": baseline_mse,
                        "train_count": len(train), "validation_count": len(validation), "test_count": len(test)},
            "preprocessing": {"fit_partition": "train", "columns": list(scaler.columns_),
                              "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                              "ddof": 0, "constant_feature": "Column 2 passed through unchanged",
                              "coefficients": weights.tolist()},
            "training_only_audit": training_only_audit(x, y, train, validation, test),
            "provenance": {"data": "Synthetic AR(1) series with a prescribed coefficient shift", "numpy": np.__version__,
                           "cost_assumptions": "Local CPU and NumPy only; no paid services. Runtime, energy and monetary cost not measured."},
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
        args.output.write_text(result, encoding="utf-8", newline="\n")
    else:
        print(result)
