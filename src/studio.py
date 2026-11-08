"""Chronological nonlinear volatility forecasts, shift diagnostics and conformal bands."""

import numpy as np

try:
    from .common import finalize, metric
except ImportError:
    from common import finalize, metric


def generate(shift=False, seed=73, n=2000):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    r = np.zeros(n)
    v = np.ones(n) * 0.8
    for t in range(1, n):
        shock = shift and t >= 1450
        # The stressed case deliberately uses alpha+beta=1 (persistent variance stress),
        # rather than assuming a stationary high-volatility distribution after the break.
        v[t] = (
            (0.12 if shock else 0.025)
            + (0.16 if shock else 0.1) * r[t - 1] ** 2
            + 0.84 * v[t - 1]
        )
        r[t] = np.sqrt(v[t]) * z[t]
    # Forecast next-day realized variance from features available at the current close.
    x = []
    y = []
    indices = []
    for t in range(20, n - 1):
        past = r[t - 19 : t + 1]
        x.append(
            [
                r[t] ** 2,
                np.mean(past[-5:] ** 2),
                np.mean(past**2),
                r[t],
                np.mean(abs(past)),
                np.mean(past[-5:]),
            ]
        )
        y.append(r[t + 1] ** 2)
        indices.append(t + 1)
    return np.array(x), np.array(y), np.array(indices), v


def ridge_fit(x, y, alpha):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-12, 1, std)
    a = np.column_stack([np.ones(len(x)), (x - mean) / std])
    penalty = np.eye(a.shape[1]) * alpha
    penalty[0, 0] = 0
    beta = np.linalg.solve(a.T @ a + penalty, a.T @ y)
    return lambda future: np.maximum(
        0, np.column_stack([np.ones(len(future)), (future - mean) / std]) @ beta
    )


def neural_fit(x, y, seed=53, epochs=350):
    """Small tanh network trained on log1p variance; all normalization is training-only."""
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), 1e-12)
    a = (x - mean) / std
    target = np.log1p(y)
    ym = target.mean()
    ys = target.std()
    target = (target - ym) / ys
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.2, (a.shape[1], 12))
    b = np.zeros(12)
    out = rng.normal(0, 0.2, 12)
    intercept = 0.0
    # Full-batch gradient descent, fixed training budget; validation never contributes gradients.
    for _ in range(epochs):
        hidden = np.tanh(a @ w + b)
        err = hidden @ out + intercept - target
        derivative = 2 * err / len(a)
        dh = derivative[:, None] * out * (1 - hidden**2)
        dw = a.T @ dh + 1e-3 * w
        db = dh.sum(axis=0)
        do = hidden.T @ derivative + 1e-3 * out
        di = derivative.sum()
        w -= 0.025 * dw
        b -= 0.025 * db
        out -= 0.025 * do
        intercept -= 0.025 * di
    return lambda future: np.maximum(
        0,
        np.expm1(
            np.clip(
                (np.tanh((future - mean) / std @ w + b) @ out + intercept) * ys + ym,
                -20,
                20,
            )
        ),
    )


def conformal_quantile(errors, coverage=0.9):
    rank = min(len(errors), int(np.ceil((len(errors) + 1) * coverage)))
    return float(np.sort(errors)[rank - 1])


def experiment(seed=73, include_repeated=True):
    """Original two cases, optionally extended with independent-seed evidence."""
    cases = []
    for shift in [False, True]:
        x, y, time, true_variance = generate(shift, seed=seed)
        # Cutoffs in target time; one-target embargo between fitting/tuning/calibration/test partitions.
        train = time < 1000
        validation = (time >= 1001) & (time < 1200)
        calibration = (time >= 1201) & (time < 1450)
        test = time >= 1451
        train_x = x[train]
        train_y = y[train]
        validation_y = y[validation]
        candidates = [0.01, 1, 10, 100]
        scores = []
        for alpha in candidates:
            predict = ridge_fit(train_x, train_y, alpha)
            scores.append(float(np.mean((predict(x[validation]) - validation_y) ** 2)))
        alpha = candidates[int(np.argmin(scores))]
        ridge = ridge_fit(train_x, train_y, alpha)
        neural = neural_fit(train_x, train_y)
        constant = float(train_y.mean())
        models = {
            "Training mean": lambda a: np.full(len(a), constant),
            "Lagged square": lambda a: a[:, 0],
            "Ridge": ridge,
            "Nonlinear MLP": neural,
        }
        predictions = {name: predict(x[test]) for name, predict in models.items()}
        truth = y[test]
        times = time[test]
        radius = conformal_quantile(abs(y[calibration] - ridge(x[calibration])))
        lower = np.maximum(0, predictions["Ridge"] - radius)
        upper = predictions["Ridge"] + radius
        coverage = float(np.mean((truth >= lower) & (truth <= upper)))
        mae = lambda a: float(np.mean(abs(a - truth)))
        # Causal expanding-window refit: fixed hyperparameter, fitting includes only targets strictly before the block.
        rolling = np.zeros(len(truth))
        for start in range(0, len(times), 60):
            now = times[start]
            allowed = time < now
            rolling[start : start + 60] = ridge_fit(x[allowed], y[allowed], alpha)(
                x[test][start : start + 60]
            )
        predictions["Expanding ridge"] = rolling
        qlike = lambda p: float(
            np.mean(np.log(np.maximum(p, 1e-5)) + truth / np.maximum(p, 1e-5))
        )
        stride = 3
        shown = np.arange(0, len(truth), stride)
        blocks = [slice(i, min(i + 60, len(truth))) for i in range(0, len(truth), 60)]
        block_x = [int(times[b.start]) for b in blocks]
        record = {
            "seed": int(seed),
            "regime": "shift" if shift else "stationary",
            "selected_alpha": alpha,
            "validation_mse": dict(zip(map(str, candidates), scores)),
            "counts": {name: int(mask.sum()) for name, mask in
                       [("train", train), ("validation", validation),
                        ("calibration", calibration), ("test", test)]},
            "coverage": coverage,
            "interval_radius": radius,
            "mean_interval_width": float(np.mean(upper - lower)),
            "models": {
                name: {"mae": mae(p), "rmse": float(np.sqrt(np.mean((p - truth)**2))),
                       "qlike": qlike(p), "floor_fraction": float(np.mean(p < 1e-5))}
                for name, p in predictions.items()
            },
        }
        cases.append(
            {
                "protocol_record": record,
                "id": "shift" if shift else "stationary",
                "label": "Volatility regime break" if shift else "Stable regime",
                "kicker": "TIME-SERIES ML / CAUSAL EVALUATION",
                "title": "A forecast is only useful out of sample.",
                "description": "Predict next-day realized variance from six causal return features. Compare a small neural network, regularized regression, persistence and an expanding-window adaptation under a controlled volatility shock.",
                "metrics": [
                    metric("Frozen ridge MAE", mae(predictions["Ridge"])),
                    metric("Expanding ridge MAE", mae(rolling)),
                    metric("Nominal interval coverage", 90, "%"),
                    metric("Observed test coverage", coverage * 100, "%"),
                ],
                "charts": [
                    {
                        "title": "Held-out realized variance and frozen prediction interval · every third target",
                        "x": times[shown].tolist(),
                        "xlabel": "Target day index",
                        "series": [
                            {
                                "name": "Realized squared return",
                                "values": truth[shown].tolist(),
                            },
                            {
                                "name": "Frozen ridge",
                                "values": predictions["Ridge"][shown].tolist(),
                            },
                            {"name": "90% lower", "values": lower[shown].tolist()},
                            {"name": "90% upper", "values": upper[shown].tolist()},
                        ],
                    },
                    {
                        "title": "Error through time · non-overlapping blocks",
                        "x": block_x,
                        "xlabel": "First target in block",
                        "series": [
                            {
                                "name": name,
                                "values": [
                                    float(np.mean(abs(p[b] - truth[b]))) for b in blocks
                                ],
                            }
                            for name, p in predictions.items()
                        ],
                    },
                    {
                        "title": "Calibration transport diagnostic",
                        "x": block_x,
                        "xlabel": "First target in block",
                        "series": [
                            {
                                "name": "Observed interval coverage / %",
                                "values": [
                                    float(
                                        np.mean(
                                            (truth[b] >= lower[b])
                                            & (truth[b] <= upper[b])
                                        )
                                        * 100
                                    )
                                    for b in blocks
                                ],
                            },
                            {"name": "Nominal / %", "values": [90] * len(blocks)},
                        ],
                    },
                ],
                "table": {
                    "columns": [
                        "Model",
                        "Test MAE",
                        "Test RMSE",
                        "QLIKE (lower is better)",
                    ],
                    "rows": [
                        [
                            name,
                            mae(p),
                            float(np.sqrt(np.mean((p - truth) ** 2))),
                            qlike(p),
                        ]
                        for name, p in predictions.items()
                    ],
                },
                "method": [
                    f"Seed {seed}: Gaussian innovations with v[t]=ω+α r[t−1]²+β v[t−1]. Baseline (ω,α,β)=(0.025,0.10,0.84); from day 1450 the stress case uses (0.12,0.16,0.84), deliberately α+β=1. Features at close t: squared return, 5/20-day variance, return, 20-day mean absolute return and 5-day mean return. Label: r² at t+1.",
                    f"Training targets <1000; validation 1001–1199 chooses ridge penalty {alpha}; calibration 1201–1449 sets a finite-sample absolute-residual quantile; test starts 1451. Scaling is fitted on training only. Embargoed boundaries prevent target overlap.",
                    "A 6→12→1 tanh MLP trains for 350 full-batch epochs on log1p variance, with fixed settings and training-only normalization. No claim that this architecture is optimal is made.",
                    f"Frozen ridge interval radius = {radius:.6f}; lower endpoint clipped at zero. Expanding ridge refits every 60 targets, uses only earlier observed labels and keeps the previously selected penalty. It does not reuse the frozen interval as an adaptive guarantee.",
                ],
                "limitations": [
                    "Synthetic heteroscedastic data tests a known mechanism; it is not a historical market backtest or evidence of alpha. Squared returns are noisy variance proxies.",
                    "The post-break stress has unit variance persistence and no finite stationary unconditional variance. It is a deliberately nonstationary stress, not a stable high-volatility regime.",
                    "Serial dependence and deliberate distribution shift violate exchangeability. Split-conformal intervals here are diagnostics; 90% empirical coverage is not guaranteed under these conditions.",
                    "Mean absolute error, RMSE and QLIKE can rank models differently. QLIKE uses a 10⁻⁵ positivity floor, disclosed because near-zero forecasts can dominate it.",
                    "This trajectory shows one process seed; the repeated-seed cases quantify variability for the same generator. Shift mechanisms, model families and transaction costs are not explored.",
                ],
                "reproduce": "python src/studio.py --out docs/study.json",
            }
        )
    study = {
        "schema_version": 1,
        "slug": "statistical-learning-lab",
        "title": "RegimeLab",
        "subtitle": "Forecasting, uncertainty and adaptation when the data changes.",
        "accent": "#78b9ff",
        "seed": seed,
        "cases": cases,
        "provenance": "Seeded synthetic returns; strictly ordered training, tuning, calibration and held-out targets. All reported models are computed locally with NumPy.",
        "references": [
            {
                "title": "A gentle introduction to conformal prediction",
                "url": "https://arxiv.org/abs/2107.07511",
            },
            {
                "title": "Forecasting: principles and practice · time-series cross-validation",
                "url": "https://otexts.com/fpp3/tscv.html",
            },
            {
                "title": "scikit-learn maintainers: data leakage and feature selection",
                "url": "https://scikit-learn.org/stable/common_pitfalls.html#data-leakage",
            },
        ],
    }

    if include_repeated:
        protocol = repeated_seed_protocol()
        study["repeated_seed_protocol"] = protocol
        study["cases"].extend(repeated_seed_cases(protocol))
        ablation = feature_ablation_protocol()
        study["feature_ablation_protocol"] = ablation
        study["cases"].extend(feature_ablation_cases(ablation))
    return study


# Fixed before inspecting the repeated-seed holdouts; legacy seed 73 is excluded.
PROCESS_SEEDS = tuple(range(100, 120))


def seed_summary(values):
    """Equal-weight dataset summaries, never treating dependent days as replicates."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("At least two finite seed-level values are required")
    sd = float(values.std(ddof=1))
    return {"n": len(values), "mean": float(values.mean()), "sd": sd,
            "mcse_mean": sd / np.sqrt(len(values)),
            "median": float(np.median(values)),
            "p10": float(np.quantile(values, 0.1)),
            "p90": float(np.quantile(values, 0.9)),
            "min": float(values.min()), "max": float(values.max())}


def repeated_seed_protocol(seeds=PROCESS_SEEDS):
    """Repeat the unchanged causal protocol; stable/stress innovations are paired."""
    seeds = tuple(seeds)
    if (len(seeds) < 2 or any(isinstance(s, (bool, np.bool_)) or
            not isinstance(s, (int, np.integer)) or s < 0 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Use at least two distinct nonnegative integer process seeds")
    records = [case["protocol_record"] for seed in seeds
               for case in experiment(int(seed), include_repeated=False)["cases"]]
    regimes = {}
    comparisons = [("Ridge", "Training mean"), ("Nonlinear MLP", "Training mean"),
                   ("Expanding ridge", "Ridge")]
    for regime in ("stationary", "shift"):
        runs = [r for r in records if r["regime"] == regime]
        models = {name: {loss: seed_summary([r["models"][name][loss] for r in runs])
                         for loss in ("mae", "rmse", "qlike", "floor_fraction")}
                  for name in runs[0]["models"]}
        paired = {}
        for model, baseline in comparisons:
            paired[f"{model} minus {baseline}"] = {}
            for loss in ("mae", "rmse", "qlike"):
                delta = [r["models"][model][loss] - r["models"][baseline][loss]
                         for r in runs]
                paired[f"{model} minus {baseline}"][loss] = {
                    **seed_summary(delta), "wins": int(np.sum(np.array(delta) < 0)),
                    "ties": int(np.sum(np.array(delta) == 0))}
        regimes[regime] = {
            "models": models, "paired_differences": paired,
            **{field: seed_summary([r[field] for r in runs]) for field in
               ("coverage", "interval_radius", "mean_interval_width")},
            "alpha_counts": {str(a): sum(r["selected_alpha"] == a for r in runs)
                             for a in (0.01, 1, 10, 100)}}
    stable = [r for r in records if r["regime"] == "stationary"]
    shifted = [r for r in records if r["regime"] == "shift"]
    return {
        "process_seeds": list(map(int, seeds)), "neural_seed": 53,
        "selection_policy": "Fixed grid [0.01, 1, 10, 100]; validation MSE selects within each dataset; first minimum wins. No test tuning or seed exclusion. Selected penalty frozen for all expanding refits. MLP settings and initialization seed fixed.",
        "aggregation": "Equal weight per independent process seed. Sample SD (ddof=1), MCSE of mean = SD/sqrt(n), linear empirical p10/p90. Quantiles describe dataset variability, not confidence or prediction intervals. Regimes paired by innovations; days are not independent replicates.",
        "records": records, "regimes": regimes,
        "paired_coverage_shift_minus_stable": seed_summary(
            [b["coverage"] - a["coverage"] for a, b in zip(stable, shifted)]),
    }


def repeated_seed_cases(protocol):
    """Studio-native tables, export and per-seed charts for repeated evidence."""
    cases = []
    for regime, summary in protocol["regimes"].items():
        runs = [r for r in protocol["records"] if r["regime"] == regime]
        rows = []
        for name, losses in summary["models"].items():
            for loss in ("mae", "rmse", "qlike"):
                s = losses[loss]
                rows.append([name, loss.upper(), s["mean"], s["sd"], s["mcse_mean"],
                             s["p10"], s["p90"]])
        for comparison, losses in summary["paired_differences"].items():
            for loss, s in losses.items():
                rows.append([comparison, loss.upper(), s["mean"], s["sd"],
                             s["mcse_mean"], s["p10"], s["p90"]])
        for field in ("coverage", "mean_interval_width"):
            s = summary[field]
            rows.append(["Frozen ridge", field, s["mean"], s["sd"], s["mcse_mean"],
                         s["p10"], s["p90"]])
        cases.append({
            "id": "repeated-" + regime,
            "label": "Repeated seeds / " + ("stable" if regime == "stationary" else "stress"),
            "kicker": "INDEPENDENT DATASETS / FROZEN PROTOCOL",
            "title": "Does the conclusion survive a new process seed?",
            "description": "Repeat all five forecasts on independent synthetic histories. Compare paired errors and frozen 90% interval coverage; each dataset has equal weight. Table coverage is a fraction; chart coverage is percent.",
            "metrics": [metric("Independent process seeds", len(runs)),
                        metric("Mean coverage", 100 * summary["coverage"]["mean"], "%"),
                        metric("Coverage SD", 100 * summary["coverage"]["sd"], "pp"),
                        metric("Expanding beats ridge / RMSE",
                               summary["paired_differences"]["Expanding ridge minus Ridge"]["rmse"]["wins"],
                               f"/ {len(runs)} seeds")],
            "charts": [{
                "title": "Frozen interval coverage across process seeds",
                "x": protocol["process_seeds"], "xlabel": "Process seed (independent datasets)",
                "series": [{"name": "Coverage / %", "values": [100*r["coverage"] for r in runs]},
                           {"name": "Nominal / %", "values": [90]*len(runs)}]},
                *[{"title": loss.upper() + " across process seeds", "x": protocol["process_seeds"],
                   "xlabel": "Process seed (independent datasets)",
                   "series": [{"name": name, "values": [r["models"][name][loss] for r in runs]}
                              for name in summary["models"]]}
                  for loss in ("mae", "rmse")]],
            "table": {"columns": ["Model / paired difference", "Measure", "Mean", "Seed SD",
                                  "MCSE of mean", "Seed p10", "Seed p90"], "rows": rows},
            "method": [
                "Process seeds: " + ", ".join(map(str, protocol["process_seeds"])) +
                "; 2,000 returns each. Stable/stress histories share innovations within each seed; independent streams across seeds. Seed 73 remains a separate illustrative trajectory.",
                "Unchanged six causal features and train <1000, validation 1001-1199, calibration 1201-1449, test 1451-1999 (549 targets). Same Gaussian GARCH-like generator and day-1450 persistent stress as the trajectory cases.",
                protocol["selection_policy"], protocol["aggregation"],
                "Frozen ridge uses absolute calibration residuals at rank ceil((249+1)*0.9)=225; lower endpoint clipped at zero. Width is actual mean upper minus lower. Expanding ridge refits every 60 labels strictly before each forecast block, including earlier evaluation observations; its intervals are not assessed.",
                "Paired table differences subtract the named baseline on the same seed; negative means lower error. QLIKE floor remains 1e-5. Per-seed scores, floor-hit fractions, alpha counts and paired coverage changes are in the study JSON.",
                "Selected alpha frequencies: " + str(summary["alpha_counts"])],
            "limitations": [
                "Twenty synthetic histories from one generator and one shift mechanism do not establish real-world forecasting accuracy or market edge. Local CPU/NumPy only; runtime, energy and monetary cost are unmeasured.",
                "Fixed neural initialization isolates process variability; optimizer-seed variability and architecture search are not measured. These summaries evaluate the fixed selection procedure, not a single universal penalty.",
                "Serial dependence and shift violate ordinary split-conformal exchangeability. Coverage is empirical, not a guaranteed 90%. Stress has alpha+beta=1 and no finite stationary unconditional variance.",
                "Heavy-tailed seed errors and near-zero forecasts can dominate means and QLIKE. SD/MCSE and empirical quantiles from a small ensemble do not justify normal-theory significance claims. Regimes are paired, not 40 independent datasets."],
            "reproduce": "python src/studio.py --out docs/study.json"})
    return cases


# Disjoint from the illustrative and ML02 histories. Groups and grid are fixed
# across datasets; no test score participates in either level of selection.
ABLATION_SEEDS = tuple(range(200, 220))
FEATURE_NAMES = ("latest squared return", "5-day realized variance",
                 "20-day realized variance", "latest return",
                 "20-day mean absolute return", "5-day mean return")
FEATURE_GROUPS = {"Latest shock": (0,), "Smoothed magnitude": (1, 2, 4),
                  "Signed returns": (3, 5)}
ABLATION_COLUMNS = {"Full ridge": tuple(range(6)), **{
    "Without " + name.lower(): tuple(c for c in range(6) if c not in columns)
    for name, columns in FEATURE_GROUPS.items()}}
ABLATION_ALPHAS = (0.01, 1, 10, 100)


def fit_ablation_models(x_train, y_train, x_validation, y_validation):
    """Select penalties and a feature variant without accepting calibration/test.

    Each removal is refitted, with its own training-only scaling. Validation MSE
    selects within each variant, then across variants. Ordered first minima
    break ties. Training coefficients are never refitted on validation rows.
    """
    models = {}
    for name, columns in ABLATION_COLUMNS.items():
        fitted = [ridge_fit(x_train[:, columns], y_train, alpha)
                  for alpha in ABLATION_ALPHAS]
        losses = [float(np.mean((predict(x_validation[:, columns]) - y_validation)**2))
                  for predict in fitted]
        chosen = int(np.argmin(losses))
        models[name] = {"columns": list(columns),
                        "selected_alpha": ABLATION_ALPHAS[chosen],
                        "validation_mse": dict(zip(map(str, ABLATION_ALPHAS), losses)),
                        "selected_validation_mse": losses[chosen],
                        "predict": fitted[chosen]}
    selected = min(models, key=lambda name: models[name]["selected_validation_mse"])
    return models, selected


def _ablation_partitions(time):
    """Fixed target-time windows; reject histories with ambiguous chronology."""
    time = np.asarray(time)
    if (time.ndim != 1 or not np.issubdtype(time.dtype, np.number)
            or np.iscomplexobj(time) or not np.isfinite(time).all()
            or np.any(time != np.floor(time)) or np.any(time[1:] <= time[:-1])):
        raise ValueError("Target times must be finite, strictly increasing integer indices")
    masks = {"train": time < 1000,
             "validation": (time >= 1001) & (time < 1200),
             "calibration": (time >= 1201) & (time < 1450),
             "test": time >= 1451}
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("The fixed protocol requires nonempty train/validation/calibration/test windows")
    return masks


def feature_ablation_record(x, y, time):
    """Evaluate frozen ablations; inputs are causal features, labels and target times."""
    x, y, time = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(time)
    if (x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES)
            or y.shape != (len(x),) or time.shape != (len(x),)
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("Expected aligned finite six-feature rows, labels and target times")
    masks = _ablation_partitions(time)
    train, validation, calibration, test = (masks[name] for name in
                                          ("train", "validation", "calibration", "test"))
    models, selected = fit_ablation_models(x[train], y[train], x[validation], y[validation])
    # Selection is complete before calibration residuals or test rows are read.
    fitted = {}
    for name, model in models.items():
        columns = model["columns"]
        predict = model["predict"]
        radius = conformal_quantile(abs(y[calibration] - predict(x[calibration][:, columns])))
        fitted[name] = (predict(x[test][:, columns]), radius)
    constant = float(y[train].mean())
    mean_radius = conformal_quantile(abs(y[calibration] - constant))
    fitted["Training mean"] = (np.full(int(test.sum()), constant), mean_radius)
    fitted["Validation selected"] = fitted[selected]
    truth = y[test]
    scores = {}
    for name, (prediction, radius) in fitted.items():
        lower = np.maximum(0, prediction - radius)
        upper = prediction + radius
        positive = np.maximum(prediction, 1e-5)
        scores[name] = {
            "mae": float(np.mean(abs(prediction - truth))),
            "rmse": float(np.sqrt(np.mean((prediction - truth)**2))),
            "qlike": float(np.mean(np.log(positive) + truth / positive)),
            "floor_fraction": float(np.mean(prediction < 1e-5)),
            "coverage": float(np.mean((truth >= lower) & (truth <= upper))),
            "interval_radius": radius,
            "mean_interval_width": float(np.mean(upper - lower))}
    return {"selected_variant": selected,
            "selection": {name: {k: v for k, v in model.items() if k != "predict"}
                          for name, model in models.items()},
            "counts": {name: int(mask.sum()) for name, mask in masks.items()},
            "target_ranges": {name: [int(time[mask][0]), int(time[mask][-1])]
                              for name, mask in masks.items()},
            "excluded_target_times": time[~(train | validation | calibration | test)].tolist(),
            "models": scores}


def feature_ablation_isolation_audit(x, y, time):
    """Measured perturbation checks, never used to select a forecasting model.

    Mutations act on constructed rows, not on the return generator: regenerating
    later features from changed returns would answer a different question.
    """
    x, y, time = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(time)
    original = feature_ablation_record(x, y, time)
    masks = _ablation_partitions(time)
    excluded = ~np.logical_or.reduce(list(masks.values()))
    checks = {}
    for name in ("test_features", "test_targets", "calibration_targets", "excluded_rows"):
        changed_x, changed_y = x.copy(), y.copy()
        if name == "test_features":
            changed_x[masks["test"]] = 100 + 7 * changed_x[masks["test"]]
        elif name == "test_targets":
            changed_y[masks["test"]] += 1000
        elif name == "calibration_targets":
            changed_y[masks["calibration"]] += 1000
        else:
            changed_x[excluded] += 1000
            changed_y[excluded] += 1000
        other = feature_ablation_record(changed_x, changed_y, time)
        checks[name] = {
            "selection_unchanged": (other["selection"] == original["selection"]
                                    and other["selected_variant"] == original["selected_variant"]),
            "max_radius_change": max(abs(other["models"][m]["interval_radius"] -
                                         scores["interval_radius"])
                                     for m, scores in original["models"].items()),
            "max_test_rmse_change": max(abs(other["models"][m]["rmse"] - scores["rmse"])
                                        for m, scores in original["models"].items()),
            "entire_record_unchanged": other == original}
    return {"scope": "First listed process seed, stable history; deterministic row perturbations, not independent datasets.",
            "interventions": "Test features = 100+7*x; test targets += 1000; calibration targets += 1000; excluded features and targets += 1000. Separate interventions on copies.",
            "checks": checks}


def feature_ablation_protocol(seeds=ABLATION_SEEDS):
    """One-group removals with paired full-feature and constant baselines."""
    seeds = tuple(seeds)
    if (len(seeds) < 2 or any(isinstance(s, (bool, np.bool_)) or
            not isinstance(s, (int, np.integer)) or s < 0 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Use at least two distinct nonnegative integer process seeds")
    records = []
    for seed in seeds:
        for shift in (False, True):
            x, y, time, _ = generate(shift, seed=int(seed))
            records.append({"seed": int(seed), "regime": "shift" if shift else "stationary",
                            **feature_ablation_record(x, y, time)})
    regimes = {}
    for regime in ("stationary", "shift"):
        runs = [r for r in records if r["regime"] == regime]
        models = {name: {measure: seed_summary([r["models"][name][measure] for r in runs])
                         for measure in runs[0]["models"][name]}
                  for name in runs[0]["models"]}
        paired = {}
        for name in models:
            if name == "Full ridge":
                continue
            paired[name + " minus Full ridge"] = {}
            for loss in ("mae", "rmse", "qlike", "coverage"):
                delta = np.array([r["models"][name][loss] - r["models"]["Full ridge"][loss]
                                  for r in runs])
                paired[name + " minus Full ridge"][loss] = {
                    **seed_summary(delta),
                    "lower_count": int(np.sum(delta < 0)), "ties": int(np.sum(delta == 0))}
        regimes[regime] = {"models": models, "paired_differences": paired,
                           "selection_counts": {name: sum(r["selected_variant"] == name for r in runs)
                                                for name in ABLATION_COLUMNS}}
    audit_x, audit_y, audit_time, _ = generate(seed=int(seeds[0]))
    audit = {"seed": int(seeds[0]),
             **feature_ablation_isolation_audit(audit_x, audit_y, audit_time)}
    return {"process_seeds": list(map(int, seeds)),
            "feature_names": list(FEATURE_NAMES),
            "removed_groups": {name: list(columns) for name, columns in FEATURE_GROUPS.items()},
            "selection_policy": "Four fixed feature variants; alpha grid [0.01, 1, 10, 100]. Training-only scaling and coefficients. Validation MSE selects alpha separately per variant, then selects a variant. First minimum wins, full ridge first. Calibration sets each frozen model's 90% residual radius; test only scores. No test ranking selects a deployed variant; no refits or seed exclusions.",
            "aggregation": "Equal weight per process seed; stable/stress share innovations. Sample SD (ddof=1), MCSE=SD/sqrt(n), linear empirical p10/p90 and range. Paired differences subtract full ridge within seed; lower_count counts negative differences (lower coverage is a failure, not a win).",
            "isolation_audit": audit,
            "records": records, "regimes": regimes}


def feature_ablation_cases(protocol):
    """Expose group removals, validation choice and paired evidence in the studio."""
    cases = []
    for regime, summary in protocol["regimes"].items():
        runs = [r for r in protocol["records"] if r["regime"] == regime]
        rows = []
        for name, measures in summary["models"].items():
            for measure in ("mae", "rmse", "qlike", "coverage", "mean_interval_width"):
                s = measures[measure]
                rows.append([name, measure, s["mean"], s["sd"], s["mcse_mean"], s["p10"], s["p90"]])
        for name, measures in summary["paired_differences"].items():
            for measure in ("mae", "rmse", "qlike", "coverage"):
                s = measures[measure]
                rows.append([name, measure, s["mean"], s["sd"], s["mcse_mean"], s["p10"], s["p90"]])
        cases.append({
            "id": "ablation-" + regime,
            "label": "Feature ablation / " + ("stable" if regime == "stationary" else "stress"),
            "kicker": "FEATURE GROUPS / VALIDATION-ONLY SELECTION",
            "title": "Which information does the variance forecast need?",
            "description": "Remove one causal feature group at a time and refit ridge. Compare full ridge, training mean and a validation-selected variant on separate test targets. Table coverage is a fraction; negative paired errors favor removal.",
            "metrics": [metric("Independent process seeds", len(runs)),
                        metric("Selected variant RMSE", summary["models"]["Validation selected"]["rmse"]["mean"]),
                        metric("Full ridge RMSE", summary["models"]["Full ridge"]["rmse"]["mean"]),
                        metric("Selected interval coverage", 100*summary["models"]["Validation selected"]["coverage"]["mean"], "%")],
            "charts": [{
                "title": "Paired test RMSE difference from full ridge (negative is lower error)",
                "x": protocol["process_seeds"], "xlabel": "Independent process seed",
                "series": [{"name": name, "values": [r["models"][name]["rmse"] - r["models"]["Full ridge"]["rmse"] for r in runs]}
                           for name in (*list(ABLATION_COLUMNS)[1:], "Validation selected")]},
                {"title": "Frozen 90% interval coverage after each removal",
                 "x": protocol["process_seeds"], "xlabel": "Independent process seed",
                 "series": [{"name": name, "values": [100*r["models"][name]["coverage"] for r in runs]}
                            for name in ABLATION_COLUMNS] + [{"name": "Nominal / %", "values": [90]*len(runs)}]}],
            "table": {"columns": ["Variant / paired difference", "Measure", "Mean", "Seed SD", "MCSE of mean", "Seed p10", "Seed p90"], "rows": rows},
            "method": [
                "Process seeds " + ", ".join(map(str, protocol["process_seeds"])) + "; 2,000 synthetic returns each. Disjoint from earlier seed-73 and seeds-100-119 studies; all paths retained.",
                "Training targets <1000 (979); validation 1001-1199 (199); calibration 1201-1449 (249); test 1451-1999 (549). Boundaries 1000/1200/1450 excluded. Same Gaussian GARCH-like generator; stress starts day 1450 with alpha+beta=1.",
                "Remove latest shock: latest squared return (column 0). Remove smoothed magnitude: 5/20-day realized variance and 20-day mean absolute return (1,2,4). Remove signed returns: latest return and 5-day mean return (3,5). Full ridge uses all six columns. Removals are separate, not cumulative; refitting estimates conditional predictive value, not a causal intervention.",
                protocol["selection_policy"], protocol["aggregation"],
                "Calibration rank is ceil((249+1)*0.9)=225, with lower endpoints clipped at zero. Training mean is a feature-free control calibrated separately. QLIKE floor is 1e-5; floor fractions, all validation scores, chosen alphas, seed-level results and selection counts are exported in study JSON.",
                "Validation selection counts: " + str(summary["selection_counts"])],
            "limitations": [
                "Synthetic data and one shift mechanism; this is no evidence of market accuracy. No neural or likelihood-objective ablation is included. Local CPU/NumPy; runtime, energy and monetary costs unmeasured.",
                "Correlated magnitude features can substitute for each other. Removing a group with penalty reselection measures this training procedure's predictive dependence, not unique feature importance or a causal effect.",
                "Selecting a variant increases validation reuse and can overfit its 199 targets. Test comparisons are descriptive; selecting a new winner from them would require fresh confirmation data. No significance claim or universal best subset is made.",
                "Serial dependence and nonstationary stress invalidate ordinary exchangeable split-conformal guarantees. Coverage is empirical. Seed summaries have limited tail resolution; near-zero predictions can dominate QLIKE."],
            "reproduce": "python src/studio.py --out docs/study.json"})
        if "isolation_audit" in protocol:
            audit = protocol["isolation_audit"]
            checks = audit["checks"]
            cases[-1]["method"].append(
                f"Recorded isolation audit (stable seed {audit['seed']}): "
                f"test-feature selection unchanged={checks['test_features']['selection_unchanged']}, "
                f"test-target selection unchanged={checks['test_targets']['selection_unchanged']}; "
                f"maximum radius changes {checks['test_features']['max_radius_change']:.6g} / "
                f"{checks['test_targets']['max_radius_change']:.6g}. "
                f"Changing excluded boundary rows leaves the entire record unchanged="
                f"{checks['excluded_rows']['entire_record_unchanged']}. "
                "Calibration-target changes affect radii while test point errors stay fixed. "
                "Exact target ranges and intervention measurements are in the JSON export. "
                "These checks diagnose this implementation; they do not prove data provenance.")
    return cases


if __name__ == "__main__":
    finalize(experiment())
