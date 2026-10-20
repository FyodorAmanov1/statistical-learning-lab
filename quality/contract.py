"""Independent numerical checks for reproducible experiments."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

ALLOWED_ROOT = {"lab.py", "README.md", "REFERENCES.md", "milestone.json"}
ALLOWED_DIRS = {"src", "tests", "reports", "results", "docs", "examples", "experiments"}
NAMES = {"quantum-estimation-lab", "quantum-optimization-lab", "market-microstructure-lab",
         "statistical-learning-lab", "ai-research-workbench"}


def check_paths(names):
    for name in names:
        path = Path(name.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("Unsafe changed path")
        if any(part.startswith(".") for part in path.parts):
            raise ValueError(f"Hidden/protected path changed: {name}")
        if path.as_posix() not in ALLOWED_ROOT and path.parts[0] not in ALLOWED_DIRS:
            raise ValueError(f"Policy/infrastructure change requires manual maintenance: {name}")
        if path.suffix.lower() in {".exe", ".dll", ".pem", ".key", ".sqlite", ".db", ".pkl", ".pickle"}:
            raise ValueError(f"Forbidden published file type: {name}")


def changed_paths(root: Path, base: str) -> list[str]:
    def git(*args):
        return subprocess.check_output(["git", "-c", f"safe.directory={root.resolve().as_posix()}",
                                        "-C", str(root), *args])
    tracked = git("diff", "--name-only", "-z", base).decode("utf-8").split("\0")
    new = git("ls-files", "--others", "--exclude-standard", "-z").decode("utf-8").split("\0")
    return sorted(set(p for p in tracked + new if p))


def load_lab(root: Path):
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("portfolio_checked_lab", root / "lab.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def numeric_tree(value):
    if isinstance(value, dict):
        for item in value.values():
            numeric_tree(item)
    elif isinstance(value, list):
        for item in value:
            numeric_tree(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite reported metric")


def equivalent(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equivalent(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-10)
    return a == b


def independent_checks(lab, name):
    if name == "quantum-estimation-lab":
        assert abs(lab.black_scholes_call(rate=0.05) - 10.450583572185565) < 1e-9
        assert lab.black_scholes_call(spot=90, maturity=0) == 0
        np.testing.assert_allclose(lab.discounted_payoffs([-1, 0, 1], volatility=0), 100 - 100 * np.exp(-0.03))
        result = lab.monte_carlo(paths=65536, seed=71)
        assert abs(result["estimate"] - lab.black_scholes_call()) < 6 * result["standard_error"]
    elif name == "quantum-optimization-lab":
        result = lab.exact_portfolio(np.array([1, 2, 3]), np.zeros((3, 3)), cardinality=2)
        assert result["bits"] == [0, 1, 1] and abs(result["objective"] + 5) < 1e-12
        for gamma, beta in [(0, 0), (0.31, 0.89), (2.5, 0.2)]:
            probabilities = lab.qaoa_probabilities(np.arange(8), gamma, beta)
            assert np.all(probabilities >= 0) and abs(probabilities.sum() - 1) < 1e-10
    elif name == "market-microstructure-lab":
        account = lab.Account()
        account.fill("buy", 99, quantity=2, fee=0.1)
        account.fill("sell", 101, quantity=1, fee=0.1)
        assert account.inventory == 1 and abs(account.equity(100) - 2.7) < 1e-10
        result = lab.simulate(steps=250, seed=71)
        assert abs(result["terminal_equity"] - result["terminal_cash"] - result["terminal_inventory"] * result["terminal_mid"]) < 1e-8
        assert result["max_abs_inventory"] <= 25
    elif name == "statistical-learning-lab":
        train, validation, test = lab.chronological_split(100, gap=3)
        assert train[-1] + 3 < validation[0] and validation[-1] + 3 < test[0]
        np.testing.assert_allclose(lab.ridge_fit(np.eye(2), np.array([2., 4.]), alpha=1), [1, 2])
        x, y = lab.synthetic_series(200, seed=71)
        w = lab.ridge_fit(x, y, alpha=2)
        np.testing.assert_allclose(x.T @ (x @ w - y) + 2 * w, np.zeros(x.shape[1]), atol=1e-9)
    elif name == "ai-research-workbench":
        assert lab.retrieve("quantum amplitude estimation oracle state preparation", 1)[0]["id"] == "qae"
        assert lab.answer("volcanology tectonics")["status"] == "abstained"
        answer = lab.answer("ridge regression regularization")
        known = {d["id"]: d for d in lab.CORPUS}
        assert answer["citations"] and all(c["id"] in known and c["url"] == known[c["id"]]["url"] for c in answer["citations"])


def validate(root: Path, name: str, base: str | None = None, task_id: str | None = None) -> dict:
    root = root.resolve()
    if name not in NAMES:
        raise ValueError("Unknown project")
    if base:
        changed = changed_paths(root, base)
        check_paths(changed)
        for path in changed:
            target = root / path
            if target.is_symlink() or (target.exists() and not target.resolve().is_relative_to(root)):
                raise ValueError("Symlink or junction in changed files")
            if target.is_file() and target.stat().st_size > 2_000_000:
                raise ValueError("Publish a compact artifact rather than a large binary")
    for required in ("README.md", "REFERENCES.md", "lab.py", "results/latest.json"):
        if not (root / required).is_file():
            raise ValueError(f"Missing {required}")
    lab = load_lab(root)
    independent_checks(lab, name)
    actual = json.loads(json.dumps(lab.experiment(), allow_nan=False))
    numeric_tree(actual)
    if actual.get("project") != name or not actual.get("metrics") or not actual.get("limitations") or not actual.get("provenance"):
        raise ValueError("Experiment needs metrics, limitations, and data provenance")
    recorded = json.loads((root / "results/latest.json").read_text(encoding="utf-8"))
    if not equivalent(actual["metrics"], recorded.get("metrics")):
        raise ValueError("Recorded metrics do not reproduce; regenerate results/latest.json")
    if base and not task_id and (root / "milestone.json").is_file():
        task_id = json.loads((root / "milestone.json").read_text(encoding="utf-8")).get("task_id")
    if task_id:
        import re
        if not re.fullmatch(r"[A-Z]{2}[0-9]{2}", task_id):
            raise ValueError("Invalid task identifier")
        milestone = json.loads((root / "milestone.json").read_text(encoding="utf-8"))
        if milestone.get("task_id") != task_id or milestone.get("status") != "completed":
            raise ValueError("Task is not marked complete")
        if not milestone.get("summary") or not milestone.get("evidence_paths"):
            raise ValueError("Milestone needs a summary and evidence paths")
        report = root / "reports" / f"{task_id}.md"
        if not report.is_file() or len(report.read_text(encoding="utf-8")) < 200:
            raise ValueError("Milestone needs a useful research explanation")
        for item in milestone["evidence_paths"]:
            evidence = root / item
            if not evidence.resolve().is_relative_to(root) or not evidence.is_file():
                raise ValueError("Invalid milestone evidence path")
    return {"project": name, "independent_checks": "passed", "metrics_reproduced": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--base")
    parser.add_argument("--task-id")
    args = parser.parse_args()
    print(json.dumps(validate(args.project, args.name, args.base, args.task_id)))


if __name__ == "__main__":
    main()
