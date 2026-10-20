"""Shared artifact protocol for self-contained applied research studies."""

import argparse
import json
import math
from pathlib import Path


def metric(label, value, unit="", note=""):
    return {
        "label": label,
        "value": float(value) if isinstance(value, (float, int)) else value,
        "unit": unit,
        "note": note,
    }


def _finite(item):
    if isinstance(item, dict):
        for value in item.values():
            _finite(value)
    elif isinstance(item, list):
        for value in item:
            _finite(value)
    elif isinstance(item, float) and not math.isfinite(item):
        raise ValueError("Non-finite study output")


def _same(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(
            _same(a[k], b[k]) for k in a if k not in {"elapsed_seconds"}
        )
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=2e-4, abs_tol=2e-6)
    return a == b


def finalize(study):
    parser = argparse.ArgumentParser(description=study["title"])
    parser.add_argument("--out", type=Path, default=Path("docs/study.json"))
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    study = json.loads(json.dumps(study, allow_nan=False))
    _finite(study)
    assert study["cases"] and len({c["id"] for c in study["cases"]}) == len(
        study["cases"]
    )
    for case in study["cases"]:
        assert case["metrics"] and case["method"] and case["limitations"]
        for chart in case.get("charts", []):
            for series in chart["series"]:
                assert len(series["values"]) == len(chart["x"])
    if args.check:
        recorded = json.loads(args.check.read_text(encoding="utf-8"))
        if not _same(study, recorded):
            raise SystemExit(
                "Recorded study does not reproduce. Regenerate it and inspect the change."
            )
        print(
            f"Reproduced {len(study['cases'])} applied study cases and their numerical invariants."
        )
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(study, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(f"Wrote {args.out}: {len(study['cases'])} measured cases")
