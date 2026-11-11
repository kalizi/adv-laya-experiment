"""Audit `OER@0.75` across every archived run: is the threshold one thing?

`record["confidence"]` is not one quantity. On `k=2` (noul) the backends put
`max(p, 1-p)` in it, on `k>2` the normalised entropy `1 - H(p)/log(k)`, and the two
numerically disagree: at the same distribution they read 0.726 and 0.152. So the
`tau=0.75` that `metrics.core.oer` applies is a majority test on one primitive and
a near-flat-distribution test on another, and `aggregate()` used to report the
pooled figure with no split by primitive.

This walks every `preds.jsonl` under `results/`, recomputes the Overconfident Error
Rate on both `confidence` and on the backend-independent margin, and prints where
the two disagree. Rows already published are not touched: the point is to size the
blast radius, not to restate history.

Usage:
    uv run python tools/oer_audit.py
    uv run python tools/oer_audit.py --min-noul 1     # solo run con noul
    uv run python tools/oer_audit.py --json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from layaadv.metrics.core import (
    PRIMITIVES,
    iter_decisions,
    oer,
    oer_margin,
)

TAU = 0.75


def audit_file(path: Path) -> dict | None:
    """Recompute both OER readings over one run's records."""
    try:
        records = [json.loads(line) for line in path.open() if line.strip()]
    except (json.JSONDecodeError, OSError):
        return None
    if not records:
        return None

    kinds: Counter[str] = Counter(d.type for d in iter_decisions(records))
    if not kinds:
        return None
    return {
        "path": str(path),
        "n_decisions": sum(kinds.values()),
        "n_noul": kinds.get("noul", 0),
        "oer_conf": oer(records, TAU),
        "oer_margin": oer_margin(records, TAU),
        "by_primitive": {
            p: {"conf": oer(records, TAU, p), "margin": oer_margin(records, TAU, p)}
            for p in PRIMITIVES
            if kinds.get(p)
        },
        "accuracy": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results", help="directory to walk")
    parser.add_argument(
        "--min-noul",
        type=int,
        default=0,
        help="only report runs with at least this many noul decisions",
    )
    parser.add_argument("--top", type=int, default=25, help="rows to print")
    parser.add_argument("--json", action="store_true", help="dump every row as JSON")
    args = parser.parse_args()

    paths = sorted(Path(args.root).rglob("preds.jsonl"))
    if not paths:
        print(f"no preds.jsonl under {args.root}")
        return 1

    rows = [r for r in (audit_file(p) for p in paths) if r is not None]
    affected = [r for r in rows if r["n_noul"] >= max(1, args.min_noul)]

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    print(f"{len(rows)} runs with decisions, {len(affected)} with noul decisions\n")

    # Aggregate over affected runs: this is the number that says whether the pooled
    # figure was ever a single judgement or a blend of two populations.
    if affected:
        n = len(affected)
        print("pooled over the runs that contain noul decisions:")
        print(f"  runs                                  {n}")
        print(
            "  OER@0.75 on `confidence` (published) "
            f"{sum(r['oer_conf'] or 0 for r in affected) / n:.4f}"
        )
        print(
            "  OER@0.75 on margin                    "
            f"{sum(r['oer_margin'] or 0 for r in affected) / n:.4f}"
        )
        print("\nper primitive, means over the same runs (noul present):")
        print(f"  {'prim':8s} {'runs':>5s} {'conf':>8s} {'margin':>8s}")
        for p in PRIMITIVES:
            sel = [r for r in affected if p in r["by_primitive"]]
            if not sel:
                continue
            k = len(sel)
            print(
                f"  {p:8s} {k:5d} "
                f"{sum(r['by_primitive'][p]['conf'] or 0 for r in sel) / k:8.4f} "
                f"{sum(r['by_primitive'][p]['margin'] or 0 for r in sel) / k:8.4f}"
            )
        print()

    print("largest disagreement between the two readings (pooled OER@0.75):")
    ranked = sorted(
        affected,
        key=lambda r: abs((r["oer_conf"] or 0) - (r["oer_margin"] or 0)),
        reverse=True,
    )
    print(f"  {'conf':>7s} {'margin':>7s} {'noul':>6s} {'dec':>7s}  path")
    for r in ranked[: args.top]:
        print(
            f"  {r['oer_conf']:7.4f} {r['oer_margin']:7.4f} "
            f"{r['n_noul']:6d} {r['n_decisions']:7d}  {r['path']}"
        )

    clean = [r for r in rows if r["n_noul"] == 0]
    print(f"\n{len(clean)} runs carry no noul decision: there `confidence` is always the")
    print("normalised entropy and the threshold does mean one thing. They are unaffected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
