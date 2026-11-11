"""Ricalcola le distribuzioni CLM Q6_K dai record E03 MASSIVE-IT."""

from __future__ import annotations

import json
import pathlib
import statistics

from layaadv.metrics.core import iter_decisions, top1_margin

ROOT = pathlib.Path(__file__).resolve().parent.parent / "results"
SUMMARY = ROOT / "E03_option_injection__clm-v0.1-8B__massive-it__pad0__append-prepend.json"
RUNS = ROOT / "E03_option_injection" / "runs"


def quantiles(values: list[float]) -> list[float]:
    ordered = sorted(values)
    return [ordered[round((len(ordered) - 1) * q)] for q in (0.1, 0.5, 0.9)]


def describe(run_id: str, k: int) -> dict:
    path = RUNS / run_id / "preds.jsonl"
    with path.open() as stream:
        decisions = list(iter_decisions(json.loads(line) for line in stream if line.strip()))
    assert len(decisions) == 2974
    p1 = [max(d.probs.values()) for d in decisions]
    margin = [top1_margin(d.probs) for d in decisions]
    wrong = [i for i, d in enumerate(decisions) if not d.correct]
    threshold = 0.75 + 0.25 / k
    # I record conservano le probabilita' a piena precisione. La confidenza CLM
    # deve soddisfare esattamente la soglia derivata dalla somma unitaria.
    assert all((d.conf >= 0.75) == (p1[i] >= threshold - 1e-12)
               for i, d in enumerate(decisions))
    return {
        "n": len(decisions), "errors": len(wrong),
        "p1_q10_q50_q90": quantiles(p1),
        "margin_q10_q50_q90": quantiles(margin),
        "wrong_p1_q10_q50_q90": quantiles([p1[i] for i in wrong]),
        "wrong_margin_q10_q50_q90": quantiles([margin[i] for i in wrong]),
        "wrong_above_p1_threshold": sum(p1[i] >= threshold for i in wrong),
        "wrong_above_margin_075": sum(margin[i] >= 0.75 for i in wrong),
        "mean_p1": statistics.fmean(p1),
    }


def main() -> None:
    payload = json.loads(SUMMARY.read_text())
    out = {"source": SUMMARY.name, "confidence_threshold_p1": {}, "rows": []}
    for row in payload["results"]:
        k = row["k"]
        out["confidence_threshold_p1"][str(k)] = 0.75 + 0.25 / k
        out["rows"].append({
            "k": k, "position": row["position"], "asr": row["asr"],
            "clean_run_id": row["clean_run_id"], "adv_run_id": row["adv_run_id"],
            "clean": describe(row["clean_run_id"], k),
            "adv": describe(row["adv_run_id"], k),
        })
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
