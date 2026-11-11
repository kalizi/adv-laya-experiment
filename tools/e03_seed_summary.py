"""Summarize paired K=16 append/tail E03 reruns across option seeds."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from e03_semantic_audit import wilson

from layaadv import hashing
from layaadv.metrics.core import asr

RESULTS = Path("results")
TARGET = "convaiinnovations/laya-typed-decisions"
BASELINE = RESULTS / (
    "E03_option_injection__laya-typed-decisions__massive-it__pad0__"
    "append-prepend__9a640a91b57048c9.json"
)


def preds(run_id: str) -> list[dict]:
    path = RESULTS / "E03_option_injection" / "runs" / run_id / "preds.jsonl"
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def main() -> None:
    scopes: dict[int, tuple[Path, dict]] = {}
    for path in RESULTS.glob("E03_option_injection__laya-typed-decisions__massive-it__pad0__*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        ident = payload.get("scope_identity") or {}
        if ident.get("checkpoint") != TARGET or ident.get("k_values") not in ([16], [4, 8, 16, 24, 32, 40]):
            continue
        seed = ident.get("seed")
        if seed not in (0, 1, 2, 3):
            continue
        if seed == 0 and path != BASELINE:
            continue
        row = next((r for r in payload["results"] if r["k"] == 16
                    and r["template"] == "append" and r["position"] == "tail"), None)
        if row:
            scopes[seed] = path, row
    if set(scopes) != {0, 1, 2, 3}:
        raise ValueError(f"mancano i seed 0..3: trovati {sorted(scopes)}")
    rows = []
    for seed, (path, row) in sorted(scopes.items()):
        clean, adv = preds(row["clean_run_id"]), preds(row["adv_run_id"])
        rate, denominator = asr(clean, adv)
        flipped = round(rate * denominator)
        if denominator != row["asr_denominator"] or abs(rate - row["asr"]) > 1e-12:
            raise ValueError(f"riepilogo e record divergono per seed {seed}")
        rows.append({"seed": seed, "scope": path.name,
                     "scope_sha256": hashing.sha256_file(path),
                     "clean_run_id": row["clean_run_id"], "adv_run_id": row["adv_run_id"],
                     "n_cases": len(clean), "correct_clean": denominator,
                     "flipped": flipped, "asr": rate,
                     "wilson_95": wilson(flipped, denominator)})
    rates = [r["asr"] for r in rows]
    output = {"design": "same utterance set, K=16, append/tail; option sets vary with seed",
              "rows": rows, "asr_mean": statistics.mean(rates),
              "asr_sample_sd": statistics.stdev(rates), "asr_range": [min(rates), max(rates)],
              "interpretation": (
                  "Wilson intervals are conditional per seed; seed spread is descriptive, "
                  "not independent replication."
              )}
    path = RESULTS / "provenance/e03_seed_k16_summary.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)
    for row in rows:
        print(row["seed"], row["flipped"], "/", row["correct_clean"], row["asr"])


if __name__ == "__main__":
    main()
