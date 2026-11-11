"""Recompute four previously unarchived Wilson limits from prediction records."""

from __future__ import annotations

import json
from pathlib import Path

from e03_semantic_audit import wilson

from layaadv import hashing


def records(experiment: str, run_id: str) -> list[dict]:
    path = Path("results") / experiment / "runs" / run_id / "preds.jsonl"
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def correct(row: dict) -> bool:
    return all(row["labels"].get(qid) == gold for qid, gold in row["gold"].items())


def main() -> None:
    e03_scope = Path(
        "results/E03_option_injection__laya-typed-decisions__massive-it__pad0__"
        "append-prepend__9a640a91b57048c9.json"
    )
    e03 = next(r for r in json.loads(e03_scope.read_text())["results"]
               if r["k"] == 32 and r["template"] == "prepend")
    e15_scope = Path("results/E15_quantization__deberta-onnx__banking77.json")
    e15 = next(r for r in json.loads(e15_scope.read_text())["results"]
               if r["k"] == 16 and r["config"] == "deberta-onnx/fp32")
    specs = [
        ("E13", "real K=8, archivio storico", "E13_label_semantics",
         "2fd9b9f346f4daed", None),
        ("E15", "DeBERTa fp32 K=16, ASR", "E15_quantization",
         e15["clean_run_id"], e15["adv_run_id"]),
        ("E03", "target prepend/head K=32, ASR", "E03_option_injection",
         e03["clean_run_id"], e03["adv_run_id"]),
    ]
    out = []
    for note, claim, experiment, clean_id, adv_id in specs:
        clean_path = Path("results") / experiment / "runs" / clean_id / "preds.jsonl"
        adv_path = (Path("results") / experiment / "runs" / adv_id / "preds.jsonl"
                    if adv_id else None)
        clean = records(experiment, clean_id)
        if adv_id is None:
            numerator = sum(map(correct, clean))
            denominator = len(clean)
        else:
            adv = {str(r["id"]): r for r in records(experiment, adv_id)}
            eligible = [r for r in clean if correct(r)]
            numerator = sum(not correct(adv[str(r["id"])]) for r in eligible)
            denominator = len(eligible)
        low, high = wilson(numerator, denominator)
        out.append({"note": note, "claim": claim, "clean_run_id": clean_id,
                    "adv_run_id": adv_id, "clean_sha256": hashing.sha256_file(clean_path),
                    "adv_sha256": hashing.sha256_file(adv_path) if adv_path else None,
                    "numerator": numerator,
                    "denominator": denominator, "point": numerator / denominator,
                    "wilson_95": [low, high]})
    path = Path("results/provenance/historical_ci_audit.json")
    path.write_text(json.dumps({"method": "Wilson 95%, z=1.96, from preds.jsonl",
                                "rows": out}, indent=2, ensure_ascii=False) + "\n")
    for row in out:
        print(row)


if __name__ == "__main__":
    main()
