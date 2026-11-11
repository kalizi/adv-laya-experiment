"""Prepare a prediction-blind E03 sample, then score completed annotations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

from layaadv import hashing
from layaadv.data import loaders
from layaadv.experiments.e03_option_injection import inject_option, make_state, template_text
from layaadv.metrics.core import asr

OUT = Path("results/semantic_audit")
SAMPLE = OUT / "e03_k16_blind_sample.json"
ANNOTATIONS = OUT / "e03_k16_annotations.csv"
REPORT = OUT / "e03_k16_report.json"
LABELS = {"invariato", "ambiguo", "cambiato"}


def write_new_or_same(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != contents:
        raise FileExistsError(f"file già fissato con contenuto diverso: {path}")
    path.write_text(contents, encoding="utf-8")


def prepare() -> None:
    cases, ds, _ = loaders.load_cases("massive-it", k=16, seed=0)
    if len(cases) != 2974:
        raise ValueError(f"attesi 2974 casi, trovati {len(cases)}")
    chosen = random.Random(20261006).sample(sorted(cases, key=lambda c: c.id), 60)
    rows = []
    for case in sorted(chosen, key=lambda c: c.id):
        q = case.questions[loaders.QID]
        gold = case.gold[loaders.QID]
        target = inject_option(q, gold)
        rows.append({
            "id": case.id, "original": case.state, "gold": gold, "target": target,
            "options": list(q["criteria"]),
            "prepend_head": make_state(template_text("prepend"), target,
                                       position="head")(case.state),
            "append_tail": make_state(template_text("append"), target,
                                      position="tail")(case.state),
        })
    payload = {
        "protocol": "50-metodologia/e03-audit-semantico.md",
        "dataset": "massive-it", "split": "test", "k": 16, "option_seed": 0,
        "sample_seed": 20261006, "n": len(rows),
        "dataset_sha256": ds.file_sha256(), "subset_hash": ds.subset_hash(cases),
        "rows": rows,
    }
    write_new_or_same(SAMPLE, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"{SAMPLE}: {len(rows)} casi senza predizioni")


def wilson(successes: int, n: int) -> list[float] | None:
    if n == 0:
        return None
    p, z = successes / n, 1.96
    divisor = 1 + z * z / n
    center = (p + z * z / (2 * n)) / divisor
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / divisor
    return [center - half, center + half]


def read_preds(run_id: str) -> dict[str, dict]:
    path = Path("results/E03_option_injection/runs") / run_id / "preds.jsonl"
    return {str(row["id"]): row for row in (json.loads(line) for line in path.open())}


def report(scope_path: Path) -> None:
    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    with ANNOTATIONS.open(encoding="utf-8", newline="") as file:
        annotations = list(csv.DictReader(file))
    expected = {row["id"] for row in sample["rows"]}
    if len(annotations) != len(expected) or {row["id"] for row in annotations} != expected:
        raise ValueError("annotazioni incomplete, duplicate o fuori campione")
    for row in annotations:
        for arm in ("prepend", "append"):
            if row.get(arm) not in LABELS or not row.get("reason_" + arm, "").strip():
                raise ValueError(f"classe o motivazione mancante per {row['id']} {arm}")
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    summary = {}
    for arm in ("prepend", "append"):
        row = next(r for r in scope["results"] if r["k"] == 16 and r["template"] == arm)
        clean, adv = read_preds(row["clean_run_id"]), read_preds(row["adv_run_id"])
        valid = {item["id"] for item in annotations if item[arm] == "invariato"}
        paired_clean = [clean[id_] for id_ in sorted(valid)]
        paired_adv = [adv[id_] for id_ in sorted(valid)]
        rate, denominator = asr(paired_clean, paired_adv)
        successes = round(rate * denominator) if rate is not None else 0
        summary[arm] = {
            "classes": {label: sum(item[arm] == label for item in annotations)
                        for label in sorted(LABELS)},
            "valid_clean_correct": denominator,
            "flipped": successes, "asr_valid": rate,
            "wilson_95": wilson(successes, denominator),
            "clean_run_id": row["clean_run_id"], "adv_run_id": row["adv_run_id"],
        }
    payload = {
        "sample_sha256": hashing.sha256_file(SAMPLE),
        "annotations_sha256": hashing.sha256_file(ANNOTATIONS),
        "scope_file": scope_path.name, "scope_sha256": hashing.sha256_file(scope_path),
        "n_sampled": len(annotations), "results": summary,
    }
    write_new_or_same(REPORT, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"{REPORT}: {summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "report"])
    parser.add_argument("--scope", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        if args.scope is None:
            parser.error("report richiede --scope")
        report(args.scope)


if __name__ == "__main__":
    main()
