"""Recompute the main E16 held-out counts from saved native-forward records."""

from __future__ import annotations

import json
import random
from pathlib import Path

RESULTS = Path("results")
GLOB = "E16_calibration_defense__massive-it__*__append-tail__n2974__*.json"


def records(run_id: str) -> dict[str, dict]:
    path = RESULTS / "E16_calibration_defense" / "runs" / run_id / "preds.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    if len(rows) != len({str(r["id"]) for r in rows}):
        raise ValueError(f"ID duplicati in {run_id}")
    return {str(r["id"]): r for r in rows}


def decision(row: dict) -> tuple[bool, float]:
    qid, gold = next(iter(row["gold"].items()))
    return row["labels"][qid] == gold, float(row["confidence"][qid])


def check(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("scope_identity", {}).get("analysis_version") != "native-deploy-v2":
        return
    runs = {kind: records(run_id) for kind, run_id in payload["run_ids"].items()}
    ids = set(runs["clean"])
    if len(ids) != 2974 or any(set(arm) != ids for arm in runs.values()):
        raise ValueError(f"bracci non appaiati: {path}")
    ordered = sorted(ids)
    random.Random(payload["axes"]["split_seed"]).shuffle(ordered)
    held = ordered[payload["axes"]["n_fit"]:]
    before_ok = [case_id for case_id in held if decision(runs["clean"][case_id])[0]]
    after_ok = [case_id for case_id in held if decision(runs["deployed_clean"][case_id])[0]]
    if before_ok != after_ok:
        raise ValueError(f"temperatura ha cambiato label pulite: {path}")
    flipped = sum(not decision(runs["adversarial"][case_id])[0] for case_id in before_ok)
    flipped_after = sum(not decision(runs["deployed_adversarial"][case_id])[0]
                        for case_id in after_ok)
    if flipped != flipped_after:
        raise ValueError(f"temperatura ha cambiato ASR: {path}")
    oer = [sum(not decision(runs[arm][case_id])[0]
               and decision(runs[arm][case_id])[1] >= 0.75 for case_id in held)
           for arm in ("adversarial", "deployed_adversarial")]
    blocked = [sum(decision(runs[arm][case_id])[1] >= 0.75 for case_id in before_ok)
               for arm in ("clean", "deployed_clean")]
    verdict = payload["verdict"]
    assert payload["asr"]["denominator"] == len(before_ok)
    assert abs(payload["asr"]["before"] - flipped / len(before_ok)) < 1e-12
    assert all(abs(rate - count / len(held)) < 1e-12 for rate, count in zip(
        (verdict["oer_adv_shipped"], verdict["oer_adv_recalibrated"]), oer))
    guardrail = payload["guardrail_at_0.75"]["clean"]
    assert [guardrail[k]["blocked_correct"] for k in ("shipped", "recalibrated")] == blocked
    for replica in payload["replicas"]["replicas"]:
        replica_ids = sorted(ids)
        random.Random(replica["seed"]).shuffle(replica_ids)
        replica_held = replica_ids[replica["n_fit"]:]
        deployed_clean = records(replica["deployed_clean_run_id"])
        deployed_adv = records(replica["deployed_adv_run_id"])
        replica_ok = [case_id for case_id in replica_held
                      if decision(runs["clean"][case_id])[0]]
        replica_oer_count = sum(
            not decision(deployed_adv[case_id])[0]
            and decision(deployed_adv[case_id])[1] >= 0.75
            for case_id in replica_held
        )
        replica_blocked = sum(decision(deployed_clean[case_id])[1] >= 0.75
                              for case_id in replica_ok)
        assert replica["n_heldout"] == len(replica_held)
        assert replica["n_correct_clean"] == len(replica_ok)
        assert abs(replica["oer_adv_after"] - replica_oer_count / len(replica_held)) < 1e-12
        assert replica["blocked_correct_after"] == replica_blocked
    print(path.name, f"ASR {flipped}/{len(before_ok)}",
          f"OER {oer[0]}/{len(held)} -> {oer[1]}/{len(held)}",
          f"falsi blocchi {blocked[0]}/{len(before_ok)} -> {blocked[1]}/{len(before_ok)}")


def main() -> None:
    paths = sorted(RESULTS.glob(GLOB))
    native = [path for path in paths if json.loads(path.read_text()).get(
        "scope_identity", {}).get("analysis_version") == "native-deploy-v2"]
    if len(native) != 2:
        raise ValueError(f"attesi due checkpoint E16 correnti, trovati {len(native)}")
    for path in native:
        check(path)


if __name__ == "__main__":
    main()
