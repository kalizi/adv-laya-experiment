"""E02 — Option count collapse: accuracy vs K choice options.

Diagnostic (not a study submission): measures how accuracy degrades as
choice options grow, due to token budget pressure on head (question + criteria
+ options). Study-valid variant: use a dataset with K >= 30 options.

Uso:
    uv run adv e02 --limit 20

Output: results/E02_option_count.json
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone

from layaadv import io
from layaadv.data import typed_decisions as td
from layaadv.evaluate import evaluate
from layaadv.metrics.core import asr
from layaadv.model.engine import DEFAULT_CHECKPOINT, open_from_args
from layaadv.schema import Case

K_VALUES = [2, 4, 8, 16, 32]

# Granularita' di checkpoint: quante volte si scrive su disco, non quanti casi si
# scartano. Va nel config perche' in bf16 una diversa composizione del batch può
# spostare una previsione, quindi due run con chunk diverso non devono condividere
# il run_id. 32 => 13 campioni di latenza su 400 casi.
CHUNK = 32

DISTRACTOR_POOL = [
    ("defer", "postpone the decision"),
    ("delegate", "assign to another team"),
    ("archive", "file for later review"),
    ("flag", "mark for follow-up"),
    ("review", "re-examine the case"),
    ("hold", "keep pending"),
    ("resolve", "close with a fix"),
    ("close", "mark as done"),
    ("open", "start a new case"),
    ("cancel", "discard the request"),
    ("merge", "combine with another case"),
    ("split", "break into sub-tasks"),
    ("ignore", "no action needed"),
    ("reply", "send a short acknowledgement"),
    ("escalate", "hand off to a human"),
]


def expand_choice_question(question: dict, gold_label: str, k: int) -> dict:
    """Expand a choice question to exactly K options, keeping gold."""
    if question["type"] != "choice":
        return question
    criteria = question.get("criteria")
    if not isinstance(criteria, dict):
        return question

    new_criteria: dict = {}
    if gold_label in criteria:
        new_criteria[gold_label] = criteria[gold_label]

    for key, val in criteria.items():
        if len(new_criteria) >= k:
            break
        if key != gold_label:
            new_criteria[key] = val

    existing_labels = set(new_criteria.keys())
    for label, desc in DISTRACTOR_POOL:
        if len(new_criteria) >= k:
            break
        if label not in existing_labels:
            new_criteria[label] = desc

    i = 0
    while len(new_criteria) < k:
        key = f"alt_{i}"
        if key not in new_criteria:
            new_criteria[key] = f"alternative action {i}"
        i += 1

    return {**question, "criteria": new_criteria}


def _git_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _metadata(checkpoint: str) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_hash(),
        "checkpoint": checkpoint,
    }


def run(args) -> dict:
    limit = getattr(args, "limit", None)
    cases = td.load_cases("test", limit=limit)
    if not cases:
        raise SystemExit("nessun caso caricato: manca data/raw/typed-decisions?")
    dataset_sha, subset = td.dataset_sha256("test"), td.subset_hash(cases)

    checkpoint = getattr(args, "checkpoint", None) or [DEFAULT_CHECKPOINT]
    checkpoint = checkpoint[0] if isinstance(checkpoint, list) else checkpoint

    print(f"[E02] {len(cases)} casi, expanding choice questions to K options", flush=True)
    engine = open_from_args(args, checkpoint)

    results = []
    clean_records = None
    for k in K_VALUES:
        print(f"[E02] K={k} ...", flush=True)
        expanded_cases = []
        for case in cases:
            new_questions = {}
            for qid, q in case.questions.items():
                gold_label = case.gold.get(qid, "")
                new_questions[qid] = expand_choice_question(q, gold_label, k)
            expanded_cases.append(
                Case(id=case.id, state=case.state, questions=new_questions, gold=case.gold)
            )

        store = io.RunStore.create(
            "E02_option_count",
            {
                "experiment": "E02_option_count",
                "checkpoint": checkpoint,
                "k": k,
                "n_cases": len(cases),
                "dataset": "typed-decisions/test",
                "dataset_sha256": dataset_sha,
                "subset_hash": subset,
                # batch_size e chunk nel config: senza, due run con parametri
                # diversi condividerebbero il run_id e in bf16 la composizione
                # del batch puo' spostare una previsione.
                "batch_size": engine.batch_size,
                "chunk": CHUNK,
                **engine.run_identity(),
                "weight_sha256": engine.weight_sha256(),
            },
            resume=not getattr(args, "no_resume", False),
            force_new=getattr(args, "force_new", False),
        )

        result = evaluate(
            engine,
            expanded_cases,
            store=store,
            batch_size=engine.batch_size,
            chunk=CHUNK,
            log=lambda m: print(f"[E02] {m}", flush=True),
        )
        store.finish({"metrics": result.metrics})

        row = {"k": k, "run_id": store.run_id, **result.metrics}

        if k == K_VALUES[0]:
            clean_records = result.records
        elif clean_records is not None:
            asr_val, asr_denom = asr(clean_records, result.records)
            row["asr"] = asr_val
            row["asr_denominator"] = asr_denom

        results.append(row)
        print(f"[E02] K={k}: acc={result.metrics.get('accuracy')}", flush=True)

    payload = {
        "experiment": "E02_option_count",
        "checkpoint": checkpoint,
        "k_values": K_VALUES,
        "metadata": _metadata(checkpoint),
        "results": results,
        "env": io.env_fingerprint(engine.weight_sha256()),
    }
    io.write_result("E02_option_count", payload)
    print("[E02] wrote results/E02_option_count.json", flush=True)
    engine.close()
    return payload
