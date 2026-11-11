"""E04 — Truncation attack: push signal past token budget.

Study-valid (modifies only state): prepend neutral filler to push the
gold-determining signal past the state budget, causing right-tail truncation.

Uso:
    uv run adv e04 --limit 5

Output: results/E04_truncation.json
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone

from layaadv import io
from layaadv.data import typed_decisions as td
from layaadv.evaluate import evaluate
from layaadv.metrics.core import asr
from layaadv.model.backends import DecisionBackend
from layaadv.model.engine import DEFAULT_CHECKPOINT, open_from_args
from layaadv.schema import Case

FILLER = (
    "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua ut enim ad minim veniam "
    "quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo "
    "consequat duis aute irure dolor in reprehenderit in voluptate velit esse "
    "cillum dolore eu fugiat nulla pariatur excepteur sint occaecat cupidatat "
    "non proident sunt in culpa qui officia deserunt mollit anim id est laborum "
    "sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium "
    "doloremque laudantium totam rem aperiam eaque ipsa quae ab illo inventore "
    "veritatis et quasi architecto beatae vitae dicta sunt explicabo nemo enim "
    "ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit sed quia "
    "consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt neque "
    "porro quisquam est qui dolorem ipsum quia dolor sit amet consectetur "
    "adipisci velit sed quia non numquam eius modi tempora incidunt ut labore "
)

FILLER_MULTIPLIERS = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5]

MARKER = "ZZQMARKERE04TAIL9997"

# Granularita' di checkpoint. Sta nel config perche' deve essere quello che
# `evaluate` usa davvero: il config diceva `chunk = batch_size` mentre la run ne
# faceva uno solo (una forward pass su 400 casi), quindi il p50 e il p95 erano lo
# stesso numero e `std` era 0 per costruzione, non perche' il run fosse silenzioso.
CHUNK = 32


def push_out_state(state: str, filler_tokens: int) -> str:
    """Prepend filler to push state past budget."""
    if filler_tokens <= 0:
        return state
    words = FILLER.split()
    repeats = (filler_tokens // len(words)) + 1
    filler = " ".join((words * repeats)[:filler_tokens])
    return filler + " " + state


def _validate_truncation(engine: DecisionBackend, state: str, questions: dict, filler_tokens: int) -> dict:
    """Verify that the marker at the tail is actually truncated."""
    if filler_tokens <= 0:
        return {"truncation_verified": True, "reason": "no filler"}
    probe_state = push_out_state(state, filler_tokens) + " " + MARKER
    rows = engine.probe_sequence(probe_state, questions, marker=MARKER)
    return {
        "truncation_verified": all(not r["marker_visible"] for r in rows),
        "marker_visible_count": sum(1 for r in rows if r["marker_visible"]),
        "total_questions": len(rows),
    }


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

    print(f"[E04] {len(cases)} casi, truncation attack", flush=True)
    engine = open_from_args(args, checkpoint)

    all_budgets: dict[str, int] = {}
    for case in cases:
        for qid, budget in engine.state_budget(case.questions).items():
            all_budgets[qid] = max(all_budgets.get(qid, 0), budget)
    avg_budget = sum(all_budgets.values()) / len(all_budgets)
    print(f"[E04] state budget: {avg_budget:.0f} tokens (from {len(all_budgets)} questions)", flush=True)

    results = []
    clean_records = None
    for mult in FILLER_MULTIPLIERS:
        filler_tokens = int(avg_budget * mult)
        print(f"[E04] filler={mult}x ({filler_tokens} tokens) ...", flush=True)

        modified_cases = []
        truncation_info = None
        for case in cases:
            new_state = push_out_state(case.state, filler_tokens)
            if mult > 0 and truncation_info is None:
                truncation_info = _validate_truncation(
                    engine, case.state, case.questions, filler_tokens
                )
            modified_cases.append(
                Case(id=case.id, state=new_state, questions=case.questions, gold=case.gold)
            )

        store = io.RunStore.create(
            "E04_truncation",
            {
                "experiment": "E04_truncation",
                "checkpoint": checkpoint,
                "filler_mult": mult,
                "filler_tokens": filler_tokens,
                "n_cases": len(cases),
                "dataset": "typed-decisions/test",
                "dataset_sha256": dataset_sha,
                "subset_hash": subset,
                # batch_size and chunk belong in the config: without them a run
                # made with a different batch would share a run_id, and in bf16
                # the batch composition can move a prediction.
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
            modified_cases,
            store=store,
            batch_size=engine.batch_size,
            chunk=CHUNK,
            log=lambda m: print(f"[E04] {m}", flush=True),
        )
        store.finish({"metrics": result.metrics})

        row = {
            "filler_mult": mult,
            "filler_tokens": filler_tokens,
            "run_id": store.run_id,
            **result.metrics,
        }
        if truncation_info:
            row["truncation_check"] = truncation_info

        if mult == 0.0:
            clean_records = result.records
        elif clean_records is not None:
            asr_val, asr_denom = asr(clean_records, result.records)
            row["asr"] = asr_val
            row["asr_denominator"] = asr_denom

        results.append(row)
        print(f"[E04] filler={mult}x: acc={result.metrics.get('accuracy')}", flush=True)

    payload = {
        "experiment": "E04_truncation",
        "checkpoint": checkpoint,
        "state_budget": avg_budget,
        "metadata": _metadata(checkpoint),
        "results": results,
        "env": io.env_fingerprint(engine.weight_sha256()),
    }
    io.write_result("E04_truncation", payload)
    print("[E04] wrote results/E04_truncation.json", flush=True)
    engine.close()
    return payload
