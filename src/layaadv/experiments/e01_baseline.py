"""E01 — Baseline pulita su typed-decisions.

Uso:
    uv run adv e01                       # i 3 checkpoint sul test set completo
    uv run adv e01 --limit 20            # prova a piccola scala
    uv run adv e01 --checkpoint ...      # ripetuto: checkpoint selezionati

Output:
    results/E01_baseline.json   metriche per checkpoint + ambiente
    results/E01_preds.jsonl     predizioni per esempio (base per ASR futuri)

Verdetto (40-esperimenti/E01-baseline.md): distanza da 0.766 <= 3pp => congelare.
Ogni checkpoint è un run separato e resumable: Ctrl-C e rilancio saltano gli
esempi già predetti.
"""

from __future__ import annotations

from layaadv import io, paths
from layaadv.data import typed_decisions as td
from layaadv.evaluate import evaluate
from layaadv.model.engine import DEFAULT_CHECKPOINT, open_from_args

# I tre checkpoint della study (10-modello/checkpoint-laya.md).
CHECKPOINTS = [
    DEFAULT_CHECKPOINT,
    "convaiinnovations/laya",
    "convaiinnovations/laya-multilingual",
]

# Granularita' di checkpoint, non troncamento: nessun chunk scarta casi, ma
# chunk e batch_size finiscono nel config perche' in bf16 una diversa
# composizione del batch puo' spostare una previsione (chunk=1 dava 0.7500
# dove chunk=32 dava 0.7469 sugli stessi 64 casi). 32 => 13 campioni di
# latenza su 400 casi: abbastanza per p50/p95, non per std (run-to-run ~3%).
CHUNK = 32


def run(args) -> dict:
    limit = getattr(args, "limit", None)
    min_agreement = getattr(args, "min_agreement", None)
    cases = td.load_cases("test", limit=limit, min_agreement=min_agreement)
    if not cases:
        raise SystemExit("nessun caso caricato: manca data/raw/typed-decisions?")

    dataset_sha = td.dataset_sha256("test")
    subset = td.subset_hash(cases)
    checkpoints = getattr(args, "checkpoint", None) or CHECKPOINTS

    print(f"[E01] {len(cases)} casi, dataset sha256={dataset_sha[:16]}...", flush=True)
    reports, all_records = [], []
    weight_sha: dict[str, str | None] = {}
    for ckpt in checkpoints:
        print(f"[E01] {ckpt} ...", flush=True)
        engine = open_from_args(args, ckpt)
        weight_sha[ckpt] = engine.weight_sha256()
        store = io.RunStore.create(
            "E01_baseline",
            {
                "experiment": "E01_baseline",
                "checkpoint": ckpt,
                "dataset": "typed-decisions/test",
                "dataset_sha256": dataset_sha,
                "subset_hash": subset,
                "n_cases": len(cases),
                "min_agreement": min_agreement,
                "weight_sha256": weight_sha[ckpt],
                "batch_size": engine.batch_size,
                "chunk": CHUNK,
                **engine.run_identity(),
            },
            resume=not getattr(args, "no_resume", False),
            force_new=getattr(args, "force_new", False),
        )
        print(f"[E01]   run {store.run_id} (riparte da {len(store.completed_ids())} già fatti)", flush=True)

        result = evaluate(
            engine,
            cases,
            store=store,
            batch_size=engine.batch_size,
            chunk=CHUNK,
            tag={"checkpoint": ckpt},
            log=lambda m: print(f"[E01] {m}", flush=True),
        )
        store.finish({"metrics": result.metrics, "skipped": result.skipped})

        reports.append(
            {"checkpoint": ckpt, "run_id": store.run_id, "skipped": result.skipped, **result.metrics}
        )
        all_records.extend(result.records)
        engine.close()

    payload = {
        "experiment": "E01_baseline",
        "dataset": "typed-decisions/test",
        "dataset_sha256": dataset_sha,
        "subset_hash": subset,
        "n_cases": len(cases),
        "min_agreement": min_agreement,
        "checkpoints": reports,
        # env_fingerprint() senza argomento restituiva null: gli hash dei pesi
        # rimanevano solo nei config dei singoli run, e nel file aggregato la
        # checklist "sha256 dei pesi" risultava non soddisfatta.
        "weight_sha256_by_checkpoint": {ckpt: sha for ckpt, sha in weight_sha.items()},
        "env": io.env_fingerprint(),
    }
    io.write_result("E01_baseline", payload)
    io.write_jsonl(paths.RESULTS / "E01_preds.jsonl", all_records)
    print(f"[E01] wrote results/E01_baseline.json ({len(all_records)} predizioni)", flush=True)
    return payload
