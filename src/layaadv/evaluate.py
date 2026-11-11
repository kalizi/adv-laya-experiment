"""The shared evaluation loop: batched predict + checkpoint + metrics.

Every experiment runs through :func:`evaluate` — baseline (E01), attacks
(E03/E12) and any future sweep. Nothing else may call ``engine.predict_batch``
directly in a loop, or resume/batching semantics will drift apart.

Work is chunked; each finished chunk is appended to the :class:`RunStore`, so
an interrupted run resumes from the last complete chunk (at most one chunk of
work is lost).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from layaadv.io import RunStore
from layaadv.metrics import core as metrics
from layaadv.model.engine import LayaEngine
from layaadv.schema import Case, Prediction, answer_probs


@dataclass
class EvalResult:
    """Scored run: records in case order + the standard metric block."""

    records: list[dict]
    metrics: dict
    skipped: int  # records reused from a previous (interrupted) run


def sync_device() -> None:
    """Block until queued GPU work is done.

    CUDA kernels are async: without this the timer would measure kernel launch
    instead of inference, understating latency by an order of magnitude. No-op
    without torch, so the metrics module stays importable in a torch-free env.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def record_for(case: Case, pred: Prediction, tag: dict | None = None) -> dict:
    """One checkpointable row: predictions + gold + confidence + probabilities."""
    rec: dict[str, Any] = {
        "id": case.id,
        "types": {qid: q["type"] for qid, q in case.questions.items()},
        "labels": pred.labels,
        "confidence": {qid: float(a.get("confidence", 0.0)) for qid, a in pred.answers.items()},
        "probs": {qid: answer_probs(a) for qid, a in pred.answers.items()},
        "gold": case.gold,
        "usage": pred.usage,
    }
    if tag:
        rec.update(tag)
    return rec


def evaluate(
    engine: LayaEngine,
    cases: Sequence[Case],
    *,
    store: RunStore | None = None,
    batch_size: int | None = None,
    chunk: int | None = None,
    tag: dict | None = None,
    log: Callable[[str] | None] | None = None,
    collect: Callable[[Sequence[Case], Sequence[Prediction]], Sequence[dict] | None] | None = None,
) -> EvalResult:
    """Score ``cases`` (resuming from ``store`` when present).

    ``tag`` is merged into every record (e.g. ``{"checkpoint": name}``) so flat
    prediction dumps stay self-describing when checkpoints are concatenated.

    ``chunk`` is only the **checkpoint granularity**, i.e. how often the run is
    flushed to disk; it defaults to "one chunk = the whole run". It never drops
    cases, but it does change how many forward passes the engine is asked for,
    and in bf16 a different batch composition can move a prediction: chunk=1
    scored 0.7500 where chunk=32 scored 0.7469 on the same 64 cases. Callers
    that care must pin it (E01 records it in the run config).

    ``collect`` is the hook for a measurement that needs the same cases as the
    prediction but a different forward (E22 reads the attention the model pays to
    the injected sentence). It is called once per chunk with that chunk's cases
    and predictions and returns one dict per case, merged into the record, so the
    extra measurement keeps the loop's chunking, resume and checkpointing instead
    of growing a second loop in the experiment. Its records are checkpointed with
    the rest, so a resumed run does not pay for them twice and a partial run
    stays internally consistent.

    Real batching inside laya is governed by ``batch_size``, which is what the
    latency samples measure. A single chunk therefore yields one latency sample
    and no meaningful p95/std: use a smaller chunk when the percentiles matter.

    Adds a ``latency`` block (per forward pass, GPU-synchronised). Samples go to
    the store's sidecar, so percentiles survive a resume.
    """
    stored: dict[str, dict] = {}
    if store is not None:
        stored = {str(r["id"]): r for r in store.read_records()}

    # I record sono raccolti per `id`, quindi un id duplicato nel dataset farebbe
    # contare due volte la stessa predizione: `ordered` ripeterebbe la riga e le
    # metriche la sommerebbero due volte. Meglio fermarsi che produrre un numero
    # che non si puo' riprodurre.
    seen_ids: set[str] = set()
    duplicates = sorted({c.id for c in cases if c.id in seen_ids or seen_ids.add(c.id)})
    if duplicates:
        raise ValueError(
            f"{len(duplicates)} id duplicati nei casi da valutare (es. {duplicates[:5]}): "
            "i record vengono raccolti per id, quindi un id ripetuto fa contare la stessa "
            "predizione due volte"
        )

    done: dict[str, dict] = {}
    todo: list[Case] = []
    for case in cases:
        if case.id in stored:
            done[case.id] = stored[case.id]
        else:
            todo.append(case)

    sync_device()  # exclude any earlier GPU work from the first sample
    t_run0 = time.perf_counter()
    fresh_ms: list[float] = []
    fresh_decisions: list[int] = []
    step = chunk or len(todo) or 1
    for start in range(0, len(todo), step):
        part = todo[start : start + step]
        sync_device()
        t0 = time.perf_counter()
        preds = engine.predict_batch(part, batch_size=batch_size)
        sync_device()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if len(preds) != len(part):
            raise ValueError(
                f"il backend ha restituito {len(preds)} predizioni per {len(part)} casi"
            )
        for case, pred in zip(part, preds):
            if pred.id != case.id:
                raise ValueError(
                    f"predizioni fuori ordine: atteso id {case.id!r}, ricevuto {pred.id!r}"
                )
        rows = [record_for(c, p, tag) for c, p in zip(part, preds)]
        if collect is not None:
            extra = collect(part, preds)
            if extra is not None:
                if len(extra) != len(rows):
                    raise ValueError(
                        f"il collettore ha restituito {len(extra)} righe per {len(rows)} casi"
                    )
                # Una riga vuota e' lecita: un caso senza misura viene conservato con la
                # sola predizione, invece di far cadere il chunk intero.
                for row, one in zip(rows, extra):
                    if one:
                        row.update(one)
        # The timed unit is the decision, not the case: a case carries as many
        # questions as the dataset gives it, and that number differs per dataset.
        timed = sum(len(r["labels"]) for r in rows)
        where = engine.runtime_state()
        if store is not None:
            store.append(rows)
            store.append_latency(elapsed_ms, timed, **where)
        else:
            fresh_ms.append(elapsed_ms)
            fresh_decisions.append(timed)
        done.update({r["id"]: r for r in rows})
        if log is not None:
            log(f"    {len(done)}/{len(cases)} decisions scored")

    ordered = [done[c.id] for c in cases]
    block = metrics.aggregate(ordered)
    # Come e' andato davvero il batching, contro come e' stato chiesto. Fuori dal
    # percorso cronometrato e fuori dal `run_id`: la verita' su una run vale anche
    # quando non cambia quale run e', e metterla nell'identita' invaliderebbe ogni
    # resume esistente senza guadagnare in riproducibilita'.
    # `getattr` e non un metodo richiesto: un backend che non raggruppa (o un finto
    # nei test) non deve poter rompere il loop di misura per un campo diagnostico.
    # Sempre presente, anche a resume completo: un run ripreso non ha `todo`, e un campo che
    # esiste solo a meta' run rende il payload non confrontabile fra una run e l'altra.
    reporter = getattr(engine, "batching_report", None)
    if reporter is not None:
        report = reporter(todo if todo else list(cases))
        report["batch_size_requested"] = batch_size or getattr(engine, "batch_size", None)
        block["batching"] = report
    # Percentiles replay the whole run from the sidecar, so a resumed run is
    # not silently reported over its final segment only. Without a store the
    # only samples available are this process's own.
    samples = store.read_latency_samples() if store is not None else [
        {"ms": m, "decisions": d} for m, d in zip(fresh_ms, fresh_decisions)
    ]
    block["latency"] = metrics.latency_stats([float(s["ms"]) for s in samples])
    # Where the model actually was, per sample. More than one device in a single run
    # is the OOM demotion, and it invalidates any speed comparison of that run.
    block["latency"]["devices"] = sorted({str(s["device"]) for s in samples if s.get("device")})
    timed_decisions = sum(int(s.get("decisions", 0)) for s in samples)
    if not timed_decisions:  # sidecar scritto prima del conteggio per batch
        timed_decisions = block["n_decisions"]
    block["throughput"] = metrics.throughput(block["latency"], timed_decisions)
    block["throughput_basis"] = "decisioni dentro i batch cronometrati"
    block["runtime_s"] = {
        "this_process": round(time.perf_counter() - t_run0, 2),
        "batches_this_process": len(fresh_ms) if store is None else -(-len(todo) // step),
    }
    return EvalResult(records=ordered, metrics=block, skipped=len(stored))
