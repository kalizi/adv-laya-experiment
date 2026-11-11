"""Attack and calibration metrics — pure functions over scored records.

Definitions are normative here; prose lives in
50-metodologia/metriche-attacco.md and must not drift from this code.

Unit of account is the **decision** (one case x one question): typed-decisions
test = 400 cases x 5 questions = 2000 decisions. ASR is computed only over
decisions that were *correct* on the clean run, otherwise an attack would
"win" on pre-existing errors.

Input record (produced by ``layaadv.evaluate``)::

    {"id", "types": {qid: type}, "labels": {qid: pred},
     "confidence": {qid: float}, "probs": {qid: {label: p}},
     "gold": {qid: label}, ...}
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

PRIMITIVES = ("choice", "noul", "score")


@dataclass(frozen=True)
class Decision:
    """One (case, question) pair, scored."""

    id: str
    qid: str
    type: str
    pred: str
    gold: str
    conf: float
    probs: dict[str, float]

    @property
    def correct(self) -> bool:
        return self.pred == self.gold


def iter_decisions(records: Iterable[dict], primitive: str | None = None) -> Iterator[Decision]:
    for rec in records:
        for qid, pred in rec["labels"].items():
            qtype = rec.get("types", {}).get(qid, "")
            if primitive is not None and qtype != primitive:
                continue
            yield Decision(
                id=str(rec["id"]),
                qid=qid,
                type=qtype,
                pred=str(pred),
                gold=str(rec["gold"][qid]),
                conf=float(rec.get("confidence", {}).get(qid, 0.0)),
                probs=dict(rec.get("probs", {}).get(qid, {})),
            )


def accuracy(records: Iterable[dict], primitive: str | None = None) -> float | None:
    """argmax accuracy over decisions (per primitive when requested)."""
    decs = list(iter_decisions(records, primitive))
    if not decs:
        return None
    return sum(d.correct for d in decs) / len(decs)


def ece(records: Iterable[dict], bins: int = 10) -> float | None:
    """Expected Calibration Error, top-label, equal-width bins.

    Confidence of the *predicted* class vs empirical accuracy per bin.

    The bin index is clamped: a confidence outside [0, 1] would otherwise index
    the bucket list **from the end** (a negative index wraps), silently putting a
    decision in the wrong bin and producing a number that looks fine. Clamping is
    a floor under a bad input, not a fix for it: the caller that produced the
    confidence is the bug.
    """
    decs = list(iter_decisions(records))
    if not decs:
        return None
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for d in decs:
        idx = min(bins - 1, max(0, int(d.conf * bins)))
        buckets[idx].append((d.conf, d.correct))
    total = len(decs)
    return sum(
        len(b) / total * abs((sum(c for _, c in b) / len(b)) - (sum(c for c, _ in b) / len(b)))
        for b in buckets
        if b
    )


def brier(records: Iterable[dict], primitive: str | None = None) -> float | None:
    """Mean multiclass Brier score: mean of sum((p - onehot_gold)^2)."""
    scores = []
    for d in iter_decisions(records, primitive):
        labels = set(d.probs) | {d.gold}
        scores.append(sum((d.probs.get(lab, 0.0) - (1.0 if lab == d.gold else 0.0)) ** 2 for lab in labels))
    if not scores:
        return None
    return statistics.fmean(scores)


def oer(records: Iterable[dict], tau: float = 0.75, primitive: str | None = None) -> float | None:
    """Overconfident Error Rate: wrong AND confidence >= tau (the 'fischi per
    fiaschi' signature). tau=0.75 primary, 0.85 secondary.

    **Non derivare da questo numero la qualita' del gate.** Sotto attacco il tasso di
    errore puo' essere del 98%, e allora "la decisione bloccata era un errore" e' vero a
    caso quasi sempre: la precisione del gate resta entro 1.0-1.11x della base su tutto
    l'intervallo delle soglie, **anche dove la difesa funziona** (misurato in E16). Il
    numero che si puo' spendere e' il **costo sul traffico pulito** (quante risposte buone
    blocchi), e va letto insieme al beneficio su quello attaccato: vedi
    `e16_calibration_defense.trade_off`, che tiene i due rami separati per questo.

    **La soglia non e' confrontabile fra primitive.** `record["confidence"]` non e' una
    grandezza sola: su `k = 2` (noul) i backend ci mettono `max(p, 1 - p)`, su `k > 2`
    l'entropia normalizzata `1 - H(p)/log(k)`. Le due non coincidono: a parita' di
    distribuzione la prima vale 0.726 e la seconda 0.152. Quindi `tau = 0.75` su un `noul`
    chiede una majority, e su un `choice` chiede una distribuzione quasi smussata, e il
    confronto fra le primitive non regge. Misurato sul run pulito di E01 (2000 decisioni,
    stesso modello, stesso checkpoint): `confidence >= 0.75` scatta su **0.2917** degli
    `noul` e su **0.0000** dei `choice`. Per questo `aggregate` pubblica anche
    `oer_by_primitive`, e per questo esiste `oer_margin`.
    """
    decs = list(iter_decisions(records, primitive))
    if not decs:
        return None
    return sum(1 for d in decs if not d.correct and d.conf >= tau) / len(decs)


def top1_margin(probs: dict[str, float]) -> float:
    """top1 - top2: how far the winning option is from the runner-up.

    Una sola definizione per ogni primitiva e ogni backend, e derivata da `probs`, che
    ogni record porta gia' ed e' confrontabile per costruzione. `record["confidence"]`
    invece cambia significato con `k` e con il backend (vedi `oer`), quindi un OER letto
    da due righe prodotte da backend diversi non e' la stessa quantita'.

    Con `k = 1` non esiste il secondo candidato e il margine e' 0.0: la decisione non e'
    distinguibile da nessun'altra, non "sicura". Con un solo candidato la distinzione non
    e' nemmeno ponibile, e dichiararlo e' il punto.
    """
    values = sorted((float(p) for p in probs.values()), reverse=True)
    if len(values) < 2:
        return 0.0
    return values[0] - values[1]


def margin_at_cost(records: Iterable[dict], budget: float) -> dict:
    """Pick a margin threshold from the **clean** run by its cost, not by its number.

    Returns the smallest ``tau`` whose false-block rate on clean traffic stays
    within ``budget``, plus the rate it actually achieved and the confidence at
    which that rate is not an artefact of one or two decisions.

    Why not a constant. ``logit_scale`` in CLM is 100.81 clamped to 100.0, i.e.
    temperature 0.01: a cosine gap of 0.05 already gives 5 nats and a top
    probability of 0.994, and 0.10 saturates it to 1.0000. On a distribution
    like that a fixed threshold on a probability cannot discriminate, and a
    fixed threshold on the margin is just as arbitrary -- it would be a number
    fitted to laya's margin distribution and applied to CLM's. E16 already
    settled the currency: **the cost on clean traffic**, how many good answers a
    gate blocks. So the threshold is derived here, on the clean run, and frozen
    before anything adversarial is scored.

    ``budget`` is a fraction of *correct* clean decisions, which is the
    population a gate actually acts on: blocking an already-wrong answer costs
    the user nothing.
    """
    decs = [d for d in iter_decisions(records) if d.correct]
    if not decs:
        raise ValueError(
            "nessuna decisione corretta al pulito: non c'e' traffico su cui calibrare "
            "una soglia. Il denominatore vuoto e' un modello che non risolve il "
            "compito, non uno zero da interpretare."
        )
    margins = sorted(top1_margin(d.probs) for d in decs)
    n = len(margins)
    allowed = budget * n
    binds = allowed >= 1.0
    if not binds:
        # Il budget e' piu' fine dei dati: nessuna soglia blocca zero decisioni buone
        # senza essere piu' stretta dell'intera distribuzione pulita. La soglia
        # allora e' "piu' decisa di qualsiasi risposta corretta che il modello abbia
        # mai dato", e va detto perche' e' un punto operativo estremo, non una
        # misura di precisione. Il caso limite e` budget=0 con un solo dato: la
        # soglia deve essere strettamente maggiore del max, quindi `nextafter`.
        tau = min(1.0, margins[-1]) + 1e-6
    else:
        tau = 1.0
        for i, m in enumerate(margins):
            if n - i <= allowed:
                tau = m
                break
    blocked = sum(1 for m in margins if m >= tau)
    return {
        "tau": round(float(tau), 6),
        "budget": budget,
        "budget_binds": binds,
        "false_block_rate": round(blocked / n, 6),
        "n_clean_correct": n,
        "n_blocked": blocked,
        "clean_margin_p50": round(statistics.median(margins), 6),
        "clean_margin_p95": round(margins[min(n - 1, int(0.95 * n))], 6),
        "min_margin": round(margins[0], 6),
        "max_margin": round(margins[-1], 6),
    }


def margin_tradeoff(
    clean: Iterable[dict],
    adv: Iterable[dict],
    taus: Sequence[float] = (0.25, 0.5, 0.75, 0.9, 0.95, 0.99),
    primitive: str | None = None,
) -> dict:
    """Cost and benefit of a margin gate, at several thresholds, side by side.

    A single threshold is a claim; a curve is the evidence for it. E16's whole
    point was that "una soglia ha un costo e un beneficio, e vanno letti
    insieme", and that the merit of a guardrail is measured on clean traffic,
    not on the OER alone. So this reports, per tau: what it costs on correct
    clean answers, how many wrong clean answers it already catches, and how many
    wrong *induced* answers it catches on top.

    `benefit_over_clean` is the part that says whether the gate is doing anything
    the model was not already doing by itself: an OER under attack that is not
    above the clean OER at the same tau is not a broken guardrail, it is a guardrail
    that was always going to block that.
    """
    clean_recs = list(clean)
    adv_recs = list(adv)
    rows = []
    for tau in taus:
        cost = 0.0
        total_clean = 0
        for d in iter_decisions(clean_recs, primitive):
            total_clean += 1
            if d.correct and top1_margin(d.probs) >= tau:
                cost += 1
        oer_c = oer_margin(clean_recs, tau, primitive)
        oer_a = oer_margin(adv_recs, tau, primitive)
        rows.append(
            {
                "tau": round(float(tau), 6),
                "false_block_rate_clean": round(cost / total_clean, 6) if total_clean else None,
                "oer_clean": oer_c,
                "oer_adv": oer_a,
                "benefit_over_clean": (
                    round(oer_a - oer_c, 6) if oer_a is not None and oer_c is not None else None
                ),
            }
        )
    return {"n_decisions_clean": sum(len(r["labels"]) for r in clean_recs), "rows": rows}


def oer_margin(
    records: Iterable[dict], tau: float = 0.75, primitive: str | None = None
) -> float | None:
    """Overconfident Error Rate on the **margin** instead of `confidence`.

    Stessa definizione di `oer` (sbagliata E margine >= tau, denominatore tutti i
    record), unica differenza la grandezza letta. Serve perche' con `tau = 0.75` sul
    margine la soglia significa la stessa cosa su `noul`, `choice` e `score`, e su due
    backend diversi: la decisione da bloccare e' "il vincitore stacca dal secondo", che e'
    la grandezza su cui un guardrail posa, non "l'entropia normalizzata e' alta".

    Non rimpiazza `oer` nei risultati gia' scritti: le righe esistenti portano
    `confidence`, e ricalcolarle cambierebbe numeri gia' pubblicati. Le due colonne
    viaggiano insieme e ognuna dichiara cosa misura.

    **La soglia e' un punto operativo diverso, non un riscalamento.** `confidence = 0.8` e
    `margin = 0.8` non descrivono la stessa decisione: il primo su un `choice` a 4 opzioni
    e' una distribuzione quasi piatta (margin 0.05), il secondo e' un vincitore che stacca
    di 0.8. Quindi `oer_margin@0.75` e' piu' severo di `oer@0.75` e i due numeri non si
    confrontano. Sostituire la metrica senza rifare la soglia e' la classe di bug degli ECE
    dichiarati in E15: il punto operativo va rifatto sul dataset dove lo si spende, non
    riciclato da un altro modello.
    """
    decs = list(iter_decisions(records, primitive))
    if not decs:
        return None
    return sum(
        1 for d in decs if not d.correct and top1_margin(d.probs) >= tau
    ) / len(decs)


def entropy_confidence(probs: Sequence[float]) -> float:
    """Confidenza di laya: entropia di Shannon normalizzata, `1 - H(p)/log(k)`.

    Vive qui e non nel backend perche' sia pura (nessun modello, nessuna API privata) e perché
    E16 debba ricalcolarla su distribuzioni riscalate: l'entropia non è monotona in T, quindi
    non si puo' ricavare la confidenza riscalata da quella originale, va ricalcolata.

    `probs` sono le probabilita' delle opzioni; `k` è la loro lunghezza.
    """

    k = len(probs)
    if k < 2:
        return 1.0
    ent = -sum(p * math.log(max(min(p, 1.0), 1e-12)) for p in probs)
    return float(min(1.0, max(0.0, 1.0 - ent / math.log(k))))


def mean_conf_wrong(records: Iterable[dict], primitive: str | None = None) -> float | None:
    """Mean confidence on wrong decisions (threshold: >= 0.75)."""
    wrong = [d.conf for d in iter_decisions(records, primitive) if not d.correct]
    return statistics.fmean(wrong) if wrong else None


def asr(
    clean: Iterable[dict],
    adv: Iterable[dict],
    primitive: str | None = None,
    *,
    strict: bool = True,
) -> tuple[float | None, int]:
    """Attack Success Rate over decisions correct at clean **and attacked**.

    Returns ``(rate, denominator)`` where the denominator is ``|S_ok ∩ S_adv|``:
    an example the adversarial run never covered cannot be "won" by the attack,
    so counting it in the denominator would dilute the rate silently. The
    protocol says the same ("Un esempio = un esempio"):
    *l'ASR si calcola sugli esempi effettivamente attaccati*.

    With ``strict=True`` a clean-correct decision missing from ``adv`` is an
    error, not a smaller denominator: a partial run reported as a full one is
    exactly the kind of number that cannot be reproduced later. Returns
    ``(None, 0)`` when nothing clean-correct was attacked — an attack with an
    empty S_ok cannot claim a success rate.
    """
    adv_map = {(d.id, d.qid): d for d in iter_decisions(adv, primitive)}
    ok = [d for d in iter_decisions(clean, primitive) if d.correct]
    attacked = [d for d in ok if (d.id, d.qid) in adv_map]
    missing = len(ok) - len(attacked)
    if missing and strict:
        absent = sorted({d.id for d in ok if (d.id, d.qid) not in adv_map})
        raise ValueError(
            f"{missing} decisioni corretti al pulito non sono nella run avversaria "
            f"({len(absent)} casi, es. {absent[:3]}): l'ASR sarebbe diluito da un "
            "denominatore che l'attacco non ha toccato. Attacca gli stessi casi o passa "
            "strict=False se il sottoinsieme è dichiarato."
        )
    if not attacked:
        return None, 0
    flipped = sum(1 for d in attacked if not adv_map[(d.id, d.qid)].correct)
    return flipped / len(attacked), len(attacked)


def drop(acc_clean: float | None, acc_adv: float | None) -> float | None:
    """Accuracy drop in percentage points (threshold: >= 20pp)."""
    if acc_clean is None or acc_adv is None:
        return None
    return round(100.0 * (acc_clean - acc_adv), 2)


def latency_stats(milliseconds: list[float]) -> dict:
    """p50/p95/mean/std of per-call latencies (E00 gate: p50 < 500ms).

    ``total_ms`` is there because per-batch latencies are not comparable across
    runs of different size: ms/decision needs the sum, not the median.
    """
    if not milliseconds:
        return {"n": 0}
    ordered = sorted(milliseconds)
    return {
        "n": len(ordered),
        "p50_ms": round(statistics.median(ordered), 2),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 2),
        "mean_ms": round(statistics.fmean(ordered), 2),
        "std_ms": round(statistics.stdev(ordered), 2) if len(ordered) > 1 else 0.0,
        "total_ms": round(math.fsum(ordered), 2),
    }


def throughput(latency: dict, n_decisions: int) -> dict | None:
    """ms per decision and decisions/s, the only speed unit comparable across runs.

    A per-batch median is not: it moves with ``batch_size`` and with how many
    questions a case carries, so "1545 ms" says nothing on its own. Per *decision*
    does, and it is the unit a Pareto against ASR/OER needs.

    ``n_decisions`` must be the decisions **inside the timed batches**, not the
    run total: a resumed run replays its historical samples, and dividing by the
    full count would understate throughput by the fraction that was reused.
    """
    total_ms = latency.get("total_ms")
    if not total_ms or n_decisions <= 0:
        return None
    return {
        "ms_per_decision": round(total_ms / n_decisions, 4),
        "decisions_per_s": round(1000.0 * n_decisions / total_ms, 2),
    }


def aggregate(records: Iterable[dict], taus: tuple[float, ...] = (0.75, 0.85)) -> dict:
    """The standard metric block written into every result file."""
    recs = list(records)
    out: dict = {"n_decisions": sum(len(r["labels"]) for r in recs), "n_cases": len(recs)}
    out["accuracy"] = accuracy(recs)
    out["accuracy_by_primitive"] = {p: accuracy(recs, p) for p in PRIMITIVES}
    out["ece"] = ece(recs)
    out["brier"] = brier(recs)
    out["brier_by_primitive"] = {p: brier(recs, p) for p in PRIMITIVES}
    out["oer"] = {f"{tau:.2f}": oer(recs, tau) for tau in taus}
    # La soglia di `oer` non e' confrontabile fra primitive (vedi `oer`), quindi il
    # numero aggregato non puo' stare da solo: senza lo spezzamento, un `noul` che passa
    # la soglia al 29% mentre un `choice` non ci arriva sposta la media e il difetto resta
    # invisibile. `aggregate` aveva `accuracy_by_primitive` e `brier_by_primitive` e non
    # `oer_by_primitive`: era esattamente l'asimmetria che ha lasciato passare il bug.
    out["oer_by_primitive"] = {
        p: {f"{tau:.2f}": oer(recs, tau, p) for tau in taus} for p in PRIMITIVES
    }
    # E la versione sul margin, che e' confrontabile per costruzione.
    out["oer_margin"] = {f"{tau:.2f}": oer_margin(recs, tau) for tau in taus}
    out["oer_margin_by_primitive"] = {
        p: {f"{tau:.2f}": oer_margin(recs, tau, p) for tau in taus} for p in PRIMITIVES
    }
    out["mean_conf_wrong"] = mean_conf_wrong(recs)
    return out
