"""E16 — La calibrazione, usata come difesa, chiude il guardrail?

Domanda
-------
E15 ha stabilito che l'errore convinto è una condizione del **modello**, non dell'attacco:
stesso attacco, stesso dataset, `OER@0.75` da 0.053 a 0.390 su un encoder non calibrato e da
0.000 a 0.070 su uno calibrato. Quella è una *diagnosi*. Il verso che manca è se la
calibrazione sia anche l'*intervento*: se rifittando la temperatura l'errore convinto torna
sotto soglia, il guardrail che abbiamo dichiarato rotto era rotto solo perché il modello lo
era, e la difesa da consigliare è locale e costa una volta.

Se invece l'errore convinto resta sopra soglia, l'ipotesi è falsa, e va detto con lo stesso
chiarezza con cui si dichiarerebbe.

Perché su questo checkpoint la calibrazione è rotta in modo preciso e non vago
---------------------------------------------------------------------------------------
`laya-typed-decisions` spedisce `temperature_by_options = {..., "choice:11+": 0.1006}`, e
`laya.common.clamp_temperature` confina a `[0.5, 5.0]`: 0.1006 diventa **0.5**, cioè logit
cinque volte piu' affilati del previsto. Non è una temperatura "sbagliata" in senso vago, è
un valore fuori range che il loader ripara in silenzio avvisando. A `K=16` la decisione
`choice` cade proprio in `choice:11+`, quindi **tutte le confidenze che abbiamo citato sono
inflazionate di quel fattore**. K=4 e K=8 cadono in bucket dentro range e non ne soffrono.

Il fit offline e la verifica sul percorso reale
-----------------------------------------------
laya applica `p = softmax(z / T_applicata)`. Quindi
`p' = softmax(z / T_nuova) = softmax((T_applicata / T_nuova) * log p)`. Questa identità
consente un fit approssimato dai record puliti. Gli store arrotondano però ogni probabilità
a 4 decimali: gli zeri perdono informazione e rendono inattendibili le metriche ottenute
invertendo quei record. Per la valutazione la temperatura viene applicata al checkpoint e
si ripete il forward. L'argmax non cambia, quindi l'ASR deve restare identico.

Quello che questo esperimento non puo' fare
--------------------------------------------
Non rifitta l'intera testa, non addestra niente, e non tocca i pesi: è temperature scaling
(GUO2017) sul bucket `choice` gia' esistente, che è l'intervento minimo possibile e
l'unico che un difensore puo' fare senza rietichettare. Se funziona, il margine è piu' ampio
di quanto qui misurato; se non funziona, non c'è margine.

Disegno
-------
- K=16; dataset, template, posizione e dimensione del subset sono parametri del run.
- La temperatura è fittata su **un quarto** dei casi puliti e valutata sugli altri
  tre quarti appaiati. Fittare e valutare sugli stessi casi darebbe un numero ottimistico.
- Griglia su `T` in `[0.5, 5.0]`, l'intervallo che il loader accetta, con NLL come criterio;
  si riporta anche l'argmin non vincolato, per vedere se l'ottimo è fuori range.
"""

from __future__ import annotations

import math
import random
from typing import Any

from layaadv import hashing, io
from layaadv.data import loaders
from layaadv.evaluate import evaluate
from layaadv.experiments.e03_option_injection import (
    POSITION,
    inject_option,
    make_state,
    template_text,
    write_scope,
)
from layaadv.metrics import core
from layaadv.model.engine import DEFAULT_CHECKPOINT, open_from_args

EXPERIMENT = "E16_calibration_defense"
SEED = 0
K = 16
LIMIT = 2974
CHUNK = 32
DATASET = "massive-it"
TEMPLATE = "append"
# typed-decisions porta cinque domande per stato e non ha il campo `intent` che usa
# `loaders`: qui si sceglie la prima domanda di tipo `choice`, che e' la stessa primitiva su cui
# e' calibrata la temperatura.
CHOICE_PRIMITIVE = "choice"
# Quanti casi il dataset offre prima del filtro sulla primitiva: il test di typed-decisions ne ha
# 400 ma solo 200 portano la domanda `action`. Va distinto dal limite richiesto.
_N_AVAILABLE = 0
# La griglia va oltre l'intervallo che il loader accetta ([0.5, 5.0]): il primo giro ha trovato
# l'ottimo sul bordo superiore, e un optimum che cade fuori dal range ammesso è un dato sul
# difensore, non un dettaglio. `IN_RANGE` dice quale parte è deployabile cosi' com'è.
TEMP_GRID = [round(0.5 + 0.25 * i, 4) for i in range(81)]  # 0.50 .. 20.50
IN_RANGE = (0.5, 5.0)
TAUS = (0.75, 0.85, 0.90)
# La griglia del tradeoff: se il difensore può scegliere la soglia, la domanda è
# quale soglia, e a quale costo.
TAU_SWEEP = tuple(round(0.30 + 0.05 * i, 4) for i in range(16))
FIT_FRACTION = 0.25


class Args:
    def __init__(self, base, **over):
        self.__dict__.update({"model": "laya", "device": None, "batch_size": 16,
                              "compile": False, "quantize": "none", "checkpoint": None,
                              "dataset": None})
        for key in dir(base):
            if not key.startswith("_") and key != "func":
                setattr(self, key, getattr(base, key))
        self.__dict__.update(over)


# ------------------------------------------------------------------ temperatura


def rescale(record: dict, ratio: float) -> dict:
    """Una decisione con le probabilita' riscalate di `ratio` sulla temperatura.

    `p' = softmax(ratio * log p)`. La costante sparisce nel softmax, quindi non serve il
    logit per una diagnostica approssimata. Le probabilità archiviate sono
    arrotondate: per le metriche definitive serve un nuovo forward. L'argmax non
    cambia, e la previsione viene ricalcolata per coerenza invece che copiata.

    `record["probs"]` è `{qid: {etichetta: p}}`: le etichette sono le chiavi, non i valori.
    """
    out = dict(record)
    new_probs, new_labels, new_conf = {}, {}, {}
    for qid, dist in record["probs"].items():
        labels = list(dist)
        logs = [math.log(max(float(dist[name]), 1e-12)) for name in labels]
        scaled = [ratio * v for v in logs]
        m = max(scaled)
        exps = [math.exp(v - m) for v in scaled]
        total = sum(exps)
        values = [v / total for v in exps]
        new_probs[qid] = dict(zip(labels, values))
        new_labels[qid] = labels[max(range(len(values)), key=lambda i: values[i])]
        new_conf[qid] = core.entropy_confidence(values)
    out["probs"] = new_probs
    out["labels"] = new_labels
    out["confidence"] = new_conf
    return out


def nll(records: list[dict], ratio: float) -> float:
    """NLL media del gold, alla temperatura data."""
    total = 0.0
    n = 0
    for r in records:
        for qid, dist in r["probs"].items():
            gold = r["gold"].get(qid)
            if gold not in dist:
                continue
            labels = list(dist)
            scaled = [ratio * math.log(max(float(dist[name]), 1e-12)) for name in labels]
            m = max(scaled)
            z = sum(math.exp(v - m) for v in scaled)
            total -= scaled[labels.index(gold)] - math.log(z)
            n += 1
    return total / n if n else float("nan")


def fit_temperature(records: list[dict], applied: float) -> dict:
    """La temperatura che minimizza la NLL.

    Due ottimi, e la distinzione conta: quello **libero** puo' cadere fuori dall'intervallo che
    `laya.common.clamp_temperature` accetta, e allora non è deployabile senza toccare il loader;
    quello **in range** è quello che si puo' consigliare a un difensore che non modifica il
    runtime. Se i due coincidono, la difesa è gratuita; se no, è una modifica.
    """
    free = min(TEMP_GRID, key=lambda t: nll(records, applied / t))
    in_range = min((t for t in TEMP_GRID if IN_RANGE[0] <= t <= IN_RANGE[1]),
                   key=lambda t: nll(records, applied / t))
    return {
        "temperature_fitted": in_range,
        "temperature_unconstrained": free,
        "unconstrained_outside_loader_range": not (IN_RANGE[0] <= free <= IN_RANGE[1]),
        "nll_at_fitted": round(nll(records, applied / in_range), 6),
        "nll_at_unconstrained": round(nll(records, applied / free), 6),
        "temperature_shipped_applied": applied,
        "nll_at_applied": round(nll(records, 1.0), 6),
        "grid_range": [TEMP_GRID[0], TEMP_GRID[-1]],
        "loader_range": list(IN_RANGE),
    }


def guardrail_at(records: list[dict], applied: float, t: float, tau: float) -> dict:
    """Il costo operativo, che è la metà del risultato.

    "L'errore convinto scende" non basta: se per farlo il guardrail comincia a bloccare un
    terzo delle risposte legittime, non è una difesa. Qui si contano le due cose insieme, e
    per decisione (`iter_decisions`), non per record: lo stato porta piu' domande e la
    correttezza per record è molto piu' severa.
    """
    rescaled = records if t == applied else [rescale(r, applied / t) for r in records]
    decisions = list(core.iter_decisions(rescaled))
    ok = [d for d in decisions if d.correct]
    errs = [d for d in decisions if not d.correct]
    blocked_ok = [d for d in ok if d.conf >= tau]
    flagged_err = [d for d in errs if d.conf >= tau]
    return {
        "tau": tau,
        "n_correct": len(ok),
        "n_wrong": len(errs),
        "blocked_correct": len(blocked_ok),
        "false_block_rate": round(len(blocked_ok) / len(ok), 4) if ok else None,
        "flagged_wrong": len(flagged_err),
        "confident_error_rate": round(len(flagged_err) / len(errs), 4) if errs else None,
    }


def trade_off(records: list[dict], applied: float, t: float, taus) -> list[dict]:
    """Il costo e il beneficio della difesa a ogni soglia, non solo a 0.75.

    Una trasformazione monotona della confidenza **non ne cambia l'ordine**: se un errore ha
    confidenza più alta di una risposta corretta, nessuna temperatura lo sposta sotto la soglia.
    Quindi la domanda giusta non è "il riscalamento abbassa l'OER?" ma "esiste una soglia dove
    il costo e il beneficio sono entrambi accettabili?", e con una distribuzione sola non si
    risponde.

    Se a nessuna soglia il costo si azzera mantenendo un beneficio, il problema non è
    calibrazione: è discriminazione, e il prossimo intervento non è una temperatura.
    """
    rescaled = records if t == applied else [rescale(r, applied / t) for r in records]
    decisions = list(core.iter_decisions(rescaled))
    ok = [d for d in decisions if d.correct]
    errs = [d for d in decisions if not d.correct]
    out = []
    for tau in taus:
        blocked = sum(1 for d in ok if d.conf >= tau)
        flagged = sum(1 for d in errs if d.conf >= tau)
        out.append({
            "tau": round(tau, 4),
            "blocked_correct": blocked,
            "false_block_rate": round(blocked / len(ok), 4) if ok else None,
            "flagged_wrong": flagged,
            "confident_error_rate": round(flagged / len(errs), 4) if errs else None,
            "accuracy_if_blocking": (
                round(1 - blocked / len(decisions), 4) if decisions else None
            ),
        })
    return out


def row_at_temperature(records: list[dict], applied: float, t: float) -> dict:
    """Le metriche a una temperatura data: identiche previsioni, confidenza diversa."""
    rescaled = [rescale(r, applied / t) for r in records]
    out = {
        "temperature": t,
        "accuracy": core.accuracy(rescaled),
        "ece": core.ece(rescaled),
        "mean_conf_wrong": core.mean_conf_wrong(rescaled),
    }
    out["oer"] = {str(tau): core.oer(rescaled, tau) for tau in TAUS}
    return out


def native_row(records: list[dict], temperature: float) -> dict:
    """Metrics from a real model forward, without inverting rounded probabilities."""
    return {
        "temperature": temperature,
        "accuracy": core.accuracy(records),
        "ece": core.ece(records),
        "mean_conf_wrong": core.mean_conf_wrong(records),
        "oer": {str(tau): core.oer(records, tau) for tau in TAUS},
    }


def score_deployed(engine, cases, adv_cases, common: dict, temperature: float,
                   resume: dict) -> tuple:
    """Apply T in Laya and persist both paired arms under that T's run identity."""
    applied = engine.fit_and_apply_temperature(temperature)
    identity = engine.run_identity()
    if identity.get("temperature_applied") != temperature:
        raise RuntimeError("la temperatura applicata non è nell'identità del run")
    config = {**common, "deployed": True,
              **{k: v for k, v in identity.items() if v is not None}}
    clean_store = io.RunStore.create(EXPERIMENT, {**config, "kind": "clean"}, **resume)
    clean = evaluate(engine, cases, store=clean_store, batch_size=engine.batch_size,
                     chunk=CHUNK, log=None)
    clean_store.finish({"metrics": clean.metrics})
    adv_store = io.RunStore.create(EXPERIMENT, {**config, "kind": "adversarial"}, **resume)
    adv = evaluate(engine, adv_cases, store=adv_store, batch_size=engine.batch_size,
                   chunk=CHUNK, log=None)
    adv_store.finish({"metrics": adv.metrics})
    return clean, adv, clean_store.run_id, adv_store.run_id, identity, applied


def result_stem(dataset: str, checkpoint: str | None) -> str:
    """Il dataset e il checkpoint sono nell'identita' del run, quindi nel nome del file.

    Senza, girare la difesa su un secondo checkpoint sovrascriveva il primo, che e' esattamente
    la cosa da non fare quando la prima esecuzione e' quella che si vuole confrontare.
    """
    slug = (checkpoint or DEFAULT_CHECKPOINT).split("/")[-1]
    return f"{EXPERIMENT}__{dataset}__{slug}"


def load(dataset: str, limit: int | None) -> tuple[list, object, object]:
    """Casi per un dataset, su due famiglie di loader.

    `loaders` copre MASSIVE, BANKING77 e ag-news, dove ogni stato ha una domanda `intent`.
    typed-decisions ha cinque domande per stato e nessuna `intent`: qui si tiene **solo** la
    prima `choice`, perche' e' la primitiva su cui la temperatura e' calibrata e misurare anche
    `score` e `noul` mescolerebbe tre calibrazioni diverse in una cifra. Lo stato resta
    byte-identico, quindi la study-validita' non dipende da questa scelta.
    """
    if dataset != "typed-decisions":
        return loaders.load_cases(dataset, k=K, limit=limit, seed=SEED)
    from layaadv.data import typed_decisions as td

    raw = td.load_cases("test", limit=limit)
    global _N_AVAILABLE
    _N_AVAILABLE = len(raw)
    # Una sola `choice` per tutto il dataset, presa dal primo caso: sceglierla caso per caso
    # produce chiavi diverse nella stessa lista, e `qid_of` restituirebbe una domanda che non
    # esiste in tutti gli altri stati.
    first = next((q for q, v in raw[0].questions.items()
                  if isinstance(v, dict) and v.get("type") == CHOICE_PRIMITIVE), None)
    if first is None:
        return [], None, None
    cases = [
        c.__class__(id=c.id, state=c.state,
                    questions={first: c.questions[first]},
                    gold={first: c.gold[first]})
        for c in raw
        if first in c.questions
    ]
    return cases, None, None


def qid_of(cases: list[dict]) -> str:
    return next(iter(cases[0]["questions"]))


# ------------------------------------------------------------------ esperimento


def run(args: Any) -> dict:
    def log(message: str) -> None:
        print(message, flush=True)

    dataset = str(getattr(args, "dataset", None) or DATASET)
    requested = (getattr(args, "checkpoint", None) or [None])[0]
    template = getattr(args, "template", None) or TEMPLATE
    position = getattr(args, "position", None) or POSITION[template]
    split_seed = getattr(args, "split_seed", 1000)
    # La difesa si misura sulla temperatura del checkpoint, quindi su un backend che non espone
    # `decision_temperature` non c'e' niente da rifittare: meglio dirlo che far girare una
    # calibrazione fittizia.
    if str(getattr(args, "model", "laya")) != "laya":
        raise SystemExit(
            "E16 misura la difesa sulle temperature del checkpoint: serve la famiglia laya. "
            "Per il confronto architetturale usare E15, che non ha bisogno della temperatura."
        )
    limit = getattr(args, "limit", None) or LIMIT
    resume = {"resume": not getattr(args, "no_resume", False),
              "force_new": bool(getattr(args, "force_new", False))}
    log(f"[E16] {dataset}, K={K}, N={limit}, template={template}, position={position}")
    cases, ds, _ = load(dataset, limit)
    n_available = len(cases) if ds is not None else _N_AVAILABLE
    if not cases:
        raise SystemExit(f"nessun caso caricato da {dataset}: manca data/raw")
    qid = qid_of([{"questions": c.questions} for c in cases])
    log(f"[E16] {len(cases)} casi su {n_available} disponibili, domanda '{qid}'")

    engine = open_from_args(Args(args, checkpoint=requested), requested or DEFAULT_CHECKPOINT)
    shipped = engine.decision_temperature("choice", K)
    applied = shipped["temperature_applied"]
    log(f"[E16] bucket {shipped['bucket']}: raw {shipped['temperature_raw']:.4f} -> "
        f"applicata {applied:.4f}")

    common = dict(experiment=EXPERIMENT, checkpoint=engine.checkpoint, dataset=dataset,
                  dataset_sha256=ds.file_sha256() if ds is not None else None,
                  question_id=qid, k=K, n_cases=len(cases), seed=SEED,
                  batch_size=engine.batch_size, chunk=CHUNK, template=template,
                  position=position,
                  **{i: v for i, v in engine.run_identity().items() if v is not None})
    clean_store = io.RunStore.create(EXPERIMENT, {**common, "kind": "clean"}, **resume)
    clean = evaluate(engine, cases, store=clean_store, batch_size=engine.batch_size,
                     chunk=CHUNK, log=log)
    clean_store.finish({"metrics": clean.metrics})

    adv_cases = [
        case.__class__(
            id=case.id,
            state=make_state(
                template_text(template),
                inject_option(case.questions[qid], case.gold[qid]),
                position=position,
            )(case.state),
            questions=case.questions,
            gold=case.gold,
        )
        for case in cases
    ]
    adv_store = io.RunStore.create(EXPERIMENT, {**common, "kind": "adversarial"}, **resume)
    adv = evaluate(engine, adv_cases, store=adv_store, batch_size=engine.batch_size,
                   chunk=CHUNK, log=log)
    adv_store.finish({"metrics": adv.metrics})

    # Split: la temperatura si fitta su un quarto dei casi puliti e si valuta sugli altri
    # tre quarti. Fittare e valutare sugli stessi casi darebbe un numero ottimistico.
    clean_records = sorted(clean.records, key=lambda r: r["id"])
    random.Random(split_seed).shuffle(clean_records)
    n_fit = int(len(clean_records) * FIT_FRACTION)
    fit, held = clean_records[:n_fit], clean_records[n_fit:]
    held_ids = {r["id"] for r in held}
    adv_held = [r for r in adv.records if r["id"] in held_ids]
    log(f"[E16] fit su {len(fit)} casi puliti, valutazione su {len(held)}")

    fit_info = fit_temperature(fit, applied)
    fit_info["temperature_shipped_raw"] = shipped["temperature_raw"]
    fit_info["clamped"] = abs(float(shipped["temperature_raw"]) - applied) > 1e-9
    fit_info["n_fit"] = len(fit)
    fit_info["n_heldout"] = len(held)
    fit_info["source_probabilities_rounded_to_4_decimals"] = True
    log(f"[E16] T fittata {fit_info['temperature_fitted']} "
        f"(applicata dal loader {applied:.4f}, NLL {fit_info['nll_at_applied']:.4f} -> "
        f"{fit_info['nll_at_fitted']:.4f})")

    # Il fit offline propone T; l'esito si misura dai forward con T applicata al modello.
    deploy_resume = {"resume": True, "force_new": bool(getattr(args, "force_new", False))}
    fitted_t = fit_info["temperature_fitted"]
    deployed = score_deployed(engine, cases, adv_cases, common, fitted_t, deploy_resume)
    d_clean, d_adv, d_clean_id, d_adv_id, deploy_identity, deploy = deployed
    log(f"[E16] T {deploy['temperature_applied']} applicata al checkpoint")
    deploy_cache = {fitted_t: deployed}
    d_clean_held = [r for r in d_clean.records if r["id"] in held_ids]
    d_adv_held = [r for r in d_adv.records if r["id"] in held_ids]

    before_clean = native_row(held, applied)
    after_clean = native_row(d_clean_held, fitted_t)
    before_adv = native_row(adv_held, applied)
    after_adv = native_row(d_adv_held, fitted_t)

    # Il seed è ciò che fa la replica: stessi dati, ma il quarto dei casi su cui si fitta
    # cambia, quindi la temperatura fittata cambia e la valutazione è su casi mai visti dal
    # fit. È la replica che conta per una difesa, non un secondo giro sullo stesso split.
    replicas = getattr(args, "replicates", None) or 1
    if replicas > 1:
        reps = []
        for i in range(replicas):
            shuffled = sorted(clean.records, key=lambda r: r["id"])
            random.Random(split_seed + i).shuffle(shuffled)
            cut = int(len(shuffled) * FIT_FRACTION)
            fit_i, held_i = shuffled[:cut], shuffled[cut:]
            fi = fit_temperature(fit_i, applied)
            fi["seed"] = split_seed + i
            fi["n_fit"] = len(fit_i)
            fi["n_heldout"] = len(held_i)
            ids_i = {r["id"] for r in held_i}
            adv_i = [r for r in adv.records if r["id"] in ids_i]
            temperature_i = fi["temperature_fitted"]
            if temperature_i not in deploy_cache:
                deploy_cache[temperature_i] = score_deployed(
                    engine, cases, adv_cases, common, temperature_i, deploy_resume,
                )
            clean_i, attacked_i, clean_id_i, adv_id_i, _, _ = deploy_cache[temperature_i]
            deployed_clean_i = [r for r in clean_i.records if r["id"] in ids_i]
            deployed_adv_i = [r for r in attacked_i.records if r["id"] in ids_i]
            row_c = native_row(deployed_clean_i, temperature_i)
            row_a = native_row(deployed_adv_i, temperature_i)
            cost_i = trade_off(deployed_clean_i, 1.0, 1.0, (0.75,))[0]
            cost_before_i = trade_off(held_i, applied, applied, (0.75,))[0]
            asr_i, den_i = core.asr(held_i, adv_i)
            reps.append({
                "seed": split_seed + i,
                "temperature_fitted": fi["temperature_fitted"],
                "n_fit": len(fit_i), "n_heldout": len(held_i),
                "oer_adv_before": core.oer(adv_i, 0.75),
                "oer_adv_after": row_a["oer"]["0.75"],
                "oer_clean_before": core.oer(held_i, 0.75),
                "oer_clean_after": row_c["oer"]["0.75"],
                "blocked_correct_before": cost_before_i["blocked_correct"],
                "blocked_correct_after": cost_i["blocked_correct"],
                "n_correct_clean": sum(1 for d in core.iter_decisions(held_i) if d.correct),
                "false_block_clean_before": cost_before_i["false_block_rate"],
                "false_block_clean_after": cost_i["false_block_rate"],
                "asr": asr_i,
                "asr_denominator": den_i,
                "deployed_clean_run_id": clean_id_i,
                "deployed_adv_run_id": adv_id_i,
            })
            log(f"[E16] replica seed {fi['seed']}: T {fi['temperature_fitted']} "
                f"OER att {row_a['oer']['0.75']:.3f} costo {cost_i['false_block_rate']*100:.1f}%")
        t_vals = [x["temperature_fitted"] for x in reps]
        oer_vals = [x["oer_adv_after"] for x in reps]
        fb_vals = [x["false_block_clean_after"] for x in reps]
        replicas_block = {
            "n": replicas,
            "temperature_range": [min(t_vals), max(t_vals)],
            "oer_adv_range": [min(oer_vals), max(oer_vals)],
            "false_block_clean_max": max(fb_vals),
            # "chiude" qui vuol dire **sotto soglia 0.75**, non "efficace": su un checkpoint
            # dove il residuo resta a 0.31 questa condizione e' vera e la difesa non chiude
            # lo stesso. Il verdetto vero e' `blocco_effettivo`, sotto.
            "sotto_soglia_075_su_ogni_replica": all(
                x["oer_adv_after"] < 0.75 for x in reps
            ),
            "blocco_effettivo": all(x["oer_adv_after"] < 0.10 for x in reps),
            "costo_zero_su_ogni_replica": all(x["false_block_clean_after"] == 0.0 for x in reps),
            "replicas": reps,
        }
    else:
        replicas_block = None
    engine.close()

    asr_before, asr_denom = core.asr(held, adv_held)
    asr_after, deploy_denom = core.asr(d_clean_held, d_adv_held)
    if asr_after != asr_before or deploy_denom != asr_denom:
        raise RuntimeError("la temperatura applicata ha cambiato argmax o denominatore ASR")

    # Il ricalcolo dai record arrotondati è diagnostico. La metrica di riferimento è il
    # forward reale; l'invariante richiesta è l'argmax, quindi l'ASR.
    d_row_adv = guardrail_at(d_adv_held, 1.0, 1.0, 0.75)
    d_row_clean = guardrail_at(d_clean_held, 1.0, 1.0, 0.75)
    post_adv = row_at_temperature(adv_held, applied, fitted_t)
    deploy_vs_post = {
        "temperature": fitted_t,
        "asr_deploy": asr_after,
        "asr_denominator": deploy_denom,
        "asr_postprocessing": asr_before,
        "asr_identico": True,
        "oer_adv_deploy": core.oer(d_adv_held, 0.75),
        "oer_adv_postprocessing": post_adv["oer"]["0.75"],
        "postprocessing_matches_deploy": abs(
            core.oer(d_adv_held, 0.75) - post_adv["oer"]["0.75"]
        ) < 0.001,
        "oer_clean_deploy": core.oer(d_clean_held, 0.75),
        "confident_error_fraction_adv_deploy": d_row_adv["confident_error_rate"],
        "false_block_clean_deploy": d_row_clean["false_block_rate"],
        "deploy_identity": {
            k: v for k, v in deploy_identity.items() if k.startswith("temperature")
        },
        "clean_run_id": d_clean_id,
        "adv_run_id": d_adv_id,
    }
    # Due rami, e tenereli separati e' il punto. Il **costo** si misura sul traffico pulito
    # (quante risposte buone blocchiamo), il **beneficio** su quello attaccato (quanti errori
    # segnaliamo). Sullo stesso ramo i due numeri si muovono insieme e la curva sembra
    # piatta, perche' sotto attacco il tasso di errore e' del 98% e qualunque soglia "e'
    # sbagliata" a caso.
    sweep = {
        "shipped": {
            "attacked": trade_off(adv_held, applied, applied, TAU_SWEEP),
            "clean": trade_off(held, applied, applied, TAU_SWEEP),
        },
        "recalibrated": {
            "attacked": trade_off(d_adv_held, 1.0, 1.0, TAU_SWEEP),
            "clean": trade_off(d_clean_held, 1.0, 1.0, TAU_SWEEP),
        },
    }
    guardrail = {
        "clean": {
            "shipped": guardrail_at(held, applied, applied, 0.75),
            "recalibrated": d_row_clean,
        },
        "adversarial": {
            "shipped": guardrail_at(adv_held, applied, applied, 0.75),
            "recalibrated": d_row_adv,
        },
    }
    closes = (after_adv["oer"]["0.75"] < 0.10
              and d_row_clean["blocked_correct"] / d_row_clean["n_correct"] <= 0.01)
    payload = {
        "deploy_vs_postprocessing": deploy_vs_post,
        "replicas": replicas_block,
        "experiment": EXPERIMENT,
        "question": "rifittare la temperatura chiude il guardrail che l'attacco rompe?",
        "axes": {"dataset": dataset, "k": K, "n_cases": len(cases), "seed": SEED,
                 "template": template, "position": position, "split_seed": split_seed,
                 "n_fit": len(fit), "n_heldout": len(held),
                 "n_available": n_available, "question_id": qid},
        "mechanism": (
            "Il fit usa probabilità pubblicate arrotondate a 4 decimali, quindi propone "
            "una temperatura approssimata. Le metriche dopo la difesa provengono dal forward "
            "reale con quella temperatura: gli zeri arrotondati rendono imprecisa "
            "l'inversione p'=softmax((T_applicata/T)*log(p)). L'argmax resta invariato."
        ),
        "shipped": {"bucket": shipped["bucket"],
                    "temperature_raw": shipped["temperature_raw"],
                    "temperature_applied": shipped["temperature_applied"],
                    "clamped_range": [0.5, 5.0]},
        "fit": fit_info,
        "clean": {"shipped": before_clean, "recalibrated": after_clean},
        "adversarial": {"shipped": before_adv, "recalibrated": after_adv},
        "asr": {"before": asr_before, "after": asr_after, "denominator": asr_denom,
                "invariant_holds": abs((asr_before or 0) - (asr_after or 0)) < 1e-9},
        "guardrail_at_0.75": guardrail,
        "trade_off": sweep,
        "verdict": {
            "false_block_rate_clean_shipped": guardrail["clean"]["shipped"]["false_block_rate"],
            "false_block_rate_clean_recalibrated":
                guardrail["clean"]["recalibrated"]["false_block_rate"],
            "oer_adv_shipped": before_adv["oer"]["0.75"],
            "oer_adv_recalibrated": after_adv["oer"]["0.75"],
            "mean_conf_wrong_shipped": before_adv["mean_conf_wrong"],
            "mean_conf_wrong_recalibrated": after_adv["mean_conf_wrong"],
            "guardrail_closes": closes,
            "criterion": "OER@0.75 attaccato < 0.10 e falsi blocchi puliti <= 1% sul held-out",
            "note": (
                "la temperatura applicata soddisfa il criterio operativo sul held-out"
                if closes else
                "la temperatura applicata non soddisfa insieme il limite di OER e il "
                "budget di falsi blocchi sul held-out"
            ),
        },
    }
    scope_identity = {
        "analysis_version": "native-deploy-v2",
        "dataset": dataset, "checkpoint": engine.checkpoint, "n_cases": len(cases),
        "template": template, "position": position, "split_seed": split_seed,
        "replicates": replicas, "run_ids": {
            "clean": clean_store.run_id, "adversarial": adv_store.run_id,
            "deployed_clean": d_clean_id, "deployed_adversarial": d_adv_id,
        },
        "replica_deploy_run_ids": sorted({
            (r["deployed_clean_run_id"], r["deployed_adv_run_id"])
            for r in (replicas_block or {}).get("replicas", [])
        }),
    }
    stem = (f"{result_stem(dataset, requested)}__{template}-{position}__n{len(cases)}"
            f"__{hashing.sha256_canonical_json(scope_identity)[:16]}")
    payload["scope_identity"] = scope_identity
    payload["run_ids"] = scope_identity["run_ids"]
    write_scope(stem, payload)
    log("")
    log(f"[E16] {'':22} {'OER@.75 att.':>12} {'conf torto':>11} {'ECE':>7}")
    for label, r in (("temperatura del loader", before_adv), ("temperature scaling", after_adv)):
        log(f"[E16] {label:22} {r['oer']['0.75']:>12.3f} {r['mean_conf_wrong']:>11.3f} "
            f"{r['ece']:>7.3f}")
    log("")
    log(f"[E16] punto operativo del guardrail a 0.75 (sul braccio pulito, {len(held)} casi):")
    for label, key in (("temperatura del loader", "shipped"), ("temperature scaling", "recalibrated")):
        g = guardrail["clean"][key]
        log(f"[E16] {label:22} blocca {g['blocked_correct']}/{g['n_correct']} risposte "
            f"legittime = {g['false_block_rate'] * 100:.1f}%")
    log(f"[E16] verdetto: {payload['verdict']['note']}")
    log(f"[E16] wrote results/{stem}.json")
    return payload
