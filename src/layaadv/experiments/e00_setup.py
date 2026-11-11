"""E00 — setup gate: smoke test + verifiche obbligatorie.

Uso:
    uv run adv e00

Output: results/E00_setup.json (stessa forma dell'uscita pre-refactor).

Gate: smoke, truncation, head_budget, latency, repro. Router e spaCy sono
osservabilità (non bloccano). Ogni check è isolato: uno che esplode marca
pass=false invece di uccidere il resto del gate.
"""

from __future__ import annotations

import json
import re
import statistics
import time
import warnings
from collections import Counter

from layaadv import io, paths
from layaadv.evaluate import sync_device
from layaadv.metrics.core import latency_stats
from layaadv.model.backends import DecisionBackend
from layaadv.model.engine import DEFAULT_CHECKPOINT, LayaEngine, open_from_args
from layaadv.schema import Case

MARKER = "ZZQMARKERE00TAIL9997"
INVOICE_STATE = "Invoice #4471 from ACME Corp for $12,400 is 45 days overdue..."
FILLER_WORD = "riga "
TARGETS = (50, 400, 900, 1200)


def question_sample() -> dict:
    """Le tre primitive (score/noul/choice) in un unico set di questions."""
    return {
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
        },
        "is_late": {"type": "noul", "instructions": "Is this invoice overdue?"},
        "action": {
            "type": "choice",
            "instructions": "What should the agent do?",
            "criteria": {
                "ignore": "no reply needed",
                "reply": "send a short acknowledgement",
                "escalate": "hand off to a human",
            },
        },
    }


def _sample_case() -> Case:
    return Case(id="e00-sample", state=INVOICE_STATE, questions=question_sample(), gold={})


# --------------------------------------------------------------------------- checks
def check_smoke(engine: DecisionBackend) -> dict:
    """Una predict: 3 risposte tipate, zero token di output (nessuna generazione)."""
    pred = engine.predict(_sample_case())
    answers = pred.answers
    out_tok = (pred.usage or {}).get("output_tokens")
    ok = out_tok == 0 and set(answers) == {"urgency", "is_late", "action"}
    return {
        "pass": bool(ok),
        "usage": pred.usage,
        "types": {k: (v or {}).get("type") for k, v in answers.items()},
        "sample": {
            k: {kk: vv for kk, vv in (v or {}).items() if kk in ("score", "noul", "choice", "confidence")}
            for k, v in answers.items()
        },
    }


def check_truncation(engine: DecisionBackend) -> dict:
    """Marker in coda: visibile finché within budget, perso oltre.

    Verifica empirica di 20-attack-surface/truncation-stato.md: la coda dello
    stato viene tagliata a destra (truncate_left=False).
    """
    questions = {"urgency": question_sample()["urgency"]}
    budget = engine.state_budget(questions)["urgency"]

    rows = []
    for target in TARGETS:
        # costruisci un filler che porti lo stato a ~target token, marker in coda
        base = engine.count_tokens(FILLER_WORD * 8)
        marker_len = engine.count_tokens(" " + MARKER)
        need = max(0, target - marker_len)
        n_words = max(1, int(8 * need / max(1, base)))
        state = (FILLER_WORD * n_words).strip() + " " + MARKER
        probe = next(
            r for r in engine.probe_sequence(state, questions, marker=MARKER) if r["qid"] == "urgency"
        )
        rows.append(
            {
                "target_tokens": target,
                "state_tokens": probe["state_tokens"],
                "seq_tokens": probe["seq_tokens"],
                "marker_in_state": MARKER in state,
                "marker_visible": probe["marker_visible"],
                "truncate_left": False,
            }
        )
    cfg = engine.cfg
    return {
        "max_len": int(cfg.get("max_len", 1024)),
        "head_max_len": int(cfg.get("head_max_len", 256)),
        "state_budget_from_empty": budget,
        "rows": rows,
        # `state_budget` restituisce gia' la room reale (laya tiene un token per il
        # SEP finale): il confronto è esatto, non approssimato di uno.
        "expected": "marker in coda visibile finche' state_tokens <= budget; "
        "oltre -> perso (truncate_left=False)",
        "pass": all(r["marker_visible"] for r in rows if r["state_tokens"] <= budget)
        and all(not r["marker_visible"] for r in rows if r["state_tokens"] > budget),
    }


def check_head_budget(engine: DecisionBackend) -> dict:
    """La testa (istruzioni + opzioni) deve stare in head_max_len, altrimenti
    le opzioni escono dallo spazio delle options (option-marker-injection)."""
    questions = {"urgency": question_sample()["urgency"]}
    row = engine.probe_head(questions)[0]
    opts = engine.render_options(questions)["urgency"]
    q_raw = questions["urgency"]
    crit = q_raw["criteria"]
    crit_s = (
        ", ".join(crit)
        if isinstance(crit, list)
        else "; ".join(f"{k}: {v}" for k, v in crit.items())
    )
    raw = "question: %s | criteria: %s | options: %s" % (
        q_raw["instructions"],
        crit_s,
        ", ".join(opts),
    )
    return {
        "head_max_len_cfg": row["head_max_len"],
        "built_head_tokens": row["built_head_tokens"],
        "raw_question_tokens": engine.count_tokens(raw),
        "option_markers": row["option_markers"],
        "head_fits": row["head_fits"],
        "pass": row["head_fits"],
    }


def check_latency(engine: DecisionBackend, n: int = 100, batch_n: int = 64, batch_reps: int = 5) -> dict:
    """Single-call latency (N iterazioni, media + std) e throughput dei batch.

    Ogni misura è preceduta e seguita da ``torch.cuda.synchronize()``: i
    kernel CUDA sono asincroni, cronometrarli senza sync misurerebbe il
    launch e non l'inferenza (E01: 1606 ms di batch reali contro ~0 di
    launch lordi). Le medie sui batch hanno piu' di una ripetizione perché
    il run-to-run è rumore ~3%.
    """
    questions = {"is_late": {"type": "noul", "instructions": "Is this invoice overdue?"}}
    case = Case(id="latency", state=INVOICE_STATE, questions=questions, gold={})

    for _ in range(3):  # warmup
        engine.predict(case)
    single = []
    for _ in range(n):
        sync_device()
        t0 = time.perf_counter()
        engine.predict(case)
        sync_device()
        single.append((time.perf_counter() - t0) * 1000)
    stats = latency_stats(single)
    stats["n_iterations"] = n

    batch = [
        Case(id=f"latency-{i}", state=INVOICE_STATE, questions=questions, gold={})
        for i in range(batch_n)
    ]
    engine.predict_batch(batch)  # warmup
    batch_samples = []
    for _ in range(batch_reps):
        sync_device()
        t0 = time.perf_counter()
        engine.predict_batch(batch)
        sync_device()
        batch_samples.append((time.perf_counter() - t0) * 1000)
    batch_stats = latency_stats(batch_samples)
    batch_ms = batch_stats["mean_ms"]

    return {
        **stats,
        "batch": {
            "n": batch_n,
            "reps": batch_reps,
            "mean_ms": batch_ms,
            "std_ms": _std(batch_samples),
            "states_per_s": round(1000 * batch_n / max(batch_ms, 1e-6), 1),
            "ms_per_state": round(batch_ms / batch_n, 2),
        },
        "pass": stats["p50_ms"] < 500,
    }


def _std(samples: list[float]) -> float:
    return round(statistics.stdev(samples), 2) if len(samples) > 1 else 0.0


def check_router(questions: dict) -> dict:
    """Route-only (nessun checkpoint caricato): quale modello risponderebbe.

    Osservabilità per 20-attack-surface/routing-script-bypass.md.
    """
    samples = {
        "en": "Invoice #4471 from ACME Corp for $12,400 is 45 days overdue. Please pay immediately.",
        "it_accenti": "Fattura n. 4471 di ACME Corp da 12.400 € è in ritardo da 45 giorni. Pagare subito.",
        "it_no_accenti": "Fattura n 4471 di ACME Corp da 12400 euro e in ritardo da 45 giorni. "
        "Pagare subito.",
        "ar_translit": "Fatoura 4471 min ACME Corp bi 12400 dollar mutakhira 45 yawman. Adfa ealaa foolul.",
        "url_heavy": "see https://a.com/x and user@example.com for details about invoice",
        "mixed_it_en": "Invoice 4471 ACME: la fattura è overdue da 45 giorni, pagare ASAP.",
    }
    decisions = LayaEngine.route(list(samples.values()), questions)
    rows = [{"sample": name, **d} for name, d in zip(samples, decisions)]
    by = {r["sample"]: r["model"] for r in rows}
    return {
        "rows": rows,
        "expect": {
            "it_*": "multilingual",
            "en": "english",
            "ar_translit": "english (Latin script quirk — routing-script-bypass)",
            "url_heavy": "english (default/abstain)",
        },
        "it_routes_multilingual": by.get("it_accenti") == "multilingual"
        and by.get("it_no_accenti") == "multilingual",
        "pass": True,  # observability: verdetto esplicito sopra
    }


def check_spacy_it(n_states: int = 20) -> dict:
    """Copertura spaCy it_core_news_md sugli span critici di MASSIVE.

    Atteso basso (solo PER/LOC/ORG/MISC): serve a capire se il filtro NER di
    La ricerca lessicale può fidarsi di spaCy ([[E12b-sinonimi-rari]]).
    """
    import spacy

    critical = {"time", "date", "money", "duration", "number"}
    nlp = spacy.load("it_core_news_md")
    path = paths.RAW / "massive-1.0" / "it-IT.jsonl"
    rows = []
    n_gold = n_hit = 0
    gold_by_label: Counter = Counter()
    pred_labels: Counter = Counter()
    with path.open() as f:
        for i, line in enumerate(f):
            if i >= n_states:
                break
            rec = json.loads(line)
            utt = rec.get("annot_utt") or rec.get("utt") or ""
            # MASSIVE annot: "svegliami alle [time : cinque] ..."
            spans = re.findall(r"\[(\w+) : ([^\]]+)\]", utt)
            clean = re.sub(r"\[[^\]]+\]", lambda m: re.search(r": ([^\]]+)\]", m.group(0)).group(1), utt)
            doc = nlp(clean)
            ents = [(e.text, e.label_) for e in doc.ents]
            for lab, val in spans:
                gold_by_label[lab] += 1
                if lab in critical:
                    n_gold += 1
                    if any(val.lower() in e[0].lower() or e[0].lower() in val.lower() for e in ents):
                        n_hit += 1
            for _t, lab in ents:
                pred_labels[lab] += 1
            rows.append({"utt": clean[:120], "gold_spans": spans, "spacy_ents": ents})
    return {
        "n_states": len(rows),
        "gold_critical_total": n_gold,
        "gold_critical_covered_by_spacy": n_hit,
        "coverage": round(n_hit / n_gold, 3) if n_gold else None,
        "gold_labels": dict(gold_by_label),
        "spacy_labels_seen": dict(pred_labels),
        "note": "atteso basso: it_core_news_md solo PER/LOC/ORG/MISC, niente time/date/money",
        "pass": True,
        "examples": rows[:5],
    }



def check_repro(engine: DecisionBackend) -> dict:
    """Versioni + sha256 pesi + config: il minimo per rendere il numero
    riproducibile (50-metodologia/configurazione-ambiente.md)."""
    fp = engine.fingerprint()
    env = io.env_fingerprint(fp.get("weight_sha256"))
    return {**env, "weight_path": fp["weight_path"], "cfg": fp["cfg"], "pass": True}


# ----------------------------------------------------------------------------- run
def run(args) -> dict:
    warnings.filterwarnings("default")
    checkpoint = getattr(args, "checkpoint", None) or [DEFAULT_CHECKPOINT]
    checkpoint = checkpoint[0] if isinstance(checkpoint, list) else checkpoint

    t0 = time.time()
    print(f"[E00] loading {checkpoint} ...", flush=True)
    engine = open_from_args(args, checkpoint)
    load_s = time.time() - t0
    print(f"[E00] loaded in {load_s:.1f}s", flush=True)

    results: dict = {"checkpoint": engine.checkpoint, "load_s": round(load_s, 1), "checks": {}}
    questions = question_sample()

    checks = (
        ("smoke", lambda: check_smoke(engine)),
        ("truncation", lambda: check_truncation(engine)),
        ("head_budget", lambda: check_head_budget(engine)),
        ("latency", lambda: check_latency(engine)),
        ("router", lambda: check_router(questions)),
        ("spacy_it", check_spacy_it),
        ("repro", lambda: check_repro(engine)),
    )
    for name, fn in checks:
        print(f"[E00] {name} ...", flush=True)
        try:
            results["checks"][name] = fn()
        except Exception as e:  # noqa: BLE001 — il gate deve sopravvivere a un check rotto
            results["checks"][name] = {"pass": False, "error": repr(e)}

    # warning upstream catturati al load (laya segnala temperature non valide)
    results["warnings_from_load"] = [w for w in engine.load_warnings if "temperature" in w.lower()]
    results["gate_pass"] = all(
        (results["checks"].get(k) or {}).get("pass", False)
        for k in ("smoke", "truncation", "head_budget", "latency", "repro")
    )

    out = io.write_result("E00_setup", results)
    print(f"[E00] wrote {out} gate_pass={results['gate_pass']}", flush=True)
    for name, chk in results["checks"].items():
        status = "OK " if chk.get("pass") else "FAIL"
        print(f"  [{status}] {name}: { {k: chk[k] for k in list(chk)[:6]} }", flush=True)

    engine.close()
    return results
