"""Sonda: cosa fa un simbolo aggiunto allo stato?

Quattro braccia, per separare tre cose che si confondono:

  A  simbolo DENTRO un valore JSON   -> perturbazione lessicale, non rompe nulla
  B  simbolo che BREAKA la struttura -> attacco di formato, se il modello usa la struttura
  C  solo spazi (controllo)          -> il piano: nessun simbolo, solo whitespace
  C2 doppio spazio (controllo)       -> nessun simbolo, spazio nel punto giusto

Se A non muove niente e B crolla, il modello non legge le parole ma legge la
struttura: E12b allora non era "sostituti sbagliati", era la domanda sbagliata.
"""

import json

from layaadv.data import typed_decisions as td
from layaadv.evaluate import evaluate
from layaadv.model.engine import DEFAULT_CHECKPOINT, LayaEngine
from layaadv.schema import Case

SYMBOL = "#"
CASES_N = 30

cases = td.load_cases("test", limit=CASES_N)
eng = LayaEngine(DEFAULT_CHECKPOINT, batch_size=16)
clean = evaluate(eng, cases, batch_size=16, chunk=32)
cb = {r["id"]: r for r in clean.records}


def mutate_value(case, pos_offset, symbol=SYMBOL):
    """Inserisce il simbolo piu' vicino a una posizione di contenuto, dentro la stringa.

    Uso lo stato reale: prendo un offset di una parola di contenuto (dal modulo E12b)
    e inserisco il simbolo li'. Resto dentro i delimitatori JSON per non rompere nulla.
    """
    import random

    from layaadv.experiments.e12_rare_syn import build_nlp, content_candidates

    nlp = build_nlp("en_core_web_md")
    cands = content_candidates(nlp, case.state, 3)
    if not cands:
        return None
    rng = random.Random(0)
    rng.shuffle(cands)
    c = cands[pos_offset % len(cands)]
    # offset del token nel testo
    doc = nlp(case.state)
    tok = doc[c["index"]]
    return case.state[: tok.idx] + symbol + case.state[tok.idx :]


def break_structure(case):
    """Rimuove una virgola: JSON invalido, testo ancora leggibile."""
    for i, ch in enumerate(case.state):
        if ch == ",":
            return case.state[:i] + case.state[i + 1 :]
    return case.state


def add_space(case):
    """Controllo: solo uno spazio, nessun simbolo."""
    mid = len(case.state) // 2
    return case.state[:mid] + " " + case.state[mid:]


def valid_json(state):
    try:
        json.loads(state)
        return True
    except Exception:
        return False


arms = {
    "A  simbolo dentro un valore": [mutate_value(c, k) for k, c in enumerate(cases)],
    "B  simbolo che rompe il JSON": [break_structure(c) for c in cases],
    "C  spazio singolo (controllo)": [add_space(c) for c in cases],
    "C2 spazio in due punti (controllo)": [
        (lambda s: s[: len(s) // 3] + " " + s[len(s) // 3 :])(
            (lambda s: s[: 2 * len(s) // 3] + " " + s[2 * len(s) // 3 :])(c.state)
        )
        for c in cases
    ],
}

print(f"baseline acc: {clean.metrics['accuracy']:.4f}\n")
print(
    f"{'braccia':<32} {'JSON ok':>8} {'Δtoken':>8} {'label cambiate':>15} {'flip':>6} {'ASR':>7}"
)
print("-" * 84)

base_tokens = [eng.count_tokens(c.state) for c in cases]
for name, states in arms.items():
    ok = [s for s in states if s is not None]
    json_ok = sum(1 for s in ok if valid_json(s))
    mutants = [
        Case(id=c.id, state=s, questions=c.questions, gold=c.gold)
        for c, s in zip(cases, ok)
        if s is not None
    ]
    adv = evaluate(eng, mutants, batch_size=16, chunk=32)
    adv_by_id = {r["id"]: r for r in adv.records}
    diff = flips = corr = tot = 0
    for c in mutants:
        for qid, g in c.gold.items():
            tot += 1
            b = cb[c.id]["labels"][qid]
            a = adv_by_id[c.id]["labels"][qid]
            if b == g:
                corr += 1
            if a != b:
                diff += 1
            if b == g and a != g:
                flips += 1
    d_tok = sum(
        eng.count_tokens(s) - t for s, t in zip(ok, base_tokens)
    ) / max(len(ok), 1)
    print(
        f"{name:<32} {json_ok}/{len(ok):>3} {d_tok:>8.1f} "
        f"{diff:>8}/{tot:<6} {flips:>6} {flips / corr if corr else 0:>7.4f}"
    )

eng.close()
