"""E12n — multicarattere e bidi override.

Due domande diverse.

**1. I multicarattere allargano la copertura?** La risposta è gia' nell'utenza di
UTS #39 e va detta prima di misurare: il file **non contiene nessuna sequenza
multi-carattere verso una lettera latina** (0 righe; l'unica voce inversa è
`m` -> `rn`, che è una scomposizione latino-latina, inutile per il
cross-script). E fra le 26 lettere, `f`, `g`, `k`, `m` **non hanno confusibile
in nessuno script**. Quindi il vincolo non è l'arita' della sostituzione ma
l'esistenza del carattere: i multicarattere non possono raggiungerle.

Cosa puo' fare un multicarattere, allora? Potrebbe corrompere la
tokenizzazione **a metà costo**: sostituire un digramma con una sequenza mista
costa meno che sostituire due lettere singole. È una domanda economica, e la
misuro in ASR/token.

**2. Il bidi override funziona?** Previsione: **no**, e per un motivo che è già
noto. U+202E/U+202D cambiano l'ordine di *rendering*, non i codepoint: il
codepoint del carattere invertito è esattamente quello che l'encoder legge. Su
un encoder non c'è rendering, quindi l'attacco aggiunge due caratteri invisibili
e basta. Dovrebbe comportarsi come i zero-width. Misuro per confermare.
"""

import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from e12f_homoglyph_tiers import build_tables  # noqa: E402

from layaadv.data import typed_decisions as td  # noqa: E402
from layaadv.evaluate import evaluate  # noqa: E402
from layaadv.model.engine import DEFAULT_CHECKPOINT, LayaEngine  # noqa: E402
from layaadv.schema import Case  # noqa: E402

N_CASES = 30

# --- multicarattere: coppie 2-charattere dichiarate, NON normative.
# Sono affermazioni tipografiche (come la prova B di E12f), non UTS #39.
MULTI = {
    "rn": "м",   # rn -> m
    "vv": "ш",   # vv -> w
    "cl": "d",   # cl -> d
    "nn": "п",   # nn -> n
    "ii": "м",   # ii -> m
}
# lettere senza confusibile cross-script, con candidati tipografici dichiarati
EXTRA_LETTERS = {
    "m": "т",   # cirillico te, simile a m inCorsivo
    "n": "п",   # cirillico pe
    "k": "ķ",   # k con cediglia, altro script non ma forma diversa
}

# --- bidi
RLO = "‮"
LRO = "‭"
PDF = "‬"
LRI = "⁦"


def sub_single(state, table, dose, seed=0):
    rng = random.Random(seed)
    pos = [i for i, ch in enumerate(state) if ch in table]
    if not pos:
        return state
    rng.shuffle(pos)
    keep = set(pos[: max(1, int(len(pos) * dose))])
    return "".join(table[ch] if i in keep else ch for i, ch in enumerate(state))


def sub_multi(state, dose, seed=0):
    """Sostituisce digrammi con la loro versione 'simile', dentro la prosa."""
    rng = random.Random(seed)
    out = []
    i = 0
    hits = []
    low = state.lower()
    for dig, rep in MULTI.items():
        j = low.find(dig)
        while j != -1:
            hits.append(j)
            j = low.find(dig, j + 1)
    if not hits:
        return state
    rng.shuffle(hits)
    keep = set(hits[: max(1, int(len(hits) * dose))])
    while i < len(state):
        if i in keep:
            dig = state[i : i + 2].lower()
            if dig in MULTI:
                out.append(MULTI[dig])
                i += 2
                continue
        out.append(state[i])
        i += 1
    return "".join(out)


def inject_bidi(state, kind, dose, seed=0):
    """Inserisce un override bidirezionale davanti a una frazione di parole."""
    rng = random.Random(seed)
    opener = {"RLO": RLO, "LRO": LRO, "LRI": LRI}[kind]
    out = []
    for ch in state:
        if ch == " " and rng.random() < dose:
            out.append(opener)
        out.append(ch)
    return "".join(out)


def main():
    cases = td.load_cases("test", limit=N_CASES)
    eng = LayaEngine(DEFAULT_CHECKPOINT, batch_size=16)
    clean = evaluate(eng, cases, batch_size=16, chunk=32)
    cb = {r["id"]: r for r in clean.records}
    total = sum(len(c.gold) for c in cases)
    base_acc = clean.metrics["accuracy"]
    print(f"baseline acc {base_acc:.4f} su {total} decisioni")
    s = cases[0].state
    print(f"stati {min(eng.count_tokens(c.state) for c in cases)}-"
          f"{max(eng.count_tokens(c.state) for c in cases)} token\n")

    # quante volte esistono i digrammi
    low_all = " ".join(c.state.lower() for c in cases)
    print("digrammi presenti negli stati:")
    for dig, rep in MULTI.items():
        print(f"  {dig!r} -> {rep!r}: {low_all.count(dig)} occorrenze")
    n_hit = sum(1 for c in cases if any(d in c.state.lower() for d in MULTI))
    print(f"  stati che ne contengono almeno uno: {n_hit}/{len(cases)}\n")

    def report(name, states):
        mutants = [Case(id=c.id, state=x, questions=c.questions, gold=c.gold)
                   for c, x in zip(cases, states)]
        adv = evaluate(eng, mutants, batch_size=16, chunk=32)
        ab = {r["id"]: r for r in adv.records}
        ok = flips = corr = 0
        conf, oer = [], 0
        for c in mutants:
            for qid, g in c.gold.items():
                b, a = cb[c.id]["labels"][qid], ab[c.id]["labels"][qid]
                if a == g:
                    ok += 1
                if b == g:
                    corr += 1
                    if a != g:
                        flips += 1
                        cf = ab[c.id]["confidence"][qid]
                        conf.append(cf)
                        oer += cf >= 0.75
        d_tok = sum(eng.count_tokens(x) - eng.count_tokens(c.state)
                    for c, x in zip(cases, states)) / len(cases)
        acc = ok / total
        mc = sum(conf) / len(conf) if conf else 0.0
        asr = flips / corr if corr else 0.0
        print(f"  {name:<40} Δtok {d_tok:>6.1f}  drop {(base_acc - acc) * 100:>6.1f}pp  "
              f"ASR {asr:>6.4f}  conf {mc:>5.3f}  OER {oer}  "
              f"ASR/tok {asr / d_tok if d_tok > 0 else 0:>7.5f}")
        return d_tok, asr

    print("A) multicarattere (digrammi -> carattere 'simile', dichiarato):")
    for dose in (0.50, 1.00):
        report(f"multicarattere dose {dose:.0%}",
               [sub_multi(c.state, dose, k) for k, c in enumerate(cases)])
    print()

    print("B) confronto economico a parita' di copertura:")
    base19 = build_tables()["A"]
    report("single-char, 19 lettere UTS, dose 50%",
           [sub_single(c.state, base19, 0.5, k) for k, c in enumerate(cases)])
    ext = dict(base19)
    ext.update(EXTRA_LETTERS)
    report(f"single-char, {len(ext)} lettere (+m,n,k), dose 50%",
           [sub_single(c.state, ext, 0.5, k) for k, c in enumerate(cases)])
    print()

    print("C) bidi override (previsione: si comporta come i zero-width):")
    for kind in ("RLO", "LRO", "LRI"):
        for dose in (0.50, 1.00):
            report(f"{kind} dose {dose:.0%}",
                   [inject_bidi(c.state, kind, dose, k) for k, c in enumerate(cases)])
    print()

    print("D) il collasso del vocabolario, per le due famiglie")
    for name, states in (
        ("multicarattere 100%", [sub_multi(c.state, 1.0, k) for k, c in enumerate(cases)]),
        ("RLO 100%", [inject_bidi(c.state, "RLO", 1.0, k) for k, c in enumerate(cases)]),
    ):
        adv = evaluate(eng, [Case(id=c.id, state=x, questions=c.questions, gold=c.gold)
                             for c, x in zip(cases, states)], batch_size=16, chunk=32)
        pc, pa = Counter(), Counter()
        for c in cases:
            for qid in c.gold:
                pc[cb[c.id]["labels"][qid]] += 1
        for r in adv.records:
            for qid in r["labels"]:
                pa[r["labels"][qid]] += 1
        print(f"  {name:<24} label {len(pc)} -> {len(pa)}, "
              f"sparite {sorted(lbl for lbl in pc if pa[lbl] == 0)}")

    print("\nE) il testo col RLO è identico a occhio?")
    withb = inject_bidi(s, "RLO", 1.0)
    print(f"  lunghezza {len(s)} -> {len(withb)} (+{len(withb) - len(s)})")
    print(f"  rimuovendo RLO torna l'originale: {withb.replace(RLO, '') == s}")
    print(f"  i codepoint dei caratteri sono invariati: "
          f"{[hex(ord(c)) for c in s[:12]] == [hex(ord(c)) for c in withb[:12]]}")

    # esempi tipografici dichiarati
    print("\nF) i multicarattere dichiarati sono davvero 2->1?")
    for dig, rep in MULTI.items():
        print(f"  {dig!r} ({len(dig)} char) -> {rep!r} ({len(rep)} char)  "
              f"{'contrazione' if len(rep) < len(dig) else 'stesso numero'}")
    eng.close()


if __name__ == "__main__":
    main()
