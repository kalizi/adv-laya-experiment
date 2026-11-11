"""E12l — caratteri a larghezza zero: invisibili davvero.

Motivazione: nella letteratura i zero-width (U+200B/C/D) danno 54.2% di successo e
frammentano le parole *dentro*, che è esattamente il meccanismo che E12d ha
misurato come leva (spazi distribuiti ≈ omoglifi, ASR 0.6726). Sono l'unica
perturbazione che unisce l'efficacia della corruzione in place con
l'invisibilita' assoluta: larghezza zero, nessun rendering, quindi un revisore
umano non li vede piu' di quanto non vede un carattere.

Prima di tutto un test che chiude la porta: **il tokenizer li normalizza via?**
Se NFKC li collassa, l'attacco non esiste e inutile misurarlo.
"""

import random
import unicodedata
from collections import Counter

from layaadv.data import typed_decisions as td
from layaadv.evaluate import evaluate
from layaadv.model.engine import DEFAULT_CHECKPOINT, LayaEngine
from layaadv.schema import Case

N_CASES = 30

# Sequenze di escape e NON letterali: un carattere a larghezza zero scritto
# direttamente nel sorgente sparisce in modo silenzioso (misurato: la entry
# "ZWSP U+200B" conteneva uno spazio ASCII, e il risultato sembrava dire che il
# tokenizer elimina lo ZWSP. Non era vero: non era mai stato testato).
ZW = {
    "ZWSP U+200B": "\u200b",
    "ZWNJ U+200C": "\u200c",
    "ZWJ U+200D": "\u200d",
    "FEFF U+FEFF": "\ufeff",
    "WORD JOINER U+2060": "\u2060",
}


def main():
    eng = LayaEngine(DEFAULT_CHECKPOINT, batch_size=16)
    cases = td.load_cases("test", limit=N_CASES)
    clean = evaluate(eng, cases, batch_size=16, chunk=32)
    cb = {r["id"]: r for r in clean.records}
    total = sum(len(c.gold) for c in cases)
    base_acc = clean.metrics["accuracy"]
    sample = "Rotate the expired TLS certificate"

    print(f"baseline acc {base_acc:.4f} su {total} decisioni")
    print(f"stati {min(eng.count_tokens(c.state) for c in cases)}-"
          f"{max(eng.count_tokens(c.state) for c in cases)} token\n")

    # --- test 1: il tokenizer li normalizza via?
    print("=== 1. sopravvivenza al tokenizer ===")
    t0 = eng.count_tokens(sample)
    print(f"  testo pulito: {t0} token")
    for name, ch in ZW.items():
        ins = sample[:6] + ch + sample[6:]
        n1 = eng.count_tokens(ins)
        ins3 = ch.join(list(sample[:12]))  # uno ogni carattere
        n3 = eng.count_tokens(ins3)
        # NFKC/NFKC del testo: se collassa, un normalizzatore a valle lo cancellerebbe
        nfk = unicodedata.normalize("NFKC", ins3)
        survived = nfk.count(ch)
        print(f"  {name:<20} uno in mezzo: {n1:>3} tok (+{n1 - t0})   "
              f"uno ogni 12 char: {n3:>4} tok (+{n3 - t0})   "
              f"NFKC ne lascia: {survived}")

    # il tokenizer di laya normalizza? confronto diretto
    print("\n  il tokenizer di laya collassa i zero-width?")
    for name, ch in ZW.items():
        ids = eng._agent.tok.encode(sample[:6] + ch + sample[6:])
        toks = eng._agent.tok.convert_ids_to_tokens(ids)
        print(f"    {name:<20} {toks}")

    # --- test 2: attacco vero
    def inject_inside(state, ch, dose, seed=0):
        """Inserisce il carattere dentro le parole, alle posizioni alfabetiche."""
        rng = random.Random(seed)
        pos = [i for i, c in enumerate(state) if c.isalpha()]
        if not pos:
            return state
        rng.shuffle(pos)
        keep = set(pos[: max(1, int(len(pos) * dose))])
        return "".join(ch + c if i in keep else c for i, c in enumerate(state))

    def inject_between(state, ch, dose, seed=0):
        rng = random.Random(seed)
        out = []
        for c in state:
            if c == " " and rng.random() < dose:
                out.append(ch)
            out.append(c)
        return "".join(out)

    def report(name, states):
        mutants = [Case(id=c.id, state=s, questions=c.questions, gold=c.gold)
                   for c, s in zip(cases, states)]
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
        d_tok = sum(eng.count_tokens(s) - eng.count_tokens(c.state)
                    for c, s in zip(cases, states)) / len(cases)
        acc = ok / total
        mc = sum(conf) / len(conf) if conf else 0.0
        asr = flips / corr if corr else 0.0
        print(f"  {name:<38} Δtok {d_tok:>6.1f}  drop {(base_acc - acc) * 100:>6.1f}pp  "
              f"ASR {asr:>6.4f}  conf {mc:>5.3f}  OER {oer}  "
              f"ASR/tok {asr / d_tok if d_tok > 0 else 0:>7.5f}")
        return d_tok

    print("\n=== 2. curva di dose, dentro le parole ===")
    for name, ch in ZW.items():
        for dose in (0.10, 0.25, 0.50):
            report(f"{name} dose {dose:.0%}",
                   [inject_inside(c.state, ch, dose, k) for k, c in enumerate(cases)])

    print("\n=== 3. fra le parole (controllo di posizione) ===")
    for dose in (0.25, 0.50, 1.00):
        report(f"ZWSP fra le parole dose {dose:.0%}",
               [inject_between(c.state, ZW["ZWSP U+200B"], dose, k) for k, c in enumerate(cases)])

    print("\n=== 4. riferimenti a parita' di concetto ===")
    # stesso effetto (frammentare le parole) ma con un carattere NON invisibile
    report("spazi dentro le parole dose 50% (NON invisibile)",
           [inject_inside(c.state, " ", 0.5, k) for k, c in enumerate(cases)])
    report("sotto-griglia: '_' dentro le parole dose 50%",
           [inject_inside(c.state, "_", 0.5, k) for k, c in enumerate(cases)])

    print("\n=== 5. collasso del vocabolario, ZWSP 50% dentro ===")
    states = [inject_inside(c.state, ZW["ZWSP U+200B"], 0.5, k) for k, c in enumerate(cases)]
    adv = evaluate(eng, [Case(id=c.id, state=s, questions=c.questions, gold=c.gold)
                         for c, s in zip(cases, states)], batch_size=16, chunk=32)
    pc, pa = Counter(), Counter()
    for c in cases:
        for qid in c.gold:
            pc[cb[c.id]["labels"][qid]] += 1
    for r in adv.records:
        for qid in r["labels"]:
            pa[r["labels"][qid]] += 1
    print(f"  label distinte {len(pc)} -> {len(pa)}, "
          f"sparite {sorted(lbl for lbl in pc if pa[lbl] == 0)}")

    print("\n=== 6. il testo è davvero identico a occhio? ===")
    s = cases[0].state
    a = inject_inside(s, ZW["ZWSP U+200B"], 1.0)
    print(f"  lunghezza {len(s)} -> {len(a)} caratteri")
    print(f"  identico se si rimuovono i zero-width: "
          f"{a.replace(ZW['ZWSP U+200B'], '') == s}")
    print(f"  i numeri sono intatti: {'50' in a} / {'32.5' in a}")
    eng.close()


if __name__ == "__main__":
    main()
