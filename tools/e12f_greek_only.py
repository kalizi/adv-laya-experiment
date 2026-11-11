"""E12f-bis — set cross-script a 2 byte: invisibilita' e costo insieme.

E12f ha trovato che la prova A (confusabili UTS #39) costa 286 token dove la prova B
(greci visibili) ne costa 135, a parita' di ASR. La spiegazione proposta è la
lunghezza in byte: il set di A include cirillici a 3 byte (ҽ, Ӏ, ѡ) che il
tokenizer<byte> frammenta.

Se è così, un set di confusabili **solo greco, tutti a 2 byte**, deve pareggiare A
in costo mantenendo l'invisibilita'. È un test falsificabile in una riga: si
costruisce A filtrando per lunghezza UTF-8 e si misura.

Se non pareggia, l'ipotesi dei byte è falsa e la differenza viene da altro.
"""

import random
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from e12e_confusables import family_of, load_inverted  # noqa: E402

from layaadv.data import typed_decisions as td  # noqa: E402
from layaadv.evaluate import evaluate  # noqa: E402
from layaadv.model.engine import DEFAULT_CHECKPOINT, LayaEngine  # noqa: E402
from layaadv.schema import Case  # noqa: E402

N_CASES = 30
CONFUSABLES = Path(__file__).parent / "data" / "confusables.txt"


def official(seed=0):
    inv = load_inverted(CONFUSABLES)
    rng = random.Random(seed)
    out = {}
    for proto, sources in inv.items():
        keep = [
            c for c in sorted(sources)
            if c != proto and unicodedata.category(c).startswith("L")
            and family_of(c) in ("GREEK", "CYRILLIC")
        ]
        if keep:
            out[proto] = rng.choice(keep)
    return out


def greek_only(table):
    """Solo i sostituti greci, che sono tutti a 2 byte UTF-8."""
    return {k: v for k, v in table.items() if family_of(v) == "GREEK"}


def substitute(state, table, dose, seed=0):
    rng = random.Random(seed)
    pos = [i for i, ch in enumerate(state) if ch in table]
    if not pos:
        return state
    rng.shuffle(pos)
    chosen = set(pos[: max(1, int(len(pos) * dose))])
    return "".join(table[ch] if i in chosen else ch for i, ch in enumerate(state))


def main():
    a_all = official()
    a_greek = greek_only(a_all)
    print(f"prova A completa: {len(a_all)} lettere")
    print(f"prova A solo greco: {len(a_greek)} lettere — perse "
          f"{sorted(set(a_all) - set(a_greek))}")
    for k, v in sorted(a_greek.items()):
        print(f"  {k}->{v} ({len(v.encode())} byte)")

    cases = td.load_cases("test", limit=N_CASES)
    eng = LayaEngine(DEFAULT_CHECKPOINT, batch_size=16)
    clean = evaluate(eng, cases, batch_size=16, chunk=32)
    cb = {r["id"]: r for r in clean.records}
    corr = sum(
        1 for c in cases for qid, g in c.gold.items() if cb[c.id]["labels"][qid] == g
    )
    print(f"\nbaseline acc {clean.metrics['accuracy']:.4f}, {corr} decisioni corrette\n")

    def report(name, table):
        for dose in (0.25, 0.50, 1.00):
            states = [substitute(c.state, table, dose, k) for k, c in enumerate(cases)]
            mutants = [Case(id=c.id, state=s, questions=c.questions, gold=c.gold)
                       for c, s in zip(cases, states)]
            adv = evaluate(eng, mutants, batch_size=16, chunk=32)
            ab = {r["id"]: r for r in adv.records}
            flips = conf = oer = 0
            for c in mutants:
                for qid, g in c.gold.items():
                    if cb[c.id]["labels"][qid] == g and ab[c.id]["labels"][qid] != g:
                        flips += 1
                        cf = ab[c.id]["confidence"][qid]
                        conf += cf
                        oer += cf >= 0.75
            d_tok = sum(eng.count_tokens(s) - eng.count_tokens(c.state)
                        for c, s in zip(cases, states)) / len(cases)
            asr = flips / corr
            mc = conf / flips if flips else 0.0
            print(f"  {name} dose {dose:>4.0%}  Δtok {d_tok:>6.0f}  flip {flips:>3}  "
                  f"ASR {asr:>6.4f}  conf {mc:>5.3f}  OER {oer}  "
                  f"ASR/tok {asr / d_tok if d_tok > 0 else 0:>7.5f}")
        print()

    report("A completa (greco+cirillico)", a_all)
    report("A solo greco (2 byte)      ", a_greek)

    eng.close()


if __name__ == "__main__":
    main()
