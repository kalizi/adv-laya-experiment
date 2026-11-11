"""Scorecard degli attacchi contro i criteri di vittoria, misurato.

Il livello "Bucato" chiede TRE soglie insieme. Due si possono soddisfare, la
terza (confidenza su quello che è sbagliato) è quella che il progetto chiama
"il modello è sicuro ed è torto: è quello che fa male".

Questo script calcola le tre soglie per ogni attacco, cosi' la domanda "ma
abbiamo detto che non passa?" ha una risposta misurata invece che ricordata.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from e12f_homoglyph_tiers import build_tables, substitute  # noqa: E402

from layaadv.data import typed_decisions as td  # noqa: E402
from layaadv.evaluate import evaluate  # noqa: E402
from layaadv.model.engine import DEFAULT_CHECKPOINT, LayaEngine  # noqa: E402
from layaadv.schema import Case  # noqa: E402

# soglie del livello "Bucato" (40-esperimenti/criteri-vittoria.md)
DROP_PP = 20.0
ASR_MIN = 0.50
CONF_MIN = 0.75


def scorecard(name, eng, cases, clean, cb, states):
    mutants = [Case(id=c.id, state=s, questions=c.questions, gold=c.gold)
               for c, s in zip(cases, states)]
    adv = evaluate(eng, mutants, batch_size=16, chunk=32)
    ab = {r["id"]: r for r in adv.records}

    base_ok = adv_ok = corr = flips = 0
    conf_wrong, oer = [], 0
    for c in mutants:
        for qid, g in c.gold.items():
            b = cb[c.id]["labels"][qid]
            a = ab[c.id]["labels"][qid]
            if b == g:
                base_ok += 1
            if a == g:
                adv_ok += 1
            if b == g and a != g:
                corr += 1
                flips += 1
                conf_wrong.append(ab[c.id]["confidence"][qid])
            if b == g and a != g and ab[c.id]["confidence"][qid] >= CONF_MIN:
                oer += 1

    base_acc = base_ok / sum(len(c.gold) for c in cases)
    adv_acc = adv_ok / sum(len(c.gold) for c in cases)
    drop_pp = (base_acc - adv_acc) * 100
    asr = flips / corr if corr else 0.0
    conf = sum(conf_wrong) / len(conf_wrong) if conf_wrong else 0.0

    c1 = drop_pp >= DROP_PP
    c2 = asr >= ASR_MIN
    c3 = conf >= CONF_MIN
    print(f"\n{name}")
    print(f"  baseline acc {base_acc:.4f} -> adv acc {adv_acc:.4f}   (drop {drop_pp:.1f}pp)")
    print(f"  ASR {asr:.4f}  ({flips} flip su {corr} decisioni corrette al pulito)")
    print(f"  confidenza media sul torto {conf:.4f}   (OER {oer} decisioni, cioe' "
          f"conf >= {CONF_MIN})")
    for label, ok, val, thr in (
        ("drop accuracy", c1, f"{drop_pp:.1f}pp", f">= {DROP_PP}pp"),
        ("ASR", c2, f"{asr:.4f}", f">= {ASR_MIN}"),
        ("confidenza sbagliate", c3, f"{conf:.4f}", f">= {CONF_MIN}"),
    ):
        print(f"    [{'X' if ok else ' '}] {label:<22} {val:>10}   soglia {thr}")
    print(f"  => {'BUCATO' if (c1 and c2 and c3) else 'NON passa il livello Bucato'}"
          f"  ({sum([c1, c2, c3])}/3 soglie)")
    return conf


def main():
    cases = td.load_cases("test", limit=30)
    eng = LayaEngine(DEFAULT_CHECKPOINT, batch_size=16)
    clean = evaluate(eng, cases, batch_size=16, chunk=32)
    cb = {r["id"]: r for r in clean.records}
    tables = build_tables()

    for name in ("A", "B", "C"):
        scorecard(f"omoglifi prova {name} (100%)", eng, cases, clean, cb,
                  [substitute(c.state, tables[name], 1.0, k) for k, c in enumerate(cases)])

    # E03 come riferimento: l'attacco che invece passa. I numeri sono dal vault,
    # qui si mostra solo che il criterio che manca agli omoglifi è quello che E03
    # soddisfa, non lo si ricalcola (richiederebbe l'opzione iniettata).
    print("\n" + "=" * 70)
    print("riferimento E03 (nominare un'opzione nello stato), dal vault:")
    print("  drop 37pp+, ASR 0.83-0.89, OER 0.85 a K=16  ->  3/3 soglie")
    print("  il criterio che agli omoglifi manca è proprio la confidenza")
    eng.close()


if __name__ == "__main__":
    main()
