"""E12e — omoglifi con la tabella normativa UTS #39 invece di una scritta a mano.

E12d ha usato una tabella scritta a mano e ha misurato ASR 0.62-0.76. Ma la
tesi dell'attacco (indistinguibile per un umano) poggiava su di essa, cioe' su
quello che avevo scritto io. Verifico con UTS #39, che è il dato normativo usato
da browser e IDN, e confronto le due tabelle sulla stessa procedura.

Se la tabella normativa funziona quanto quella a mano, il risultato di E12d
regge su dato e non su assunzione. Se funziona peggio, allora una parte dei
numeri di E12d veniva da sostituti che un umano NON confonderebbe, e va detto.

Oltre al confronto, qui si separa cross-script (greco/cirillico) da same-script
(matematico/fullwidth): E12d li aveva mescolati, e la pretesa è diversa.

Uso: uv run python tools/e12e_homoglyph_uts.py
"""

import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from e12e_confusables import load  # noqa: E402

from layaadv.data import typed_decisions as td  # noqa: E402
from layaadv.evaluate import evaluate  # noqa: E402
from layaadv.model.engine import DEFAULT_CHECKPOINT, LayaEngine  # noqa: E402
from layaadv.schema import Case  # noqa: E402

N_CASES = 30


def build_map(cross: dict[str, list[str]], rng_seed: int = 0) -> dict[str, str]:
    """Un omoglifo per lettera, scelto in modo deterministico fra quelli ufficiali."""
    rng = random.Random(rng_seed)
    out: dict[str, str] = {}
    for proto, choices in cross.items():
        if choices:
            out[proto] = rng.choice(choices)
    return out


def substitute(state: str, table: dict[str, str], dose: float, seed: int = 0) -> str:
    """Sostituisce un `dose` delle lettere presenti nella tabella."""
    rng = random.Random(seed)
    pos = [i for i, ch in enumerate(state) if ch in table]
    if not pos:
        return state
    rng.shuffle(pos)
    chosen = set(pos[: max(1, int(len(pos) * dose))])
    return "".join(table[ch] if i in chosen else ch for i, ch in enumerate(state))


def spaces_distributed(state: str, n: int, seed: int = 0) -> str:
    """Controllo E12d: stessi caratteri toccati, ma inerti."""
    rng = random.Random(seed)
    pos = [i for i, ch in enumerate(state) if ch.isalpha()]
    if not pos:
        return state
    rng.shuffle(pos)
    chosen = set(pos[:n])
    return "".join(" " if i in chosen else ch for i, ch in enumerate(state))


def main():
    tables = load()
    cross_map = build_map(tables["cross"])
    same_map = build_map(tables["same"])
    print(f"tabella UTS #39 in uso: {len(cross_map)} lettere cross-script, "
          f"{len(same_map)} same-script")
    print(f"  cross: {' '.join(f'{k}->{v}' for k, v in sorted(cross_map.items()))}")
    print()

    cases = td.load_cases("test", limit=N_CASES)
    eng = LayaEngine(DEFAULT_CHECKPOINT, batch_size=16)
    clean = evaluate(eng, cases, batch_size=16, chunk=32)
    cb = {r["id"]: r for r in clean.records}
    again = evaluate(eng, cases, batch_size=16, chunk=32)
    same = sum(
        1
        for x, y in zip(clean.records, again.records)
        for qid in x["labels"]
        if x["labels"][qid] != y["labels"][qid]
    )
    total = sum(len(c.gold) for c in cases)
    print(f"[1] determinismo {same}/{total}, baseline acc {clean.metrics['accuracy']:.4f}\n")

    def report(name, states):
        mutants = [
            Case(id=c.id, state=s, questions=c.questions, gold=c.gold)
            for c, s in zip(cases, states)
        ]
        adv = evaluate(eng, mutants, batch_size=16, chunk=32)
        ab = {r["id"]: r for r in adv.records}
        diff = flips = corr = 0
        conf, oer = [], 0
        for c in mutants:
            for qid, g in c.gold.items():
                b = cb[c.id]["labels"][qid]
                a = ab[c.id]["labels"][qid]
                if b == g:
                    corr += 1
                if a != b:
                    diff += 1
                if b == g and a != g:
                    flips += 1
                    cf = ab[c.id]["confidence"][qid]
                    conf.append(cf)
                    oer += cf >= 0.75
        d_tok = sum(
            eng.count_tokens(s) - eng.count_tokens(c.state) for c, s in zip(cases, states)
        ) / len(cases)
        mc = sum(conf) / len(conf) if conf else 0.0
        asr = flips / corr if corr else 0.0
        print(
            f"  {name:<34} Δtok {d_tok:>6.0f}  rotte {diff:>3}/{total}  "
            f"flip {flips:>3}  ASR {asr:>6.4f}  conf {mc:>5.3f}  OER {oer}"
        )
        return d_tok, asr

    print("[2] curva di dose, tabella NORMATIVA cross-script:")
    for dose in (0.10, 0.25, 0.50, 1.00):
        report(
            f"UTS cross dose {dose:.0%}",
            [substitute(c.state, cross_map, dose, k) for k, c in enumerate(cases)],
        )
    print()

    print("[3] cross-script contro same-script (stessa dose, 100%):")
    report(
        "UTS cross (greco/cirillico)",
        [substitute(c.state, cross_map, 1.0, k) for k, c in enumerate(cases)],
    )
    report("UTS same (math/fullwidth)", [substitute(c.state, same_map, 1.0, k) for k, c in enumerate(cases)])
    print()

    print("[4] controllo del meccanismo (spazi distribuiti, come E12d):")
    n = int(sum(len(c.state) for c in cases) / len(cases))
    report("UTS cross 100%", [substitute(c.state, cross_map, 1.0, k) for k, c in enumerate(cases)])
    report(f"spazi distribuiti ~{n} char", [spaces_distributed(c.state, n, k) for k, c in enumerate(cases)])
    print()

    print("[5] collasso del vocabolario (cross 100%):")
    pred_clean, pred_adv = Counter(), Counter()
    for c in cases:
        for qid in c.gold:
            pred_clean[cb[c.id]["labels"][qid]] += 1
    states = [substitute(c.state, cross_map, 1.0, k) for k, c in enumerate(cases)]
    adv = evaluate(
        eng,
        [Case(id=c.id, state=s, questions=c.questions, gold=c.gold) for c, s in zip(cases, states)],
        batch_size=16,
        chunk=32,
    )
    for r in adv.records:
        for qid in r["labels"]:
            pred_adv[r["labels"][qid]] += 1
    lost = sorted(k for k in pred_clean if pred_adv[k] == 0)
    print(f"    label distinte: pulito {len(pred_clean)}, attacco {len(pred_adv)}")
    print(f"    sparite: {lost}")

    eng.close()


if __name__ == "__main__":
    main()
