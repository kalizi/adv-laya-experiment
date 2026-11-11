"""E12f — omoglifi per livelli di confusabilità visiva.

E12e (prova A) ha usato i confusabili che UTS #39 dichiara confondibili: sono la
prova A, l'attacco piu' furtivo perche' un umano non li distingue. Ma l'ipotesi
originale è piu' larga di UTS #39, e la prova B la testa: anche un carattere
**visibile a occhio** segue la stessa linea di principio, perche' il principio è
"cambiare lo script", non "imitare la forma".

Il confronto risponde a una domanda precisa: **la confusabilita' visiva predice
l'efficacia dell'attacco?**

  A  confusabile per UTS #39   (greco/cirillico)  -> invisibile
  B  greco plausibile, NON confusabile per lo standard -> visibile
  C  greco senza analogo latino (phi, theta, psi, xi) -> Obviously different, CONTROLLO

Se A ~ B ~ C, il modello è ingannato dal **cambio di script**, non dalla forma:
l'invisibilita' è gratis. Se A > B > C, la somiglianza conta e l'invisibilita'
ha un prezzo. Se B > A, il byte diverso disturba piu' della forma simile.

Le lettere coperte dalle tre tabelle non coincidono: si riporta la copertura e,
dove esiste, un confronto **within-letter** (stessa lettera latina, sostituto
confusabile contro sostituto plausibile ma non confusabile).
"""

import random
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from e12e_confusables import family_of, load_inverted  # noqa: E402

from layaadv import io
from layaadv.data import typed_decisions as td  # noqa: E402
from layaadv.evaluate import evaluate  # noqa: E402
from layaadv.model.engine import DEFAULT_CHECKPOINT, LayaEngine  # noqa: E402
from layaadv.schema import Case  # noqa: E402

N_CASES = 30
CONFUSABLES = Path(__file__).parent / "data" / "confusables.txt"

# Prova B: greci plausibili che UTS #39 NON dichiara confusibili con quella
# lettera latina. La scelta è dichiarata (plausibilita' di forma a occhio), non
# normativa: è esattamente il punto della prova B.
TIER_B = {
    "b": "β", "e": "ε", "k": "κ", "l": "λ", "m": "μ",
    "n": "η", "t": "τ", "x": "χ", "z": "ζ",
}
# Prova C: greci senza alcun analogo latino. Controllo.
TIER_C_POOL = "φθψξωΘΨΞΩ"


def official_confusables() -> dict[str, list[str]]:
    """Come in E12e: cross-script greco/cirillico per ogni lettera latina."""
    inv = load_inverted(CONFUSABLES)
    out: dict[str, list[str]] = {}
    for proto, sources in inv.items():
        keep = [
            ch
            for ch in sorted(sources)
            if ch != proto
            and unicodedata.category(ch).startswith("L")
            and family_of(ch) in ("GREEK", "CYRILLIC")
        ]
        if keep:
            out[proto] = keep
    return out


def build_tables(seed: int = 0):
    off = official_confusables()
    rng = random.Random(seed)
    a = {k: rng.choice(v) for k, v in off.items()}
    b = {k: v for k, v in TIER_B.items()}
    c = {k: rng.choice(TIER_C_POOL) for k in off}
    return {"A": a, "B": b, "C": c}


def substitute(state: str, table: dict[str, str], dose: float, seed: int = 0) -> str:
    rng = random.Random(seed)
    pos = [i for i, ch in enumerate(state) if ch in table]
    if not pos:
        return state
    rng.shuffle(pos)
    chosen = set(pos[: max(1, int(len(pos) * dose))])
    return "".join(table[ch] if i in chosen else ch for i, ch in enumerate(state))


def main():
    tables = build_tables()
    for name in ("A", "B", "C"):
        print(f"prova {name}: {len(tables[name])} lettere  "
              f"{' '.join(f'{k}->{v}' for k, v in sorted(tables[name].items()))}")
    shared = sorted(set(tables["A"]) & set(tables["B"]))
    print(f"\nlettere coperte da A e B insieme (confronto within-letter): {shared}")
    for k in shared:
        print(f"  {k}: A={tables['A'][k]!r}  B={tables['B'][k]!r}")

    cases = td.load_cases("test", limit=N_CASES)
    eng = LayaEngine(DEFAULT_CHECKPOINT, batch_size=16)
    clean = evaluate(eng, cases, batch_size=16, chunk=32)
    cb = {r["id"]: r for r in clean.records}
    total = sum(len(c.gold) for c in cases)
    print(f"\nbaseline acc {clean.metrics['accuracy']:.4f} su {total} decisioni\n")

    rows = []

    def report(name, states):
        mutants = [
            Case(id=c.id, state=s, questions=c.questions, gold=c.gold)
            for c, s in zip(cases, states)
        ]
        adv = evaluate(eng, mutants, batch_size=16, chunk=32)
        ab = {r["id"]: r for r in adv.records}
        flips = corr = 0
        conf, oer = [], 0
        for c in mutants:
            for qid, g in c.gold.items():
                b = cb[c.id]["labels"][qid]
                a = ab[c.id]["labels"][qid]
                if b == g:
                    corr += 1
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
        eff = asr / d_tok if d_tok > 0 else float("nan")
        print(
            f"  {name:<30} Δtok {d_tok:>6.0f}  flip {flips:>3}  ASR {asr:>6.4f}  "
            f"conf {mc:>5.3f}  OER {oer}  ASR/tok {eff:>7.5f}"
        )
        # `level` e `dose` sono colonne, non testo: senza, "prova A dose 100%" e
        # "prova B dose 25%" sono la stessa stringa da parsare e una tabella dose non
        # e' confrontabile con una tabella livello.
        parts = name.split()
        rows.append({
            "probe": name,
            "level": parts[1] if parts and parts[0] == "prova" else None,
            "dose": float(parts[3].rstrip("%")) / 100 if len(parts) > 3 and parts[3].endswith("%") else None,
            "delta_tokens_mean": d_tok,
            "flips": flips,
            "n_decisions": total,
            "asr": asr,
            "asr_denominator": corr,
            "mean_conf_on_wrong": mc,
            "oer_adv_conf_075": oer,
            "asr_per_token": eff,
        })
        return asr, d_tok

    print("curva di dose per livello:")
    for name in ("A", "B", "C"):
        for dose in (0.25, 0.50, 1.00):
            report(
                f"prova {name} dose {dose:.0%}",
                [substitute(c.state, tables[name], dose, k) for k, c in enumerate(cases)],
            )
        print()

    print("confronto within-letter (stessa lettera, due tipi di sostituto):")
    for k in shared:
        for name in ("A", "B"):
            sub = {k: tables[name][k]}
            report(
                f"{name}: solo {k}->{tables[name][k]}",
                [substitute(c.state, sub, 1.0, j) for j, c in enumerate(cases)],
            )
    print()

    print("collasso del vocabolario, per livello:")
    for name in ("A", "B", "C"):
        states = [substitute(c.state, tables[name], 1.0, k) for k, c in enumerate(cases)]
        adv = evaluate(
            eng,
            [Case(id=c.id, state=s, questions=c.questions, gold=c.gold) for c, s in zip(cases, states)],
            batch_size=16, chunk=32,
        )
        pred = Counter()
        for r in adv.records:
            for qid in r["labels"]:
                pred[r["labels"][qid]] += 1
        pred_clean = Counter()
        for c in cases:
            for qid in c.gold:
                pred_clean[cb[c.id]["labels"][qid]] += 1
        lost = sorted(lbl for lbl in pred_clean if pred[lbl] == 0)
        print(f"  prova {name}: {len(pred_clean)} -> {len(pred)} label, sparite {lost}")

    eng.close()

    # Fino al 2026-10-05 questa curva non finiva su disco: la nota la riportava e il
    # controllo numerico la segnalava come non rintracciabile. Il file adesso porta anche
    # `level` e `dose` separati, perche' la conclusione della serie ("la somiglianza non
    # predice l'efficacia") e' un confronto fra livelli a dose piena, e mescolarlo con le
    # dosi parziali la renderebbe non verificabile.
    payload = {
        "experiment": "E12f_homoglyph_tiers",
        "question": "la somiglianza visibile predice l'efficacia dell'attacco?",
        "axes": {
            "checkpoint": DEFAULT_CHECKPOINT,
            "n_cases": len(cases),
            "seed": 0,
            "levels": ["A", "B", "C"],
            "doses": [0.25, 0.50, 1.00],
            "study_valid": True,
        },
        "study_valid": {
            "why": "la sostituzione ha firma (state) -> str: non vede questions ne gold",
            "verified_per_row": "questions_untouched",
        },
        "results": rows,
    }
    io.write_result("E12f_homoglyph_tiers", payload)
    print(f"\n[E12f] wrote results/E12f_homoglyph_tiers.json ({len(rows)} righe)")


if __name__ == "__main__":
    main()
