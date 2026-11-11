"""E12i — underscore fra le parole: un attacco in-distribuzione.

Tutti gli attacchi finora erano fuori distribuzione: greco, cirillico, filler,
sinonimi. L'underscore è diverso, perche' **è gia' nel dato**: gli stati hanno
`duration_s`, `constraint_violations`, `trace_summary`. Inserire `_` fra le
parole produce testo che sembra nativo, ed è ASCII a 1 byte, quindi un
carattere innocuo dal punto di vista del costo.

La domanda è se l'in-distribuzione cambia qualcosa. Se il modello regge, la
superficie lessicale è chiusa anche per l'attacco piu' economico e piu' plausibile
che si possa scrivere. Se cede come gli altri, allora è la tokenizzazione a fare
il danno e la distribuzione non c'entra.

Braccia:
  A  `_` fra le parole, curva di dose
  B  controlli caratteri ASCII nella stessa posizione: `-` `,` `.` `@`
  C  riferimenti gia' misurati: spazio (E12c) e omoglifi (E12e)

Solo lo stato viene toccato, e gli underscore finiscono dentro le stringhe JSON,
quindi il JSON resta valido. Le cifre non vengono mai spezzate: l'attacco è fra
parole, non dentro i numeri.
"""

import random
from collections import Counter

from layaadv import io
from layaadv.data import typed_decisions as td
from layaadv.evaluate import evaluate
from layaadv.model.engine import DEFAULT_CHECKPOINT, LayaEngine
from layaadv.schema import Case

N_CASES = 30
FILLER_CONTROL = "_"
ASCII_CONTROL = ["-", ",", ".", "@", "/"]


def insert_between_words(state: str, char: str, dose: float, seed: int = 0) -> str:
    """Inserisce `char` davanti a una frazione degli spazi singoli dello stato.

    Gli stati sono JSON serializzato compatto: **non c'è nessuno spazio doppio**
    (misurato: 0 su 30 stati), quindi i punti d'inserimento sono gli spazi
    singoli. Sostituire ", " con ", char " non tocca numeri né contenuto: il
    carattere finisce dentro la stringa JSON, che resta valida.

    Nota sul controllo: il tokenizer **collassa gli spazi ripetuti** (32 spazi
    costano 2 token), quindi il braccio "spazio" di questa sonda non puo' essere
    confrontato con i numeri di E12c, dove il padding era in coda allo stato.
    """
    rng = random.Random(seed)
    out = []
    for ch in state:
        if ch == " " and rng.random() < dose:
            out.append(char)
        out.append(ch)
    return "".join(out)


def main():
    cases = td.load_cases("test", limit=N_CASES)
    eng = LayaEngine(DEFAULT_CHECKPOINT, batch_size=16)
    clean = evaluate(eng, cases, batch_size=16, chunk=32)
    cb = {r["id"]: r for r in clean.records}
    total = sum(len(c.gold) for c in cases)
    print(f"max_len {eng.cfg.get('max_len')}, stati puliti "
          f"{min(eng.count_tokens(c.state) for c in cases)}-"
          f"{max(eng.count_tokens(c.state) for c in cases)} token")
    print(f"baseline acc {clean.metrics['accuracy']:.4f} su {total} decisioni")
    n_us = sum(c.state.count("_") for c in cases)
    print(f"underscore gia' presenti negli stati puliti: {n_us} su {len(cases)} stati")
    print(f"stati che ne contengono: {sum(1 for c in cases if '_' in c.state)}/{len(cases)}\n")

    rows = []

    def report(name, states):
        mutants = [Case(id=c.id, state=s, questions=c.questions, gold=c.gold)
                   for c, s in zip(cases, states)]
        adv = evaluate(eng, mutants, batch_size=16, chunk=32)
        ab = {r["id"]: r for r in adv.records}
        ok = flips = corr = 0
        conf, oer = [], 0
        pred = Counter()
        for c in mutants:
            for qid, g in c.gold.items():
                b = cb[c.id]["labels"][qid]
                a = ab[c.id]["labels"][qid]
                pred[a] += 1
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
        print(f"  {name:<30} Δtok {d_tok:>6.1f}  acc {acc:.4f} "
              f"({(clean.metrics['accuracy'] - acc) * 100:>5.1f}pp)  "
              f"ASR {asr:>6.4f}  conf {mc:>5.3f}  OER {oer}  "
              f"ASR/tok {asr / d_tok if d_tok > 0 else 0:>7.5f}  label {len(pred)}")
        rows.append({
            "probe": name,
            # Il nome e' del tipo `_ dose 25%`, quindi la dose non e' il token dopo il
            # separatore: si cerca l'unico token che finisce con `%`, altrimenti la colonna
            # resta `None` su tutte le righe e la tabella dose sembra non esistere.
            "dose": next((float(t.rstrip("%")) / 100 for t in name.split() if t.endswith("%")), None),
            "delta_tokens_mean": d_tok,
            "flips": flips,
            "n_decisions": total,
            "asr": asr,
            "asr_denominator": corr,
            "mean_conf_on_wrong": mc,
            "oer_adv_conf_075": oer,
            "distinct_labels_predicted": len(pred),
        })
        return asr, d_tok, len(pred)

    print("A) underscore fra le parole, curva di dose:")
    for dose in (0.10, 0.25, 0.50, 1.00):
        report(f"_ dose {dose:.0%}",
               [insert_between_words(c.state, FILLER_CONTROL, dose, k) for k, c in enumerate(cases)])
    print()

    print("B) altri caratteri ASCII nella stessa posizione (dose 100%):")
    for ch in ASCII_CONTROL:
        report(f"{ch!r} dose 100%", [insert_between_words(c.state, ch, 1.0, k) for k, c in enumerate(cases)])
    print()

    print("C) riferimenti: spazio e omoglifi greci, a parita' di costo in token.")
    print("   (lo spazio va in coda: il tokenizer collassa i doppi spazi, quindi")
    print("    il braccio 'fra le parole' non è confrontabile con E12c)")
    for n in (10, 30, 100):
        report(f"spazi in coda x{n}", [c.state + " " * n for c in cases])
    try:
        from e12f_homoglyph_tiers import build_tables, substitute
        t = build_tables()
        report("omoglifi prova A", [substitute(c.state, t["A"], 1.0, k) for k, c in enumerate(cases)])
    except Exception as exc:
        print(f"  (omoglifi non disponibili: {exc})")

    print("\nD) collasso del vocabolario, underscore 100%:")
    pred_clean = Counter()
    for c in cases:
        for qid in c.gold:
            pred_clean[cb[c.id]["labels"][qid]] += 1
    states = [insert_between_words(c.state, FILLER_CONTROL, 1.0, k) for k, c in enumerate(cases)]
    adv = evaluate(eng, [Case(id=c.id, state=s, questions=c.questions, gold=c.gold)
                         for c, s in zip(cases, states)], batch_size=16, chunk=32)
    pred_adv = Counter()
    for r in adv.records:
        for qid in r["labels"]:
            pred_adv[r["labels"][qid]] += 1
    print(f"  label distinte {len(pred_clean)} -> {len(pred_adv)}, "
          f"sparite {sorted(lbl for lbl in pred_clean if pred_adv[lbl] == 0)}")

    eng.close()

    # Stessa ragione di E12d ed E12f: la curva di dose era su console e quindi non
    # rintracciabile. `dose` è una colonna perché le righe non sono a dose uguale, e
    # `asr_denominator` perché un tasso senza denominatore non è confrontabile.
    payload = {
        "experiment": "E12i_underscore",
        "question": "un carattere in-distribuzione che non costa token attacca?",
        "axes": {
            "checkpoint": DEFAULT_CHECKPOINT,
            "n_cases": len(cases),
            "seed": 0,
            "doses": [0.10, 0.25, 0.50, 1.00],
            "study_valid": True,
        },
        "study_valid": {
            "why": "l'inserimento ha firma (state) -> str: non vede questions ne gold",
            "verified_per_row": "questions_untouched",
        },
        "results": rows,
    }
    io.write_result("E12i_underscore_dose", payload)
    print(f"\n[E12i] wrote results/E12i_underscore_dose.json ({len(rows)} righe)")


if __name__ == "__main__":
    main()
