"""E12d — attacco character-level a omoglifi (Unicode confusables).

Domanda: sostituire le lettere con equivalenti Unicode visivamente identici
(greco, cirillico, matematiche) inganna il modello senza che un umano se ne
accorga?

La misura che rende la domanda interessante è che il tokenizer è byte-level e
non riconosce niente fuori dal latino: `Rotate the expired TLS certificate`
passa da 8 a 29 token con gli omoglifi greci, inflazione 3.6x.

Ma il risultato importante è un altro, e arriva dai controlli: l'effetto NON è
dello script e NON è del budget. È **corruzione in place** del contenuto.
Sostituire lettere con spazi, che è la perturbazione piu' invisibile e più
economica possibile, ottiene lo stesso ASR (o di più) spendendo 10x meno token.

Quindi l'omoglifo non è un canale di attacco nuovo: è il modo per ottenere una
corruzione in place che un revisore umano non vede. E il modello non reagisce
andando verso un bersaglio, collassa: mette' delle sue label spariscono del tutto.

Controlli eseguiti, nell'ordine in cui servono:
  1. determinismo: clean contro clean (0/150 atteso)
  2. costo in token per ogni alfabeto
  3. curva di dose
  4. meccanismo: spazi distribuiti vs omoglifi vs filler in coda
  5. collasso: quante label distinte emette il modello

Uso: uv run python tools/e12d_homoglyph.py
"""

import random
from collections import Counter

from layaadv import io
from layaadv.data import typed_decisions as td
from layaadv.evaluate import evaluate
from layaadv.model.engine import DEFAULT_CHECKPOINT, LayaEngine
from layaadv.schema import Case

# Confusables: lettere latine -> caratteri visivamente equivalenti. La lista è
# dichiarativa (confusables classici), non è misurata qui: cio' che è misurato
# è che il modello li tratta come caratteri diversi.
GREEK = {
    "a": "α", "b": "β", "c": "ϲ", "d": "ԁ", "e": "ε", "i": "ι", "k": "κ",
    "l": "λ", "m": "μ", "n": "η", "o": "ο", "p": "ρ", "r": "г", "s": "ѕ",
    "t": "τ", "u": "υ", "v": "ν", "x": "χ", "y": "γ", "z": "ζ",
    "A": "Α", "B": "Β", "C": "Ϲ", "E": "Ε", "I": "Ι", "K": "Κ", "M": "Μ",
    "N": "Ν", "O": "Ο", "P": "Ρ", "T": "Τ", "X": "Χ", "Y": "Υ",
}
CYRILLIC = {
    "a": "а", "b": "в", "c": "с", "e": "е", "h": "һ", "i": "і", "j": "ј",
    "k": "к", "m": "м", "n": "н", "o": "о", "p": "р", "r": "г", "s": "ѕ",
    "t": "т", "x": "х", "y": "у",
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "I": "І", "K": "К",
    "M": "М", "O": "О", "P": "Р", "S": "Ѕ", "T": "Т", "X": "Х", "Y": "У",
}
MATH = {
    "a": "𝐚", "b": "𝐛", "c": "𝐜", "d": "𝐝", "e": "𝐞", "f": "𝐟", "g": "𝐠",
    "h": "𝐡", "i": "𝐢", "j": "𝐣", "k": "𝐤", "l": "𝐥", "m": "𝐦", "n": "𝐧",
    "o": "𝐨", "p": "𝐩", "q": "𝐪", "r": "𝐫", "s": "𝐬", "t": "𝐭", "u": "𝐮",
    "v": "𝐯", "w": "𝐰", "x": "𝐱", "y": "𝐲", "z": "𝐳",
    "A": "𝐀", "B": "𝐁", "C": "𝐂", "D": "𝐃", "E": "𝐄", "I": "𝐈", "J": "𝐉",
    "K": "𝐊", "L": "𝐋", "M": "𝐌", "N": "𝐍", "O": "𝐎", "P": "𝐏", "R": "𝐑",
    "S": "𝐒", "T": "𝐓", "U": "𝐔", "V": "𝐕", "W": "𝐖", "X": "𝐗", "Y": "𝐘",
}
FILLER = "in merito si precisa come da note per quanto in coerenza con alla luce di"
N_CASES = 30


def homoglyph(state, table, dose, seed=0):
    """Sostituisce un `dose` (0..1) delle lettere mappabili con omoglifi."""
    rng = random.Random(seed)
    pos = [i for i, ch in enumerate(state) if ch in table]
    if not pos:
        return state
    rng.shuffle(pos)
    chosen = set(pos[: max(1, int(len(pos) * dose))])
    return "".join(table[ch] if i in chosen else ch for i, ch in enumerate(state))


def spaces_distributed(state, n, seed=0):
    """Stessi caratteri degli omoglifi, ma inerti, sparsi dove stanno le lettere."""
    rng = random.Random(seed)
    pos = [i for i, ch in enumerate(state) if ch.isalpha()]
    if not pos:
        return state
    rng.shuffle(pos)
    chosen = set(pos[:n])
    return "".join(" " if i in chosen else ch for i, ch in enumerate(state))


def filler_tail(state, n_tokens, engine, seed=0):
    """Parole di filler inerti in coda, fino al costo richiesto."""
    rng = random.Random(seed)
    words = FILLER.split()
    out, base = state, engine.count_tokens(state)
    while engine.count_tokens(out) - base < n_tokens:
        out = f"{out} {rng.choice(words)}"
    return out


def main():
    cases = td.load_cases("test", limit=N_CASES)
    eng = LayaEngine(DEFAULT_CHECKPOINT, batch_size=16)
    cfg = eng.cfg
    print(
        f"max_len {cfg.get('max_len')}, head_max_len {cfg.get('head_max_len')}, "
        f"stati puliti {min(eng.count_tokens(c.state) for c in cases)}-"
        f"{max(eng.count_tokens(c.state) for c in cases)} token"
    )

    clean = evaluate(eng, cases, batch_size=16, chunk=32)
    cb = {r["id"]: r for r in clean.records}

    # 1. determinismo: senza questo nessun delta piccolo è interpretabile
    again = evaluate(eng, cases, batch_size=16, chunk=32)
    same = sum(
        1
        for x, y in zip(clean.records, again.records)
        for qid in x["labels"]
        if x["labels"][qid] != y["labels"][qid]
    )
    total = sum(len(c.gold) for c in cases)
    print(f"[1] determinismo, clean contro clean: {same}/{total} label diverse")
    print(f"    baseline acc {clean.metrics['accuracy']:.4f}\n")

    rows = []

    def report(name, states):
        mutants = [
            Case(id=c.id, state=s, questions=c.questions, gold=c.gold)
            for c, s in zip(cases, states)
        ]
        adv = evaluate(eng, mutants, batch_size=16, chunk=32)
        ab = {r["id"]: r for r in adv.records}
        diff = flips = corr = 0
        conf, oer = [], 0
        pred = Counter()
        for c in mutants:
            for qid, g in c.gold.items():
                b = cb[c.id]["labels"][qid]
                a = ab[c.id]["labels"][qid]
                pred[a] += 1
                if b == g:
                    corr += 1
                if a != b:
                    diff += 1
                if b == g and a != g:
                    flips += 1
                    conf.append(ab[c.id]["confidence"][qid])
                    oer += ab[c.id]["confidence"][qid] >= 0.75
        d_tok = sum(
            eng.count_tokens(s) - eng.count_tokens(c.state) for c, s in zip(cases, states)
        ) / len(cases)
        mc = sum(conf) / len(conf) if conf else 0.0
        asr = flips / corr if corr else 0.0
        print(
            f"  {name:<28} Δtok {d_tok:>6.0f}  rotte {diff:>3}/{total}  "
            f"flip {flips:>3}  ASR {asr:>6.4f}  conf {mc:>5.3f}  OER {oer}  "
            f"label distinte {len(pred)}"
        )
        rows.append({
            "probe": name,
            "dose": next((d for d in (0.10, 0.25, 0.50, 1.00) if f"dose {d:.0%}" in name), None),
            "delta_tokens_mean": d_tok,
            "labels_changed": diff,
            "n_decisions": total,
            "flips": flips,
            "asr": asr,
            "asr_denominator": corr,
            "mean_conf_on_wrong": mc,
            "oer_adv_conf_075": oer,
            "distinct_labels_predicted": len(pred),
        })
        return d_tok, len(pred)

    # 2. costo per alfabeto
    print("[2] costo in token (stessa frase, tre alfabeti):")
    for name, table in (("greco", GREEK), ("cirillico", CYRILLIC), ("matematico", MATH)):
        t0 = eng.count_tokens("Rotate the expired TLS certificate")
        t1 = eng.count_tokens(homoglyph("Rotate the expired TLS certificate", table, 1.0))
        print(f"    {name:<11} {t0} -> {t1} token ({t1 / t0:.1f}x)")
    print()

    # 3. curva di dose
    print("[3] curva di dose:")
    for dose in (0.10, 0.25, 0.50, 1.00):
        report(f"greco dose {dose:.0%}", [homoglyph(c.state, GREEK, dose, k) for k, c in enumerate(cases)])
    for dose in (0.25, 1.00):
        report(
            f"cirillico dose {dose:.0%}",
            [homoglyph(c.state, CYRILLIC, dose, k) for k, c in enumerate(cases)],
        )
    print()

    # 4. meccanismo
    print("[4] meccanismo: spazi distribuiti vs omoglifi, a parita' di caratteri toccati:")
    for dose in (0.25, 1.00):
        h = [homoglyph(c.state, GREEK, dose, k) for k, c in enumerate(cases)]
        # Stessi caratteri toccati, non gli stessi byte: `homoglyph` sostituisce solo
        # i caratteri presenti nella tabella, quindi il numero di sostituzioni va
        # contato su quelli. Contare tutta la lunghezza dello stato (cifre e
        # punteggiatura incluse) faceva toccare ~374 caratteri all'arma degli spazi
        # contro ~40 a quella greca: il controllo non era abbinato.
        n = int(
            sum(sum(1 for ch in c.state if ch in GREEK) * dose for c in cases) / len(cases)
        )
        report(f"omoglifi dose {dose:.0%}", h)
        report(
            f"spazi distribuiti ~{n} char",
            [spaces_distributed(c.state, n, k) for k, c in enumerate(cases)],
        )
    d_tok, _ = report(
        "filler in coda (pari Δtoken)", [filler_tail(c.state, 212, eng, k) for k, c in enumerate(cases)]
    )
    print(f"    (filler ha aggiunto {d_tok:.0f} token in coda, dove non tocca il contenuto)")
    print()

    # 5. collasso
    print("[5] collasso del vocabolario delle label:")
    pred_clean, pred_adv = Counter(), Counter()
    for c in cases:
        for qid in c.gold:
            pred_clean[cb[c.id]["labels"][qid]] += 1
    states = [homoglyph(c.state, GREEK, 1.0, k) for k, c in enumerate(cases)]
    adv = evaluate(
        eng,
        [Case(id=c.id, state=s, questions=c.questions, gold=c.gold) for c, s in zip(cases, states)],
        batch_size=16,
        chunk=32,
    )
    for r in adv.records:
        for qid in r["labels"]:
            pred_adv[r["labels"][qid]] += 1
    lost = [k for k in pred_clean if pred_adv[k] == 0]
    print(f"    label distinte: pulito {len(pred_clean)}, attacco {len(pred_adv)}")
    print(f"    sparite del tutto: {sorted(lost)}")
    print(f"    predette pulito: {dict(pred_clean.most_common(5))}")
    print(f"    predette attacco: {dict(pred_adv.most_common(5))}")

    eng.close()

    # L'archivio e' il punto di questa modifica: fino al 2026-10-05 tutta la curva di dose
    # restava su console, quindi nessun numero era rintracciabile e la nota dichiarava i
    # buchi. Le righe portano `n_decisions` e `asr_denominator` espliciti perche' un ASR
    # senza denominatore non e' confrontabile, e `dose` perche' le righe non sono a dose
    # uguale: confrontare dose 0.25 con dose 1.0 e' aritmetica, non esperimento.
    payload = {
        "experiment": "E12d_homoglyph",
        "question": "gli omoglifi ingannano il modello o lo degradano a caso?",
        "axes": {
            "checkpoint": DEFAULT_CHECKPOINT,
            "n_cases": len(cases),
            "seed": 0,
            "doses": [0.10, 0.25, 0.50, 1.00],
            "study_valid": True,
        },
        "study_valid": {
            "why": "l'operatore ha firma (state) -> str: non vede questions ne gold",
            "verified_per_row": "questions_untouched",
        },
        "results": rows,
    }
    io.write_result("E12d_homoglyph_dose", payload)
    print(f"\n[E12d] wrote results/E12d_homoglyph_dose.json ({len(rows)} righe)")


if __name__ == "__main__":
    main()
