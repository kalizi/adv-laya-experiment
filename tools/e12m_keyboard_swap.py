"""E12m — riordino di caratteri nel cluster dei tasti adiacenti.

È l'attacco piu' pulito possibile: **costa zero token**. Gli stessi caratteri,
solo in ordine diverso, quindi il confronto con ogni altro esperimento non ha
confound di budget. Se il modello cambia risposta, non è perche' ha mangiato
context: è perche' l'ordine dei caratteri conta.

Il "cluster dei tasti vicini" è l'ipotesi che rende l'attacco plausibile: due
tasti adiacenti sulla tastiera producono un errore di battitura che un umano
non corregge e quasi non nota. Scambiando due tasti *lontani* l'attacco diventa
visibile, e il confronto fra i due dice se la plausibilita' da refuso conta o se
è solo rumore.

Restrizioni: solo lettere, mai dentro numeri (cambierebbe il valore) né dentro
identificatori (cambierebbe un nome di campo, che non è un refuso ma un'altra
voce). Solo lettere vicine in posizione.
"""

import random
from collections import Counter

from layaadv.data import typed_decisions as td
from layaadv.evaluate import evaluate
from layaadv.model.engine import DEFAULT_CHECKPOINT, LayaEngine
from layaadv.schema import Case

N_CASES = 30

# Tastiera QWERTY fisica: tre righe, con la riga centrale spostata di mezzo tasto.
ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]


def build_neighbours(max_col_offset=1):
    """Dizionario lettera -> lettere raggiungibili con uno spostamento plausibile.

    `max_col_offset` controlla quanto è "vicino": 1 = tasto adiacente o
    diagonale, 3 = tasto lontano (controllo). Le righe sono allineate per
    colonna con l'offset fisico della riga centrale.
    """
    pos = {}
    for r, row in enumerate(ROWS):
        # la riga centrale è spostata di mezzo tasto a destra
        shift = 0.5 if r == 1 else 0.0
        for c, ch in enumerate(row):
            pos[ch] = (r, c + shift)
    nb = {}
    for a, (ra, ca) in pos.items():
        near = []
        for b, (rb, cb) in pos.items():
            if a == b or abs(rb - ra) > 1:
                continue
            if abs(cb - ca) <= max_col_offset:
                near.append(b)
        nb[a] = sorted(near)
    return nb


def swap_in_prose(state, neighbours, dose, seed=0, protected=("_",)):
    """Scambia coppie di lettere adiacenti in posizione, dentro il testo.

    Una posizione è ammissibile se i due caratteri sono lettere consecutive,
    non identiche, entrambe mappate nella tabella, e la parola non è un
    identificatore (contiene `_` o cifre).
    """
    rng = random.Random(seed)
    words = state.split()
    for wi, w in enumerate(words):
        if any(p in w for p in protected) or any(ch.isdigit() for ch in w):
            continue
        chars = list(w)
        spots = []
        for i in range(len(chars) - 1):
            a, b = chars[i].lower(), chars[i + 1].lower()
            if a == b or a not in neighbours or b not in neighbours:
                continue
            if b in neighbours[a]:          # scambio legittimo: b vicino ad a
                spots.append(i)
        if not spots:
            continue
        rng.shuffle(spots)
        n = max(1, int(len(spots) * dose))
        for i in spots[:n]:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        words[wi] = "".join(chars)
    return " ".join(words)


def main():
    eng = LayaEngine(DEFAULT_CHECKPOINT, batch_size=16)
    cases = td.load_cases("test", limit=N_CASES)
    clean = evaluate(eng, cases, batch_size=16, chunk=32)
    cb = {r["id"]: r for r in clean.records}
    total = sum(len(c.gold) for c in cases)
    base_acc = clean.metrics["accuracy"]
    print(f"baseline acc {base_acc:.4f} su {total} decisioni\n")

    near = build_neighbours(1)
    far = build_neighbours(3)
    print("cluster tasti adiacenti (offset 1):")
    for k in "qwertyasdf":
        print(f"  {k} -> {''.join(near[k])}")
    print(f"\n  lettere con almeno un vicino: {sum(1 for v in near.values() if v)}/26")
    print(f"  media vicini per tasto: {sum(len(v) for v in near.values()) / 26:.1f}")

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
        changed = sum(1 for c, s in zip(cases, states) if s != c.state)
        acc = ok / total
        mc = sum(conf) / len(conf) if conf else 0.0
        asr = flips / corr if corr else 0.0
        print(f"  {name:<36} Δtok {d_tok:>5.1f}  stati cambiati {changed:>2}/30  "
              f"drop {(base_acc - acc) * 100:>5.1f}pp  ASR {asr:>6.4f}  "
              f"conf {mc:>5.3f}  OER {oer}")

    print("\ncurva di dose, tasti adiacenti (offset 1):")
    for dose in (0.10, 0.25, 0.50, 1.00):
        report(f"adiacenti dose {dose:.0%}",
               [swap_in_prose(c.state, near, dose, k) for k, c in enumerate(cases)])
    print("\ncontrollo: tasti lontani (offset 3):")
    for dose in (0.25, 0.50, 1.00):
        report(f"lontani dose {dose:.0%}",
               [swap_in_prose(c.state, far, dose, k) for k, c in enumerate(cases)])

    print("\nesempio di stato riordinato (prime 200 caratteri):")
    ex = swap_in_prose(cases[0].state, near, 1.0, 0)
    print(f"  ORIGINALE: {cases[0].state[:200]}")
    print(f"  RIORDINATO: {ex[:200]}")
    print(f"\n  i numeri sono intatti: {'50' in ex} / {'32.5' in ex}")
    print(f"  stesse lunghezza: {len(ex) == len(cases[0].state)}")

    print("\ncollasso del vocabolario, adiacenti 100%:")
    states = [swap_in_prose(c.state, near, 1.0, k) for k, c in enumerate(cases)]
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
    eng.close()


if __name__ == "__main__":
    main()
