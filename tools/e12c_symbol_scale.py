"""Il simbolo scala? Iniezione cumulativa di simboli e di spazi, con confidenza.

E12b ha mostrato che sostituire parole non scala. Qui la domanda è se una
perturbazione che non cambia significato scala almeno in confidenza: 1, 3, 10,
30 iniezioni. Se la confidenza sul torto resta bassa, è rumore. Se sale, è
superficie, e sarebbe la piu' economica in assoluto (non serve alcun vocabolario).
"""

import random

from layaadv.data import typed_decisions as td
from layaadv.evaluate import evaluate
from layaadv.model.engine import DEFAULT_CHECKPOINT, LayaEngine
from layaadv.schema import Case

SYMBOLS = ["#", "*", "~", "|", "@", "+", "=", "^", "%", "$"]
LEVELS = [1, 3, 10, 30]

cases = td.load_cases("test", limit=30)
eng = LayaEngine(DEFAULT_CHECKPOINT, batch_size=16)
clean = evaluate(eng, cases, batch_size=16, chunk=32)
cb = {r["id"]: r for r in clean.records}


def inject(state, level, alphabet, seed=0):
    """Inserisce `level` caratteri in posizioni casuali dentro i valori JSON.

    Uso un set di ancoraggio: le chiavi di primo livello sono identiche in tutti
    gli stati, quindi non inietto mai dentro una chiave, solo dentro i valori.
    """
    rng = random.Random(seed)
    out = state
    for _ in range(level):
        ch = rng.choice(alphabet)
        # posizione casuale dopo la prima graffa, per stare dentro il corpo
        lo = out.find("{") + 1
        hi = len(out) - 1
        if hi <= lo:
            return out
        at = rng.randrange(lo, hi)
        out = out[:at] + ch + out[at:]
    return out


def report(name, states):
    mutants = [
        Case(id=c.id, state=s, questions=c.questions, gold=c.gold) for c, s in zip(cases, states)
    ]
    adv = evaluate(eng, mutants, batch_size=16, chunk=32)
    ab = {r["id"]: r for r in adv.records}
    diff = flips = corr = 0
    conf_on_wrong = []
    oer = 0
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
                conf_on_wrong.append(cf)
                if cf >= 0.75:
                    oer += 1
    toks = sum(eng.count_tokens(s) - eng.count_tokens(c.state) for c, s in zip(cases, states)) / len(states)
    mean_conf = sum(conf_on_wrong) / len(conf_on_wrong) if conf_on_wrong else 0.0
    print(
        f"{name:<26} Δtok {toks:>5.1f}  label {diff:>3}/{len(cases) * 3:<4} "
        f"flip {flips:>2}  ASR {flips / corr if corr else 0:>6.4f}  "
        f"conf {mean_conf:>5.3f}  OER {oer}"
    )
    return flips, mean_conf, oer


print(f"baseline acc {clean.metrics['accuracy']:.4f}\n")
print(f"{'braccia':<26} {'budget':>10}  {'movenute':>12}  {'flip':>6}  {'ASR':>8}  {'conf':>6}  OER")
print("-" * 92)
for level in LEVELS:
    report(f"simbolo x{level}", [inject(c.state, level, SYMBOLS, k) for k, c in enumerate(cases)])
print()
for level in LEVELS:
    report(f"spazio x{level}", [inject(c.state, level, [" "], k) for k, c in enumerate(cases)])

eng.close()
