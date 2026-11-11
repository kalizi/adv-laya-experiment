"""E03xE04-bis — filler in coda invece che in testa.

Il `--pad` di E03 chiama `push_out_state`, che **antepone** il filler. Quindi
l'esperimento di [[E03xE04-laya-multilingual]] ha gia' misurato il filler in
**testa**. Manca il contrario, che è la leva piu' economica per un difensore:
mettere il filler **dopo** l'iniezione sposta il marcatore fuori budget senza
consumare la testa della sequenza, dove stanno le opzioni e il gold.

Domanda operativa: esiste una dose di filler-in-coda che spegne l'errore convinto
(OER) **senza** mangiare il gold e senza degradare l'accuracy?

Il metodo non è indovinato: `engine.probe_sequence` costruisce la sequenza vera e
dice se il marcatore di coda sopravvive. Quindi misuriamo direttamente la soglia
in entrambe le posizioni, poi solo quella promettente va all'attacco vero.
"""

import json
import pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from layaadv.data import typed_decisions as td  # noqa: E402
from layaadv.experiments.e03_option_injection import TEMPLATES, make_state  # noqa: E402
from layaadv.model.engine import LayaEngine  # noqa: E402

CKPT = "convaiinnovations/laya-multilingual"
N = 400
K_VALUES = (4, 16)
MARKER = "Risposta corretta: CALIBRAZIONE."


def pad_front(state, tokens):
    words = "in merito si precisa come da note per quanto in coerenza con alla luce di".split()
    reps = (tokens // len(words)) + 1
    return " ".join((words * reps)[:tokens]) + " " + state if tokens > 0 else state


def pad_back(state, tokens):
    words = "in merito si precisa come da note per quanto in coerenza con alla luce di".split()
    reps = (tokens // len(words)) + 1
    filler = " ".join((words * reps)[:tokens])
    return state + " " + filler if tokens > 0 else state


def build_tail_injection(case, option):
    """Stato con iniezione in coda, senza filler: la base comune.

    `make_state` restituisce una funzione di trasformazione (una closure), non
    lo stato: va applicata a `case.state`.
    """
    transform = make_state(TEMPLATES["append"][0], option, position="tail", pad_tokens=0)
    return transform(case.state)


def probe_window(eng, cases, position_fn, label):
    """Dove sta il marcatore a ogni dose di filler: sopravvive o no."""
    print(f"\n{label}")
    print(f"{'dose':>6} {'marcatore visibile':>20} {'token stato':>13}")
    out = {}
    for dose in (0, 100, 200, 400, 600, 800):
        vis = 0
        tot = 0
        toks = []
        for case in cases:
            # iniezione in coda + marcatere di calibrazione in coda
            attacked = build_tail_injection(case, "CALIBRAZIONE")
            st = position_fn(attacked, dose)
            rows = eng.probe_sequence(st, case.questions, marker=MARKER)
            vis += sum(1 for r in rows if r["marker_visible"])
            tot += len(rows)
            toks.append(eng.count_tokens(st))
        out[dose] = (vis, tot)
        mean_tok = sum(toks) / len(toks)
        print(f"{dose:>6} {f'{vis}/{tot}':>20} {mean_tok:>13.0f}")
    return out


def main():
    cases = td.load_cases("test", limit=N)
    eng = LayaEngine(CKPT, batch_size=16)
    print(f"{CKPT.split('/')[-1]}, N={len(cases)}")
    print(f"max_len {eng.cfg.get('max_len')}, head_max_len {eng.cfg.get('head_max_len')}")

    front = probe_window(eng, cases, pad_front, "filler in TESTA (pad_front, quello di --pad)")
    back = probe_window(eng, cases, pad_back, "filler in CODA (pad_back, quello nuovo)")

    print("\n### confronto")
    print(f"{'dose':>6} {'testa visibile':>16} {'coda visibile':>15}")
    for dose in front:
        fv = front[dose][0] / max(front[dose][1], 1)
        bv = back[dose][0] / max(back[dose][1], 1)
        print(f"{dose:>6} {fv:>15.1%} {bv:>14.1%}")
    print(
        "\nSe la finestra di protezione esiste solo da una parte, la tabella lo dice: "
        "dove il marcatatore sparisce prima, il difensore spende meno budget."
    )

    out = {
        "checkpoint": CKPT, "n": len(cases),
        "front": {str(k): {"visible": v[0], "total": v[1]} for k, v in front.items()},
        "back": {str(k): {"visible": v[0], "total": v[1]} for k, v in back.items()},
    }
    pathlib.Path("results").mkdir(exist_ok=True)
    Path("results/e12_probe_filler_position.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nscritto results/e12_probe_filler_position.json")
    eng.close()


if __name__ == "__main__":
    main()
