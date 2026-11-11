"""E01-terzo — baseline pulita di laya-multilingual, e verifica della calibrazione.

Il vault chiede `laya-multilingual` come terzo checkpoint di E03, con un
prerequisito esplicito: **rifittare le temperature**, altrimenti i numeri di
confidence non sono confrontabili con gli altri due checkpoint.

Qui si controlla se il prerequisito serve davvero. La misura non è l'ECE ma
una cosa che risponde alla domanda operativa: **la confidence del checkpoint
separa il giusto dallo sbagliato?** Se s'è confrontabile, l'uguaglianza con
laya-typed-decisions basta. Se no, ogni OER di questo progetto è locale al
proprio checkpoint e i confronti trasferibili sono invalidi.
"""

import json
from collections import defaultdict

from layaadv.data import typed_decisions as td
from layaadv.evaluate import evaluate
from layaadv.model.engine import LayaEngine

CHECKPOINTS = [
    "convaiinnovations/laya-typed-decisions",
    "convaiinnovations/laya",
    "convaiinnovations/laya-multilingual",
]


def ece_and_gap(records, cases, n_bins=10):
    """ECE a 10 bin e separazione giusto/sbagliato."""
    bins = defaultdict(list)
    for c in cases:
        for qid, g in c.gold.items():
            r = next(x for x in records if x["id"] == c.id)
            conf = r["confidence"][qid]
            correct = int(r["labels"][qid] == g)
            bins[min(int(conf * n_bins), n_bins - 1)].append((conf, correct))
    ece = 0.0
    n = sum(len(v) for v in bins.values())
    for v in bins.values():
        acc = sum(c for _, c in v) / len(v)
        conf = sum(cf for cf, _ in v) / len(v)
        ece += len(v) / n * abs(acc - conf)
    right = [cf for v in bins.values() for cf, c in v if c == 1]
    wrong = [cf for v in bins.values() for cf, c in v if c == 0]
    return ece, (sum(right) / len(right) if right else 0.0), (
        sum(wrong) / len(wrong) if wrong else 0.0
    )


def main():
    cases = td.load_cases("test", limit=400)
    out = {}
    print(f"typed-decisions test, N={len(cases)} casi\n")
    print(
        f"{'checkpoint':<44} {'acc':>7} {'ECE':>7} {'conf giusto':>12} "
        f"{'conf sbagliato':>15}"
    )
    print("-" * 92)
    for ckpt in CHECKPOINTS:
        eng = LayaEngine(ckpt, batch_size=16)
        r = evaluate(eng, cases, batch_size=16, chunk=32)
        ece, c_right, c_wrong = ece_and_gap(r.records, cases)
        temps = getattr(eng._agent.model, "temperature", None)
        t_list = [round(float(x), 4) for x in temps.flatten()] if temps is not None else None
        out[ckpt] = {
            "accuracy": r.metrics["accuracy"],
            "ece": ece,
            "conf_right": c_right,
            "conf_wrong": c_wrong,
            "temperatures": t_list,
            "load_warnings": eng.load_warnings,
            "max_len": eng.cfg.get("max_len"),
            "head_max_len": eng.cfg.get("head_max_len"),
            "weight_sha256": eng.weight_sha256(),
        }
        print(
            f"{ckpt.split('/')[-1]:<44} {r.metrics['accuracy']:>7.4f} {ece:>7.3f} "
            f"{c_right:>12.3f} {c_wrong:>15.3f}"
        )
        print(f"{'':44} temperature: {t_list}   warning: {len(eng.load_warnings)}")
        eng.close()
    print("\n### cosa significa per il progetto")
    ml = out["convaiinnovations/laya-multilingual"]
    print(
        f"  multilingual: ECE {ml['ece']:.3f}, conf giusto {ml['conf_right']:.3f} "
        f"contro sbagliato {ml['conf_wrong']:.3f}"
    )
    if ml["conf_right"] - ml["conf_wrong"] > 0.15:
        print("  la confidence separa: il confronto dell'OER fra checkpoint è leggibile")
    else:
        print("  la confidence NON separa: l'OER è confrontabile solo dentro il checkpoint")
    import pathlib

    pathlib.Path("results/e01_third_checkpoint.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n  scritto results/e01_third_checkpoint.json")
    return out


if __name__ == "__main__":
    main()
