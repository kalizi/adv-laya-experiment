"""Does CLM's request concurrency move the predictions?

CLM's only batching lever is how many cases are in flight against
`llama-server`'s continuous-batching slots, so `--batch-size` is not inert here
the way it is on laya over `massive-it`. But a changed batch composition moved
bf16 predictions on laya (chunk=1 -> 0.7500 where chunk=32 -> 0.7469), and
E17's whole result is that the configuration knobs do not move E03's ASR. Both
of those were measured; this one has to be too.

Reads a run's records from the store and re-scores the same cases at several
concurrencies, reporting label flips and the confidence drift. Run it before
trusting a CLM number, and put the outcome in the notes either way.

Usage:
    uv run python tools/clm_concurrency_check.py --k 16 --n 128
    uv run python tools/clm_concurrency_check.py --k 16 --n 128 --taus 1 8
"""

from __future__ import annotations

import argparse
import json

from layaadv.data import loaders
from layaadv.metrics.core import top1_margin
from layaadv.model.engine import open_backend


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="massive-it")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--n", type=int, default=128, help="casi (piccolo: serve un confronto, non un ASR)")
    parser.add_argument("--taus", default="1,8", help="concurrency da confrontare")
    parser.add_argument("--quantize-note", default="")
    args = parser.parse_args()

    taus = [int(x) for x in args.taus.split(",")]
    engine = open_backend("clm-llama", None)
    print(f"[clm] dataset={args.dataset} K={args.k} n={args.n} taus={taus}", flush=True)
    ident = engine.run_identity()
    print(f"[clm] {ident['encoder_runtime']} {ident['quantization']}", flush=True)

    cases, _ds, _labels = loaders.load_cases(args.dataset, k=args.k, limit=args.n, seed=0)
    qid = next(iter(cases[0].questions))

    results: dict[int, dict[str, dict]] = {}
    for tau in taus:
        preds = engine.predict_batch(cases, batch_size=tau)
        results[tau] = {
            str(c.id): dict(p.answers[qid]["probabilities"]) for c, p in zip(cases, preds)
        }
        print(f"[clm]   tau={tau} fatto", flush=True)

    base = taus[0]
    print(f"\n{'confronto':>14s} {'label diverse':>14s} {'max |Δconf|':>13s} {'max |Δmargin|':>14s}")
    rows = []
    for tau in taus[1:]:
        flips = 0
        max_dc = 0.0
        max_dm = 0.0
        for cid in results[base]:
            pb = results[base][cid]
            pt = results[tau][cid]
            if max(pb, key=lambda k: (pb[k], k)) != max(pt, key=lambda k: (pt[k], k)):
                flips += 1
            max_dc = max(max_dc, abs(sum(pb.values()) - sum(pt.values())))
            max_dm = max(max_dm, abs(top1_margin(pb) - top1_margin(pt)))
        rows.append(
            {
                "tau": tau,
                "label_flips": flips,
                "max_dsum": round(max_dc, 6),
                "max_dmargin": round(max_dm, 6),
            }
        )
        print(f"{f'{base} vs {tau}':>14s} {flips:>7d}/{len(cases):<6d} {max_dc:13.6f} {max_dm:14.6f}")

    print("\n" + json.dumps(rows, indent=2))
    print(
        "\nSe label_flips > 0 la concorrenza e' una condizione sperimentale e va nel "
        "run_id, perche' due run con concorrenza diversa non si confrontano."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
