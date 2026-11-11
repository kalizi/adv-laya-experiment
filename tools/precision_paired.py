"""Confronto appaiato fra due precisioni dello stesso backend, con McNemar.

Perche' un confronto aggregato non basta. Su CLM i numeri di E03 a `Q6_K` e a
`Q5_K_M` coincidono a quattro decimali, il che sembra "la quantizzazione non muove
niente". Ma due ASR uguali possono nascondere due insiemi di decisioni ribaltate
diverse che si compensano. L'unica domanda che risponde alla causa e' **sulle
stesse righe**: quali decisioni cambiano stato di ribaltamento quando cambia la
precisione.

Il test e' McNemar sulle righe appaiate, lo stesso metodo con cui E15 ha
risolto la stessa domanda sulla famiglia laya: la differenza fra due
precisioni su dati accoppiati si stima con un intervallo di proporzioni
indipendenti sbaglia, perche' le due condizioni condividono lo stesso caso.

Risponde a tre cose, in ordine:
1. quanto la quantizzazione muove il modello **pulito** (quante label cambiano);
2. se muove il **tasso di ribaltamento** dell'attacco (McNemar sui ribaltamenti);
3. se muove l'**argomento dell'esperimento**: gli ASR vanno ricalcolati
   sull'intersezione dei corretti-puliti delle due precisioni, altrimenti i
   denominatori differiscono e il confronto non e' appaiato.

Uso:
    uv run python tools/precision_paired.py --a Q6_K --b Q5_K_M
"""

from __future__ import annotations

import argparse
import json
import math

from layaadv import paths

RUNS = paths.RESULTS / "E03_option_injection" / "runs"


def read_run(run_id: str) -> dict[str, dict]:
    """Records by id, so two runs are compared on the same cases."""
    path = RUNS / run_id / "preds.jsonl"
    return {
        str(r["id"]): r for r in (json.loads(line) for line in path.open() if line.strip())
    }


def per_decision(records: dict[str, dict]) -> dict[tuple[str, str], bool]:
    """{(case, qid): correct} su tutto cio' che il record contiene."""
    out: dict[tuple[str, str], bool] = {}
    for cid, rec in records.items():
        for qid, pred in rec["labels"].items():
            out[(cid, qid)] = pred == rec["gold"][qid]
    return out


def mcnemar(b: int, c: int) -> dict:
    """Two-sided exact binomial p on the discordant pairs.

    `b` e `c` sono le due celle discordanti. Il test esatto di McNemar e' il
    binomiale a due code sulla moneta equa: `p = 2 * P(X >= max(b,c))`, con X
    binomiale(n = b + c, 1/2). Non uso la normale approssimata perche' con b e c
    piccoli sbaglia, ed e' esattamente il caso in cui il risultato e' utile.
    """
    n = b + c
    if n == 0:
        return {"discordant_b": b, "discordant_c": c, "p": 1.0, "n_discordant": 0}
    k = max(b, c)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2**n)
    return {
        "discordant_b": b,
        "discordant_c": c,
        "n_discordant": n,
        "p": min(1.0, 2 * tail),
    }


def scopes_for(prefix: str) -> dict[tuple, dict]:
    """Righe E03 di uno slug di checkpoint, indicizzate per (dataset, k, posizione)."""
    index = json.loads((paths.RESULTS / "E03_option_injection.json").read_text())
    out: dict[tuple, dict] = {}
    for filename in index["files"]:
        if prefix not in filename:
            continue
        payload = json.loads((paths.RESULTS / filename).read_text())
        for row in payload["results"]:
            key = (row["dataset"], row["k"], row["position"])
            out[key] = row
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", default="clm-v0.1-8B", help="slug del backend A (riferimento)")
    parser.add_argument("--b", default="clm-v0.1-8B-Q5_K_M", help="slug del backend B")
    parser.add_argument("--only", default=None, help="filtra il dataset")
    args = parser.parse_args()

    A = scopes_for(args.a)
    B = scopes_for(args.b)
    common = sorted(set(A) & set(B))
    if args.only:
        common = [k for k in common if k[0] == args.only]
    if not common:
        print(f"nessuna riga in comune fra {args.a!r} e {args.b!r}")
        return 1

    print(f"confronto appaiato: {args.a} (A) contro {args.b} (B)\n")
    hdr = (
        f"{'dataset':11s} {'K':>3s} {'pos':>5s} {'ASR A':>7s} {'ASR B':>7s} "
        f"{'lab diverse':>12s} {'clean':>7s} {'b':>4s} {'c':>4s} {'p':>9s}"
    )
    print(hdr)
    summary = []
    for key in common:
        ra, rb = A[key], B[key]
        ca, cb = read_run(ra["clean_run_id"]), read_run(rb["clean_run_id"])
        aa, ab = read_run(ra["adv_run_id"]), read_run(rb["adv_run_id"])

        da = per_decision(ca)
        db = per_decision(cb)
        keys = sorted(set(da) & set(db))

        # 1. quanto la quantizzazione muove il modello pulito
        moved_clean = sum(1 for k in keys if da[k] != db[k])
        # 2. McNemar sullo stato di ribaltamento, ristretto ai corretti in entrambi
        ok = [k for k in keys if da[k] and db[k]]
        flips_a = {k for k in ok if not per_decision(aa).get(k, True)}
        flips_b = {k for k in ok if not per_decision(ab).get(k, True)}
        b_only = len(flips_a - flips_b)
        c_only = len(flips_b - flips_a)
        test = mcnemar(b_only, c_only)

        # 3. ASR ricalcolato sulla intersezione, cosi' i denominatori sono uguali
        asr_a = len(flips_a) / len(ok) if ok else None
        asr_b = len(flips_b) / len(ok) if ok else None

        print(
            f"{key[0]:11s} {key[1]:3d} {key[2]:>5s} {asr_a:7.4f} {asr_b:7.4f} "
            f"{moved_clean:12d} {len(ok):7d} {b_only:4d} {c_only:4d} {test['p']:9.4f}"
        )
        summary.append(
            {
                "key": key,
                "n_intersezione": len(ok),
                "asr_a": round(asr_a, 4),
                "asr_b": round(asr_b, 4),
                "label_diverse_pulito": moved_clean,
                "mcnemar": test,
            }
        )

    print("\n" + json.dumps(summary, indent=1))
    moved = sum(s["label_diverse_pulito"] for s in summary)
    total = sum(s["n_intersezione"] for s in summary)
    flips = sum(s["mcnemar"]["discordant_b"] + s["mcnemar"]["discordant_c"] for s in summary)
    print(
        f"\nIn totale la quantizzazione cambia {moved} label su {total} decisioni puliti "
        f"appaiate ({moved / total:.4f} del campione), e lo stato di ribaltamento "
        f"sull'attacco differisce su {flips} decisioni."
    )
    print(
        "Se `label diverse` e' grande ma i flip discordanti sono pochi, allora la "
        "quantizzazione muove molte singole decisioni senza spostare l'attacco: che e'"
        "il caso di E15 sulla famiglia laya, e significa che la quantizzazione non e' una "
        "difesa ne' qui."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
