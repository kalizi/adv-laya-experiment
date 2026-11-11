"""Does quantising the KV cache move the embeddings, and by how much.

The question matters because of what we are trying to settle. E21's result is
measured `@Q6_K`, and the standing objection is "that is a quantisation artefact".
We bounded it with `Q5_K_M`, whose own distance from bf16 is known. The KV cache is
a *third* quantiser, on a different axis: it does not touch the weights at all, only
the stored keys and values that attention reads. So it cannot be a shortcut to
"run bf16" -- bf16 weights are 16.4 GB and the card has 10.7, and the largest KV
saving available is 1.69 GiB. But it *is* usable as evidence, or it is not, and the
only way to know is to measure how far it moves the pooled vector.

Read: if the drift is larger than the drift we attribute to weight quantisation, it
is a coarser instrument than the one we already have, and using it to bound the
bf16 question would be moving away from bf16 rather than towards it.

Usage:
    llama-server -m <gguf> --embedding --pooling last -ngl 36 -ctk f16 -ctv f16  --port 8091
    llama-server -m <gguf> --embedding --pooling last -ngl 36 -ctk q8_0 -ctv q8_0 --port 8092
    uv run python tools/kv_quant_drift.py --ports 8091:fp16 8092:q8_0
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.request


def embed(port: int, texts: list[str]) -> list[list[float]]:
    body = json.dumps({"input": texts}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/embeddings",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        out = json.load(resp)
    return [d["embedding"] for d in sorted(out["data"], key=lambda d: d["index"])]


def cos(a: list[float], b: list[float]) -> float:
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (da * db)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ports",
        nargs="+",
        required=True,
        help="coppie porta:etichetta, la prima è il riferimento",
    )
    parser.add_argument("--n", type=int, default=32)
    args = parser.parse_args()

    specs = []
    for item in args.ports:
        port, label = item.split(":", 1)
        specs.append((int(port), label))
    base_port, base_label = specs[0]

    # Stati realistici: ~250 token, la lunghezza di un caso E03 su MASSIVE-it.
    states = [
        f"Agente: la richiesta #{4000 + i} di ACME Corp da 12400 EUR e in ritardo da "
        f"{30 + i} giorni. Il cliente ha gia segnalato il problema due volte. "
        f"Sintomi: timeout ripetuti, {i} errori nel log."
        for i in range(args.n)
    ]

    base = embed(base_port, states)
    print(f"riferimento: {base_label} su porta {base_port}, {len(states)} stati\n")
    hdr = f"{'confronto':>22s} {'cos min':>10s} {'cos medio':>11s} {'cos sotto 0.999':>16s}"
    print(hdr)
    rows = []
    for port, label in specs[1:]:
        other = embed(port, states)
        cs = [cos(a, b) for a, b in zip(base, other)]
        below = sum(1 for c in cs if c < 0.999)
        print(
            f"{f'{base_label} vs {label}':>22s} {min(cs):10.5f} "
            f"{sum(cs) / len(cs):11.6f} {below:16d}"
        )
        rows.append(
            {
                "kv": label,
                "cos_min": round(min(cs), 6),
                "cos_mean": round(sum(cs) / len(cs), 6),
                "n_below_0.999": below,
            }
        )
    print("\n" + json.dumps(rows, indent=1))
    print(
        "\nPer confronto: la carta GGUF misura cos min 0.98539 e cos medio 0.99937 fra "
        "Q6_K e un riferimento bf16 dello stesso runtime, cioe' la deriva che attribuiamo "
        "alla quantizzazione dei pesi. Se questa tabella e' piu' stretta di quella, la KV "
        "e' uno strumento piu' fine; se e' piu' larga, e' uno strumento piu' grossolano e "
        "non serve a chiudere la domanda sul bf16."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
