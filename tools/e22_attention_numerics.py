"""Il profilo d'attenzione appartiene alla decisione pubblicata?

L'estrazione gira in **eager** perche' SDPA non restituisce i pesi, e in bf16 l'eager
non e' bit-identico al kernel fuso. Il numero che conta e' quanti casi decidono
diversamente, e la causa va isolata: se un forward fatto con il **nostro** percorso ma
in SDPA decide come il run stock, la differenza e' dell'implementazione dell'attenzione e
non della tubatura; se non decide come il run stock, la tubatura ha un problema e il
profilo non e' confrontabile con nessuna predizione pubblicata.

Tre conti sullo stesso campione di casi:
1. il run stock (`predict_batch`, SDPA);
2. lo stesso batch attraverso il percorso dell'estrattore, in SDPA;
3. lo stesso batch attraverso il percorso dell'estrattore, in eager (quello che gira).

Uso:
    uv run python tools/e22_attention_numerics.py [--limit 200] [--json PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from layaadv import io, paths
from layaadv.data import loaders
from layaadv.model.engine import LayaEngine

DEFAULT_CHECKPOINT = "convaiinnovations/laya-typed-decisions"
QID = loaders.QID


def direct_argmax(engine: LayaEngine, case, implementation: str) -> str | None:
    """Il forward di un caso per il percorso dell'estrattore, con una data implementazione."""
    import torch
    from laya.agent import _amp_context
    from laya.common import collate_items

    encoder = engine._agent.model.encoder
    previous = encoder.config._attn_implementation
    encoder.config._attn_implementation = implementation
    try:
        questions = dict(case.questions)
        ids = list(questions)
        internal = engine._internal(questions)
        items = engine._agent._encode_state(case.state, ids, internal)
        batch = collate_items([items], engine._agent.tok.pad_token_id)
        device = engine._agent.device
        with torch.no_grad(), _amp_context(engine._agent.device, engine._agent.dtype):
            logits, _act = engine._agent.model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device),
                batch["marker_pos"].to(device), batch["marker_mask"].to(device),
                batch["qtype"].to(device),
            )
    finally:
        encoder.config._attn_implementation = previous
    options = list(case.questions[ids[0]]["criteria"])
    index = int(logits[0].float().argmax())
    return options[index] if index < len(options) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    engine = LayaEngine(args.checkpoint)
    cases, ds, _ = loaders.load_cases("massive-it", k=16, limit=args.limit, seed=0)
    stock = engine.predict_batch(cases, batch_size=1)
    mine_sdpa = [direct_argmax(engine, c, "sdpa") for c in cases]
    mine_eager = [direct_argmax(engine, c, "eager") for c in cases]
    stock_labels = [p.labels[QID] for p in stock]
    engine.close()

    counts = {
        "n_cases": len(cases),
        "plumbing_vs_stock": sum(1 for a, b in zip(mine_sdpa, stock_labels) if a != b),
        "eager_vs_stock": sum(1 for a, b in zip(mine_eager, stock_labels) if a != b),
        "eager_vs_plumbing": sum(1 for a, b in zip(mine_eager, mine_sdpa) if a != b),
    }
    payload = {
        "experiment": "E22_attention_numerics",
        "question": "il profilo eager appartiene alla decisione pubblicata?",
        "checkpoint": args.checkpoint,
        "dataset": "massive-it",
        "k": 16,
        "seed": 0,
        "counts": counts,
        "env": io.env_fingerprint(engine.weight_sha256()),
    }
    out = Path(args.json) if args.json else (
        paths.RESULTS / "provenance" / "e22_attention_numerics.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")

    print(f"[E22-numerica] {counts['n_cases']} casi: "
          f"tubatura-SDPA vs stock {counts['plumbing_vs_stock']}, "
          f"tubatura-eager vs stock {counts['eager_vs_stock']}, "
          f"eager vs SDPA {counts['eager_vs_plumbing']}")
    if counts["plumbing_vs_stock"]:
        print("[E22-numerica] ATTENZIONE: la tubatura dell'estrattore non riproduce il run "
              "stock, il profilo non e' confrontabile con le predizioni pubblicate")
        return 1
    print("[E22-numerica] la tubatura riproduce il run stock: la differenza residua e' "
          "solo dell'implementazione dell'attenzione")
    print(f"[E22-numerica] scritto {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
