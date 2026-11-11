"""The encoder's attention layout, read from the loaded model.

The vault notes claim "1/3 global, 2/3 local window 128" for Laya
(10-modello/architettura-laya.md, 20-attack-surface/truncation-stato.md). A claim
about the model has to be read from the model: this loads the checkpoint, reads
both the config and the attention modules, checks them against each other, and
writes the layout as a provenance file. A note and a checkpoint drift apart, and
the notes of this project are the only place the claim exists.

Uso:
    uv run python tools/e22_attention_structure.py [--checkpoint REPO] [--json PATH]

Refuses to write when the loaded model disagrees with the note, so the check
cannot pass by rewriting the claim to match whatever was loaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from layaadv import io, paths
from layaadv.model.engine import LayaEngine

DEFAULT_CHECKPOINT = "convaiinnovations/laya-typed-decisions"

# Cosa dicono le note **oggi**, dopo la verifica del 2026-10-08. La verifica e' su questi
# numeri, non su "qualunque" layout: una riga per tipo di livello, con la frazione e la
# finestra. `window` e' il numero che la nota scrive (`local_attention`), quindi la
# raggiungibilita' attesa e' la sua meta': era il punto che la nota sbagliava.
CLAIMED = {
    "fraction_global": 10 / 28,
    "fraction_local": 18 / 28,
    "window": 128,
    "alternating": True,
}
CLAIMED_REACH = CLAIMED["window"] // 2

# Cosa dicevano le note prima, tenuto perche' la correzione e' il risultato di questo
# tool: senza, il payload non distinguerebbe "la nota era sbagliata" da "la nota non e' mai
# stata verificata". `1/3` perche' `global_attn_every_n_layers = 3` su 28 livelli da' 10
# globali (0.357), non 9.33; e la "finestra 128" era letta come raggio invece che come
# mezza finestra, quindi ±64 e non ±128.
CLAIMED_BEFORE = {
    "fraction_global": 1 / 3,
    "reach": 128,
    "corretto_il": "2026-10-08",
}


def layout(engine: LayaEngine) -> dict:
    """What the loaded model says, layer by layer."""
    import torch  # noqa: F401  (il controllo sul tipo del modulo ha bisogno di torch)

    model = engine._agent.model
    encoder = model.encoder
    config = encoder.config

    modules = [(name, mod) for name, mod in encoder.named_modules()
               if type(mod).__name__ == "ModernBertAttention"]
    layer_types = list(config.layer_types)
    if len(modules) != len(layer_types):
        raise RuntimeError(
            f"{len(modules)} moduli di attenzione per {len(layer_types)} layer_types: il "
            "modello caricato non e' quello che questo tool sa leggere"
        )

    rows = []
    disagreements = []
    for index, ((name, mod), kind) in enumerate(zip(modules, layer_types)):
        declared = getattr(encoder.layers[index], "attention_type", None)
        window = getattr(mod, "sliding_window", None)
        if declared != kind:
            disagreements.append(f"layer {index}: modulo {declared!r} vs config {kind!r}")
        if (window is None) != (kind == "full_attention"):
            disagreements.append(
                f"layer {index}: {kind} con sliding_window={window!r} sul modulo"
            )
        rows.append({
            "layer": index,
            "module": name,
            "layer_type": kind,
            "sliding_window_module": window,
            "heads": config.num_attention_heads,
        })

    n_global = sum(1 for r in rows if r["layer_type"] == "full_attention")
    n_local = len(rows) - n_global
    windows = sorted({r["sliding_window_module"] for r in rows if r["sliding_window_module"]})
    # Il modulo porta `config.sliding_window + 1`: il +1 serve ai kernel con confine
    # inclusivo, quindi la finestra *effettiva* in token e' window - 1.
    window_effective = windows[0] - 1 if len(windows) == 1 else None

    return {
        "checkpoint": engine.checkpoint,
        "weight_sha256": engine.weight_sha256(),
        "encoder": config._name_or_path if hasattr(config, "_name_or_path") else None,
        "architectures": list(getattr(config, "architectures", []) or []),
        "attn_implementation": config._attn_implementation,
        "n_layers": len(rows),
        "layers": rows,
        "layer_types_config": layer_types,
        "n_global": n_global,
        "n_local": n_local,
        "fraction_global": n_global / len(rows),
        "fraction_local": n_local / len(rows),
        "windows_module": windows,
        "window_effective": window_effective,
        "local_attention_config": getattr(config, "local_attention", None),
        "sliding_window_config": getattr(config, "sliding_window", None),
        "global_attn_every_n_layers": getattr(config, "global_attn_every_n_layers", None),
        "num_attention_heads": config.num_attention_heads,
        "hidden_size": config.hidden_size,
        "position_embedding_type": getattr(config, "position_embedding_type", None),
        "rope_parameters": getattr(config, "rope_parameters", None),
        "module_disagreements": disagreements,
        "cfg": engine.cfg,
    }


def reach(engine: LayaEngine) -> dict:
    """How far each layer can see, measured on the weights of a real forward.

    The notes' "window 128" is `local_attention`; the mask is
    ``abs(query - key) <= sliding_window`` with `sliding_window = local_attention // 2`,
    so the reach of a local layer is ±64 and a global one is the whole sequence.
    Measuring it instead of deriving it is the point: the boundary is what makes a
    position attackable, and a reader should not have to trust the docstring.
    """
    from layaadv.data import loaders
    from layaadv.schema import Case

    filler = " ".join(
        f"frase {i} di riempimento per allungare lo stato oltre la finestra locale" for i in range(12)
    )
    case = Case(
        id="probe",
        state=filler,
        questions={loaders.QID: {
            "type": "choice",
            "instructions": "Qual è l'intento dell'utente?",
            "criteria": {lab: "" for lab in ("alarm_set", "play_music", "calendar_query")},
        }},
        gold={loaders.QID: "play_music"},
    )
    field = engine.attention_receptive_field(case)
    by_type: dict[str, list[int]] = {}
    for row in field["layers"]:
        by_type.setdefault(row["layer_type"], []).extend(
            v for v in row["reach"].values() if v is not None
        )
    return {
        "seq_tokens": field["seq_tokens"],
        "probes": field["probes"],
        "max_reach_by_type": {k: max(v) for k, v in by_type.items() if v},
        "layers": field["layers"],
    }


def check(structure: dict, reachability: dict) -> list[str]:
    """Does the loaded model agree with what the notes claim? Named problems only."""
    problems = []
    if structure["module_disagreements"]:
        problems += structure["module_disagreements"]
    if abs(structure["fraction_global"] - CLAIMED["fraction_global"]) > 0.02:
        problems.append(
            f"la frazione globale e' {structure['fraction_global']:.4f}, la nota dice "
            f"{CLAIMED['fraction_global']:.4f}"
        )
    measured = reachability["max_reach_by_type"].get("sliding_attention")
    if measured != CLAIMED["window"] // 2:
        problems.append(
            f"la raggiungibilita' misurata di un livello locale e' ±{measured} token, la "
            f"nota dice una finestra di {CLAIMED['window']} "
            f"(cioe' ±{CLAIMED['window'] // 2})"
        )
    if reachability["max_reach_by_type"].get("full_attention") != reachability["seq_tokens"] - 1:
        problems.append(
            "un livello globale non raggiunge l'intera sequenza: la misura e' "
            f"{reachability['max_reach_by_type'].get('full_attention')} su "
            f"{reachability['seq_tokens'] - 1}"
        )
    alternating = all(
        structure["layers"][i]["layer_type"] == "full_attention" if i % 3 == 0 else
        structure["layers"][i]["layer_type"] == "sliding_attention"
        for i in range(structure["n_layers"])
    )
    if alternating != CLAIMED["alternating"]:
        problems.append("l'alternanza global/local non e' quella che la nota descrive")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--device", default="cpu", help="default: cpu, la struttura non ha bisogno della GPU")
    ap.add_argument("--json", default=None, help="destinazione (default: results/provenance/...)")
    ap.add_argument(
        "--gpu", action="store_true",
        help=(
            "misura la raggiungibilita' su GPU invece che su CPU "
            "(default: CPU, la struttura non ha bisogno della GPU)"
        ),
    )
    args = ap.parse_args()

    engine = LayaEngine(args.checkpoint, device="cuda" if args.gpu else "cpu")
    structure = layout(engine)
    reachability = reach(engine)
    problems = check(structure, reachability)
    env = io.env_fingerprint(structure["weight_sha256"])
    engine.close()

    payload = {
        "experiment": "E22_attention_structure",
        "question": (
            "la nota dice 1/3 globale e 2/3 locale con finestra 128: lo dice anche il "
            "modello caricato?"
        ),
        "claimed_in_notes": CLAIMED,
        "claimed_in_notes_before": CLAIMED_BEFORE,
        "notes": [
            "10-modello/architettura-laya.md",
            "20-attack-surface/truncation-stato.md",
        ],
        "structure": structure,
        "receptive_field": reachability,
        "env": env,
        "problems": problems,
        "note_matches_model": not problems,
    }
    out = Path(args.json) if args.json else (
        paths.RESULTS / "provenance" / "e22_attention_structure.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")

    print(
        f"[E22-struttura] {structure['n_layers']} livelli: {structure['n_global']} globali "
        f"({structure['fraction_global']:.3f}) e {structure['n_local']} locali "
        f"({structure['fraction_local']:.3f}); local_attention="
        f"{structure['local_attention_config']} (modulo {structure['windows_module']})"
    )
    print(
        "[E22-struttura] raggiungibilita' misurata sui pesi: "
        + ", ".join(f"{kind} ±{v}" for kind, v in sorted(reachability["max_reach_by_type"].items()))
        + f" su {reachability['seq_tokens']} token"
    )
    print(f"[E22-struttura] implementazione attenzione: {structure['attn_implementation']}")
    if problems:
        print(f"[E22-struttura] {len(problems)} discrepanze con le note:")
        for p in problems:
            print(f"    - {p}")
        print(f"[E22-struttura] scritto {out} (la nota va corretta)")
        return 1
    print(f"[E22-struttura] le note corrispondono al modello; scritto {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
