"""E22b — La distanza decide cosa possono leggere i livelli locali?

E22 ha misurato la frase in testa e in coda su `massive-it` a K=16, dove lo stato è corto
(~24 token): entrambi gli span stavano **dentro** la finestra ±64 dei livelli locali, quindi
la distinzione globale/locale non poteva esprimersi e la previsione H1 andava dichiarata non
decidable. Questo modulo la rende decidibile: stessa frase, stessi casi, stessa K, con lo
stato **allungato** da 200 token di filler neutro (quello di E04) messo *davanti* alla frase,
così la frase resta leggibile e non troncata.

Disegno 2×2: posizione (testa/coda) × distanza (vicina/lontana). I bracci **vicini** non
vengono rimisurati: vengono dagli store di E22 e sono appaiati per id, quindi la differenza
fra distanze è sullo stesso caso, non fra campioni.

Previsioni registrate prima in 40-esperimenti/E22-attenzione-laya.md: **H3** la massa dai
livelli **locali** sullo span crolla alla distanza lontana e quella dai **globali** no;
**H4** se i globali bastano, l'ASR regge alla distanza lontana.

Uso:
    uv run adv e22b                        # N pieno, filler 200 token
    uv run adv e22b --filler 400 --limit 40
"""

from __future__ import annotations

import json
from typing import Any

from layaadv import io, paths
from layaadv.data import loaders
from layaadv.experiments import e22_attention as e22
from layaadv.experiments.e03_option_injection import inject_option
from layaadv.experiments.e04_truncation import push_out_state
from layaadv.metrics.core import asr
from layaadv.model.engine import DEFAULT_CHECKPOINT, open_from_args

EXPERIMENT = "E22b_attention_distance"
NEAR_PAYLOAD = paths.RESULTS / "E22_attention.json"
FILLER = 200
QID = loaders.QID

# I bracci della geometria lontana. `clean` non ha suffisso di posizione perche' lo span si
# misura dallo stato: per il controllo in testa lo stato e' `filler + pulito`, in coda
# `pulito + filler`, e lo span e' il primo/ultimo tratto dello stato, cioe' il punto che
# occupava la frase.
ARMS = ("attack", "neutral")


def far_state(clean: str, text: str, position: str, filler: int) -> str:
    """Lo stato con la frase alla distanza lontana.

    Il filler sta **davanti** alla frase, non dietro: in coda verrebbe troncato per primo
    (`build_sequence` taglia la coda destra) e la frase sparirebbe, che e' esattamente ciò
    che questo esperimento non deve misurare.
    """
    if position == "head":
        return f"{push_out_state('', filler)} {text} {clean}"
    return f"{clean} {push_out_state('', filler)} {text}"


def clean_state(clean: str, position: str, filler: int) -> str:
    """Il controllo posizionale: stessa geometria, nessuna frase."""
    if position == "head":
        return push_out_state(clean, filler)
    return f"{clean} {push_out_state('', filler)}"


def near_records(arm: str) -> list[dict]:
    """I record di un braccio vicino, dagli store di E22 (appaiati per id)."""
    payload = json.loads(NEAR_PAYLOAD.read_text())
    row = next((r for r in payload["results"] if r["arm"] == arm), None)
    if row is None:
        raise SystemExit(f"{NEAR_PAYLOAD} non ha il braccio {arm!r}: gira prima `adv e22`")
    path = paths.RESULTS / payload["experiment"] / "runs" / row["run_id"] / "preds.jsonl"
    if not path.exists():
        raise SystemExit(f"store mancante: {path}")
    return [json.loads(line) for line in path.open() if line.strip()]


def run(args: Any) -> dict:
    def log(message: str) -> None:
        print(message, flush=True)

    limit = getattr(args, "limit", None)
    filler = int(getattr(args, "filler", None) or FILLER)
    dataset = getattr(args, "dataset", None) or e22.DATASET
    checkpoint = (getattr(args, "checkpoint", None) or [DEFAULT_CHECKPOINT])[0]

    engine = open_from_args(args, checkpoint)
    identity = engine.run_identity()
    env = io.env_fingerprint(engine.weight_sha256())
    layout = engine.attention_structure()
    layer_types = [row["layer_type"] for row in layout["layers"]]
    log(
        f"[E22b] {checkpoint} su {dataset}, K=16, filler {filler} token: "
        f"{layout['n_global']} livelli globali e {layout['n_local']} locali"
    )

    cases, ds, _ = loaders.load_cases(dataset, k=e22.K, limit=limit, seed=e22.SEED)
    if not cases:
        raise SystemExit(f"nessun caso caricato: manca data/raw per {dataset}?")
    targets = {c.id: inject_option(c.questions[QID], c.gold[QID]).replace("_", " ")
               for c in cases}
    sentences = {c.id: e22.sentence_for(targets[c.id]) for c in cases}

    common = {
        "experiment": EXPERIMENT,
        "checkpoint": checkpoint,
        "dataset": dataset,
        "dataset_sha256": ds.file_sha256(),
        "subset_hash": ds.subset_hash(cases),
        "k": e22.K,
        "n_cases": len(cases),
        "seed": e22.SEED,
        "batch_size": engine.batch_size,
        "chunk": e22.CHUNK,
        **{i: v for i, v in identity.items() if v is not None},
        "weight_sha256": engine.weight_sha256(),
        "attn_implementation": "eager",
        "filler_tokens": filler,
    }
    resume = {"resume": not getattr(args, "no_resume", False),
              "force_new": getattr(args, "force_new", False)}
    measure = e22.arm_measure(engine, EXPERIMENT, common, resume, QID)

    # La lunghezza dello span viene dal tokenizzatore, come in E22: la frase dentro lo
    # stato allungato non ha la stessa lunghezza che ha da sola.
    def tokens_of(case: Any, position: str) -> int:
        state = far_state(case.state, sentences[case.id], position, filler)
        armed = case.__class__(id=case.id, state=state, questions=case.questions, gold=case.gold)
        return engine.attention_span_tokens(armed, position=position,
                                           sentence=sentences[case.id],
                                           reference_state=case.state)

    arms: dict[str, tuple[list, dict, object]] = {}
    # Il riferimento pulito lo gira E22b, non lo riusa da E22: l'ASR dei bracci lontani deve
    # essere ricalcolabile dagli store di *questo* esperimento, o il verifier non lo
    # verificherebbe. Il cleanup-vs-lontano si legge comunque per confronto appaiato.
    specs = [{"spans": [e22.span_entry(p, sentence=None, reference=None,
                                       tokens=tokens_of(c, p))
                        for p in ("head", "tail")]} for c in cases]
    records, metrics, store = measure("clean-near", list(cases), specs)
    arms["clean-near"] = (records, metrics, store)
    for arm in ARMS:
        for position in ("head", "tail"):
            armed, specs = [], []
            for case in cases:
                text = e22.NEUTRAL if arm == "neutral" else sentences[case.id]
                armed.append(case.__class__(
                    id=case.id, state=far_state(case.state, text, position, filler),
                    questions=case.questions, gold=case.gold))
                specs.append({"spans": [e22.span_entry(position, sentence=text,
                                                       reference=case.state,
                                                       tokens=tokens_of(case, position))]})
            records, metrics, store = measure(f"{arm}-{position}-far", armed, specs)
            arms[f"{arm}-{position}-far"] = (records, metrics, store)
    for position in ("head", "tail"):
        armed = [c.__class__(id=c.id, state=clean_state(c.state, position, filler),
                             questions=c.questions, gold=c.gold) for c in cases]
        # Il controllo non ha frase: lo span e' posizionale, della lunghezza che la frase
        # avrebbe avuto in quel punto.
        specs = [{"spans": [e22.span_entry(position, sentence=None, reference=None,
                                          tokens=engine.attention_span_tokens(
                                              c, position=position, sentence=sentences[c.id],
                                              reference_state=c.state))]} for c in cases]
        records, metrics, store = measure(f"clean-{position}-far", armed, specs)
        arms[f"clean-{position}-far"] = (records, metrics, store)

    # I vicini: dagli store di E22, appaiati per id. Non rimisurati di proposito: la
    # differenza fra distanze deve essere sullo stesso caso.
    near = {arm: near_records(arm) for arm in
            ("clean", "neutral-head", "neutral-tail", "attack-head", "attack-tail")}

    def spans_of(name: str, records: list) -> tuple:
        """Gli span che un braccio ha misurato, letti dai record e non dal nome."""
        return tuple(sorted(records[0]["attn"])) if records else ()

    def row_of(name: str, records: list, metrics: dict, store) -> dict:
        mismatches = sum(1 for r in records if not r["attn_meta"]["argmax_match"])
        return {
            "arm": name,
            "distance": "far",
            "position": span,
            "n_cases": len(records),
            "run_id": store.run_id,
            "accuracy": metrics["accuracy"],
            "n_decisions": metrics["n_decisions"],
            "backend": identity["backend"],
            "device": identity["device"],
            "quantization": identity["quantization"],
            "k": e22.K,
            "seed": e22.SEED,
            "argmax_mismatch_eager_vs_stock": mismatches,
            "attention": {
                span: {
                    "per_layer_all": e22.per_layer(records, span, "all"),
                    "per_layer_markers": e22.per_layer(records, span, "markers"),
                    "spread_all": e22.dispersion(records, span, "all"),
                }
                for span in spans_of(name, records)
            },
        }

    clean_near = arms["clean-near"][0]
    rows, profiles = [], {}
    for name, (records, metrics, store) in arms.items():
        for span in spans_of(name, records):
            for quantity in e22.QUANTITIES:
                profiles[f"{name}/{span}/{quantity}"] = e22.per_layer(records, span, quantity)
        row = row_of(name, records, metrics, store)
        if name != "clean-near":
            rate, denominator = asr(clean_near, records)
            row.update({
                "clean_run_id": arms["clean-near"][2].run_id,
                "adv_run_id": store.run_id,
                "asr": rate,
                "asr_denominator": denominator,
                "accuracy_clean": arms["clean-near"][1]["accuracy"],
                "drop_pp": round(100 * (arms["clean-near"][1]["accuracy"] - metrics["accuracy"]), 2),
                "asr_descrizione": e22.describe_asr(rate, denominator),
            })
        rows.append(row)
    # Le curve dei vicini ricalcolate dagli store, non copiate dai valori pubblicati.
    near_profiles: dict[str, list[float]] = {}
    for arm, records in near.items():
        span = arm.split("-", 1)[1] if "-" in arm else None
        spans = ("head", "tail") if span is None else (span,)
        for one in spans:
            for quantity in e22.QUANTITIES:
                near_profiles[f"{arm}/{one}/{quantity}"] = e22.per_layer(records, one, quantity)

    comparisons = {}
    for quantity in e22.QUANTITIES:
        for position in ("head", "tail"):
            # La domanda principale: stessa frase, stessa posizione, due distanze.
            if f"attack-{position}-far" in arms and f"attack-{position}" in near:
                comparisons[f"far_vs_near__{position}/{quantity}"] = e22.paired(
                    arms[f"attack-{position}-far"][0], near[f"attack-{position}"],
                    position, position, quantity, layer_types)
            # Posizione a distanza fissata: la firma dei tipi di livello.
            if "attack-head-far" in arms and "attack-tail-far" in arms:
                comparisons[f"far_tail_vs_far_head__{quantity}"] = e22.paired(
                    arms["attack-tail-far"][0], arms["attack-head-far"][0],
                    "tail", "head", quantity, layer_types)
            if f"neutral-{position}-far" in arms:
                comparisons[f"neutral_vs_attack__{position}/far/{quantity}"] = e22.paired(
                    arms[f"neutral-{position}-far"][0], arms[f"attack-{position}-far"][0],
                    position, position, quantity, layer_types, same_length=True)
                comparisons[f"attack_vs_clean__{position}/far/{quantity}"] = e22.paired(
                    arms[f"attack-{position}-far"][0], arms[f"clean-{position}-far"][0],
                    position, position, quantity, layer_types, same_length=True)

    payload = {
        "experiment": EXPERIMENT,
        "question": (
            "la distanza dai marker decide cosa possono leggere i livelli locali, e basta "
            "quelli globali a far funzionare l'attacco?"
        ),
        "prediction_note": (
            "40-esperimenti/E22-attenzione-laya.md: H3 la massa dai locali crolla alla "
            "distanza lontana e quella dai globali no; H4 se i globali bastano l'ASR regge"
        ),
        "axes": {
            "checkpoint": checkpoint,
            "dataset": dataset,
            "k": e22.K,
            "n_cases": len(cases),
            "seed": e22.SEED,
            "filler_tokens": filler,
            "limit": limit,
            "disegno": "posizione (testa/coda) x distanza (vicina dagli store E22 / lontana qui)",
        },
        "attention_layout": {
            "n_layers": layout["n_layers"],
            "n_global": layout["n_global"],
            "n_local": layout["n_local"],
            "fraction_global": round(layout["fraction_global"], 4),
            "local_attention": layout["local_attention"],
            "sliding_window": layout["sliding_window"],
            "attn_implementation": layout["attn_implementation"],
            "extraction_implementation": "eager",
        },
        "layer_types": layer_types,
        "span_lengths": e22.span_lengths(arms, ("head", "tail")),
        "profiles": profiles,
        "near_profiles": near_profiles,
        "comparisons": comparisons,
        "results": rows,
        "near_from": {
            "payload": NEAR_PAYLOAD.name,
            "arms": list(near),
            "nota": "appaiati per id: la differenza fra distanze e' sullo stesso caso",
        },
        "env": env,
    }
    if not rows:
        raise SystemExit("nessun braccio misurato: nessun file di risultati e' stato toccato.")
    slug = f"{EXPERIMENT}__{checkpoint.rsplit('/', 1)[-1]}"
    io.write_result(slug, payload)
    log(f"[E22b] wrote results/{slug}.json")
    return payload
