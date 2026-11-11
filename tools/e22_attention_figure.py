"""La figura di E22, generata dagli store.

Come `paper_figures.py`: nessun numero scritto a mano. Le curve per livello sono
**ricalcolate dai `preds.jsonl`** degli store che il payload dichiara, e confrontate con
le curve che l'esperimento ha pubblicato: se divergono, il tool fallisce invece di
disegnare la versione che gli piace. Una figura che nessuno ricalcola diverge dal corpo
alla prima correzione di un risultato.

Il tool stampa i `run_id` da cui ogni figura è costruita, perche' la provenienza di una
figura deve essere leggibile senza aprire il sorgente.

Due versioni, perche' servono a due posti: `--style paper` (serif, come il corpo del
paper) e `--style linkedin` (sfondo scuro, un'immagine sola, niente estetica terminale).

Uso:
    uv run python tools/e22_attention_figure.py [--out results/provenance]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

# Matplotlib scrive CreationDate nei PDF: fissarla rende le figure riproducibili
# byte per byte quando gli stessi store e la stessa versione del renderer sono usati.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from layaadv import paths  # noqa: E402
from layaadv.experiments import e22_attention as e22  # noqa: E402

# Il target e' lo scope pubblicato; gli altri checkpoint hanno il loro file, scelto con
# --payload. Un solo payload per figura: mischiare due checkpoint in una curva sarebbe un
# grafico di due esperimenti.
PAYLOAD = paths.RESULTS / "E22_attention.json"
TOLERANCE = 5e-5  # i valori pubblicati sono arrotondati a 6 decimali


def read_records(experiment: str, run_id: str) -> list[dict]:
    path = paths.RESULTS / experiment / "runs" / run_id / "preds.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} non esiste: la riga dichiara un run_id che non c'è negli store. "
            "Non ricalcolare a mano: la figura deve usare gli archivi, non i numeri di memoria."
        )
    return [json.loads(line) for line in path.open() if line.strip()]


def spans_of(payload: dict, arm: str) -> list[str]:
    """Gli span che un braccio ha misurato, letti dal payload e non dalle assunzioni.

    Un run con `--position head` misura un solo span: chiedere anche l'altro farebbe
    sollevare `per_layer` e la figura non si potrebbe generare da un run parziale, che
    e' un run legittimo.
    """
    entry = (payload.get("span_lengths") or {}).get(arm) or {}
    return [key[: -len("_mean_tokens")] for key in entry if key.endswith("_mean_tokens")]


def curves_from_stores(payload: dict) -> dict[str, dict[str, list[float]]]:
    """Le curve per livello, ricalcolate dai record grezzi."""
    out: dict[str, dict[str, list[float]]] = {}
    for row in payload["results"]:
        records = read_records(payload["experiment"], row["run_id"])
        for span in spans_of(payload, row["arm"]):
            for quantity in e22.QUANTITIES:
                out[f"{row['arm']}/{span}/{quantity}"] = e22.per_layer(records, span, quantity)
    return out


def check_against_payload(payload: dict, curves: dict) -> None:
    """Le curve degli store devono essere quelle che il payload ha pubblicato."""
    problems = []
    for key, published in (payload.get("profiles") or {}).items():
        if key not in curves:
            problems.append(f"{key}: pubblicata senza store")
            continue
        for a, b in zip(published, curves[key]):
            if abs(a - b) > TOLERANCE:
                problems.append(f"{key}: pubblicato {a} ricalcolato {b:.6f}")
                break
    if problems:
        raise ValueError("le figure non corrispondono agli store:\n  " + "\n  ".join(problems[:8]))


def draw(curves: dict, layer_types: list[str], *, quantity: str, style: str) -> plt.Figure:
    """Due pannelli, uno per posizione: la frase, la frase neutra, niente.

    Per posizione e non per tipo di attenzione: il confronto che l'esperimento si propone
    e' «stessa frase, due posizioni», quindi le colonne sono le posizioni e dentro ogni
    colonna stanno i tre bracci. I livelli globali sono evidenziati, perche' la previsione
    registrata diceva che la differenza sarebbe stata portata da loro.
    """
    dark = style == "linkedin"
    face = "#12151c" if dark else "white"
    ink = "#e8ecf4" if dark else "black"
    grid = "#39414f" if dark else "0.85"
    if dark:
        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.labelcolor": ink,
            "xtick.color": ink,
            "ytick.color": ink,
            "text.color": ink,
            "axes.edgecolor": grid,
            "figure.facecolor": face,
            "axes.facecolor": face,
            "savefig.facecolor": face,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        })
    else:
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        })

    layers = list(range(len(layer_types)))
    global_at = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    # (chiave del profilo, etichetta, colore). L'ordine e' quello della lettura.
    series = [
        ("attack", "frase che nomina l'opzione", "#c44e52"),
        ("neutral", "frase neutra, stessa cornice", "#dd8452"),
        ("clean", "nessuna frase", "#4c72b0"),
    ]
    titles = {
        "all": "Attenzione che la frase riceve",
        "markers": "Attenzione dai marker delle opzioni",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0) if dark else (6.7, 2.6), sharex=True)
    for ax, span in zip(axes, ("head", "tail")):
        for index in global_at:
            ax.axvspan(index - 0.45, index + 0.45, color=grid, alpha=0.4 if dark else 0.6,
                       linewidth=0)
        for arm, label, color in series:
            # Il braccio pulito non ha suffisso di posizione (misura i due span in un
            # forward), gli avversari si': da qui le due forme della chiave.
            curve = curves.get(f"{arm}-{span}/{span}/{quantity}") or curves.get(
                f"{arm}/{span}/{quantity}"
            )
            if curve is None:
                continue
            ax.plot(layers, curve, marker="o", markersize=2.6, linewidth=1.3, label=label,
                    color=color)
        ax.set_xlabel("livello dell'encoder")
        ax.set_title(f"frase in {'testa' if span == 'head' else 'coda'}")
        ax.grid(linewidth=0.4, color=grid, alpha=0.6)
        ax.set_xticks(range(0, len(layers), 3))
        ax.legend(loc="upper left", fontsize=9 if dark else 6.5)
    axes[0].set_ylabel("massa d'attenzione sullo span")
    fig.suptitle(titles[quantity], fontsize=12 if dark else 9)
    # Il titolo sta sopra i due pannelli e le colonne sono il confronto: lo spazio vuoto
    # serve a non farlo sovrapporre al titolo dei pannelli.
    fig.tight_layout(rect=(0, 0, 1, 0.94 if dark else 0.93))
    return fig


def draw_distance(curves: dict, layer_types: list[str], *, quantity: str,
                  style: str) -> plt.Figure:
    """La figura della distanza: vicina contro lontana, per posizione.

    Due pannelli (testa/coda), e in ognuno la stessa frase alla distanza vicina e a quella
    lontana. La distanza lontana mette la frase fuori dalla finestra ±64 dei livelli locali,
    quindi la differenza fra le due curve e' la parte che solo i livelli globali possono
    leggere: e' il disegno che rende decidibile la domanda globale/locale.
    """
    dark = style == "linkedin"
    face = "#12151c" if dark else "white"
    ink = "#e8ecf4" if dark else "black"
    grid = "#39414f" if dark else "0.85"
    if dark:
        plt.rcParams.update({
            "font.family": "DejaVu Sans", "font.size": 13, "axes.labelcolor": ink,
            "xtick.color": ink, "ytick.color": ink, "text.color": ink,
            "axes.edgecolor": grid, "figure.facecolor": face, "axes.facecolor": face,
            "savefig.facecolor": face, "legend.frameon": False,
            "axes.spines.top": False, "axes.spines.right": False,
        })
    else:
        plt.rcParams.update({
            "font.family": "serif", "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif", "font.size": 9,
            "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
        })

    layers = list(range(len(layer_types)))
    global_at = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    near_keys = {"head": "attack-head/head", "tail": "attack-tail/tail"}
    far_keys = {"head": "attack-head-far/head", "tail": "attack-tail-far/tail"}
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0) if dark else (6.7, 2.6), sharex=True)
    for ax, span in zip(axes, ("head", "tail")):
        for index in global_at:
            ax.axvspan(index - 0.45, index + 0.45, color=grid, alpha=0.4 if dark else 0.6,
                       linewidth=0)
        for key, label, color, style_line in (
            (near_keys[span], "frase vicina (~45 token dai marker)", "#c44e52", "-"),
            (far_keys[span], "frase lontana (>240 token, fuori ±64)", "#4c72b0", "-"),
            (f"clean-near/{span}/{quantity}", "controllo vicino, nessuna frase",
             "#8c8c8c", "--"),
        ):
            curve = curves.get(f"{key}/{quantity}")
            if curve is None:
                continue
            ax.plot(layers, curve, marker="o", markersize=2.6, linewidth=1.3, label=label,
                    color=color, linestyle=style_line)
        ax.set_xlabel("livello dell'encoder")
        ax.set_title(f"frase in {'testa' if span == 'head' else 'coda'}")
        ax.grid(linewidth=0.4, color=grid, alpha=0.6)
        ax.set_xticks(range(0, len(layers), 3))
        ax.legend(loc="upper left", fontsize=9 if dark else 6.5)
    axes[0].set_ylabel("massa d'attenzione sullo span")
    fig.suptitle({
        "all": "Attenzione che la frase riceve",
        "markers": "Attenzione dai marker delle opzioni",
    }[quantity], fontsize=12 if dark else 9)
    fig.tight_layout(rect=(0, 0, 1, 0.94 if dark else 0.93))
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(paths.Path("results") / "provenance"))
    parser.add_argument("--style", choices=["paper", "linkedin"], default="paper")
    parser.add_argument("--payload", default=None, help="file dei risultati (default: E22_attention.json)")
    args = parser.parse_args()

    payload_path = pathlib.Path(args.payload) if args.payload else PAYLOAD
    if not payload_path.exists():
        print(f"{payload_path} non esiste: prima `uv run adv e22`")
        return 1
    payload = json.loads(payload_path.read_text())
    curves = curves_from_stores(payload)
    check_against_payload(payload, curves)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "_linkedin" if args.style == "linkedin" else ""
    near_curves: dict[str, list[float]] = {}
    if payload.get("experiment", "").startswith("E22b"):
        # Le curve vicine stanno negli store di E22: ricalcolate da la', non copiate, e
        # dichiarate nel payload del run da cui vengono.
        near_payload_path = paths.RESULTS / payload["near_from"]["payload"]
        near_payload = json.loads(near_payload_path.read_text())
        near_curves = curves_from_stores(near_payload)
        for key, published in (payload.get("near_profiles") or {}).items():
            if key not in near_curves:
                raise ValueError(f"{key}: curva vicina pubblicata senza store")
            for a, b in zip(published, near_curves[key]):
                if abs(a - b) > TOLERANCE:
                    raise ValueError(f"{key}: pubblicato {a} ricalcolato {b:.6f}")

    # Il nome del file porta il checkpoint: due scope nello stesso nome sovrascriverebbero
    # le figure di uno dei due, che e' il difetto per cui gli scope di E03 hanno il nome
    # lungo. Il target resta il nome corto, per non rompere i riferimenti gia' scritti.
    slug = payload["axes"]["checkpoint"].rsplit("/", 1)[-1]
    prefisso = "e22b_distanza" if payload.get("experiment", "").startswith("E22b") else (
        "e22_attenzione" if slug == "laya-typed-decisions" else f"e22_attenzione__{slug}"
    )
    for quantity in e22.QUANTITIES:
        if payload.get("experiment", "").startswith("E22b"):
            fig = draw_distance({**near_curves, **curves}, payload["layer_types"],
                                quantity=quantity, style=args.style)
        else:
            fig = draw(curves, payload["layer_types"], quantity=quantity, style=args.style)
        path = out / f"{prefisso}_{quantity}{suffix}.pdf"
        fig.savefig(path)
        plt.close(fig)
        print(f"scritto {path}")

    print("run da cui la figura e' costruita:")
    for row in payload["results"]:
        print(f"  {row['arm']:<18} {row['run_id']}  accuracy {row['accuracy']:.4f}")
    with_asr = [r for r in payload["results"] if r.get("asr") is not None]
    if with_asr:
        print("  (ASR per braccio: "
              + ", ".join(f"{r['arm']}={r['asr']:.4f}" for r in with_asr) + ")")
    if near_curves:
        print(f"  (curve vicine dagli store di {payload['near_from']['payload']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
