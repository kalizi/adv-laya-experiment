"""Unified CLI: ``uv run adv <command>``.

Le importazioni degli esperimenti sono lazy dentro gli handler, così
``adv --help`` non carica torch/transformers.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Sequence


def _add_model_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default="laya",
        metavar="BACKEND",
        help=(
            "backend hot-swappabile: 'laya' (nativo, default), 'laya-onnx' (stessa testa a "
            "marker via ONNX Runtime), 'deberta-onnx' (encoder a coppie), 'clm-llama' "
            "(CLM servito da llama.cpp), 'd1-liquid' (Liquid AI d1, verbalizer). "
            "Vedi layaadv.model.backends"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        metavar="REPO",
        help="checkpoint HF o path locale (ripetibile; default: quelli dell'esperimento)",
    )
    parser.add_argument("--device", default=None, help="cpu | cuda:N (default: automatico)")
    parser.add_argument(
        "--compile",
        action="store_true",
        help=(
            "torch.compile sull'encoder: 1.44x misurato, al costo di ~55s di warmup e "
            "di numeri diversi (max |Δconf| 0.0031). Cambia il run_id, quindi un run "
            "compilato e uno eager non si confrontano"
        ),
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "percorso TileLang di laya (kernel fusi, CUDA graph per shape). Misurato "
            "sulla 3080 con il toolchain nvcc 13.4: 2.42x dove il batch e' scarso "
            "(1 stato per call, cioe' massive-it/massive-en/banking77) e 0.80x dove il "
            "batch e' pieno (typed-decisions): con batch gia' saturo i CUDA graph per "
            "shape costano piu' di quanto il fusion guadagni. Cambia i numeri (max "
            "|Δconf| 0.0216), quindi cambia il run_id. Se TileLang non e' installato "
            "laya ricade sul forward stock: `adv e12-surface` lo dichiara"
        ),
    )
    parser.add_argument(
        # 16: massimo senza OOM su RTX 3080 10GB con stati ~1000 token (misurato);
        # 32 non è più veloce e fa scattare il fallback CPU di laya.
        "--batch-size",
        type=int,
        default=16,
        help="stati per forward pass (default 16)",
    )


def _add_run_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--limit", type=int, default=None, help="max casi: prova a piccola scala")
    parser.add_argument(
        "--min-agreement",
        type=float,
        default=None,
        dest="min_agreement",
        help="filtro label_agreement sulle questions (dichiararlo nel report)",
    )
    parser.add_argument("--no-resume", action="store_true", help="rifiuta un run già esistente")
    parser.add_argument("--force-new", action="store_true", help="crea un run nuovo, non riprendere")


# nome comando -> modulo (i nomi dei file seguono quelli del vault: E00-setup...)
EXPERIMENTS = {
    "e00": "layaadv.experiments.e00_setup",
    "e01": "layaadv.experiments.e01_baseline",
    "e02": "layaadv.experiments.e02_option_count",
    "e03": "layaadv.experiments.e03_option_injection",
    "e04": "layaadv.experiments.e04_truncation",
    "e12b": "layaadv.experiments.e12_rare_syn",
    "e12-surface": "layaadv.experiments.e12_surface",
    "e13": "layaadv.experiments.e13_label_semantics",
    "e15": "layaadv.experiments.e15_quantization",
    "e16": "layaadv.experiments.e16_calibration_defense",
    "e17": "layaadv.experiments.e17_configuration",
    "e18": "layaadv.experiments.e18_nomination_detector",
    "e19": "layaadv.experiments.e19_detector_labels",
    "e20": "layaadv.experiments.e20_paraphrase_evasion",
    "e22": "layaadv.experiments.e22_attention",
    "e22b": "layaadv.experiments.e22b_attention_distance",
    "e23": "layaadv.experiments.e23_mechanism",
    "e23b": "layaadv.experiments.e23b_heads",
    "e23c": "layaadv.experiments.e23c_channel",
    "e23d": "layaadv.experiments.e23d_form",
    "e23e": "layaadv.experiments.e23e_defense",
    "e24": "layaadv.experiments.e24_verbalizer",
}


def _experiment(sub, name: str, help_: str, *, run_flags: bool = True) -> argparse.ArgumentParser:
    parser = sub.add_parser(name, help=help_, description=help_)
    _add_model_flags(parser)
    if run_flags:
        _add_run_flags(parser)
    parser.set_defaults(func=_handler(EXPERIMENTS[name]))
    return parser


def _handler(module_path: str, attr: str = "run"):
    def handle(args) -> dict:
        module = importlib.import_module(module_path)
        return getattr(module, attr)(args)

    return handle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adv", description="Adversarial Laya study — esperimenti e tooling"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    _experiment(sub, "e00", "setup gate: smoke + verifiche obbligatorie", run_flags=False)
    _experiment(sub, "e01", "baseline pulita su typed-decisions (3 checkpoint)")
    _experiment(sub, "e02", "option count collapse (diagnostic)")
    e03 = _experiment(sub, "e03", "option injection nello stato (solo stato; validità semantica da auditare)")
    e03.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "disegno delle opzioni per riga (default 0). Il gold resta in posizione "
            "variabile per costruzione: cambiare il seed non invalida l'ASR, cambia gli "
            "option set, quindi serve a misurare la varianza della curva K"
        ),
    )
    e03.add_argument(
        "--dataset",
        default=None,
        metavar="SLUG",
        help="dataset intent su cui iniettare (default: massive-it; es. massive-en)",
    )
    e03.add_argument(
        "--k",
        type=int,
        action="append",
        metavar="K",
        help="opzioni per domanda (ripetibile; default: sweep 4..40, fermo al budget della testa)",
    )
    e03.add_argument(
        "--template",
        action="append",
        choices=["prepend", "append"],
        help="dove iniettare l'opzione distratore (ripetibile; default: entrambi)",
    )
    e03.add_argument(
        "--position",
        action="append",
        choices=["head", "tail"],
        help="posizione della frase (ripetibile; default: posizione storica del template)",
    )
    e03.add_argument(
        "--pad",
        type=int,
        default=0,
        metavar="TOK",
        help=(
            "token di filler neutro davanti allo stato (E03 x E04): con un'iniezione in "
            "coda la spinge fuori da max_len e la fa troncare (default 0)"
        ),
    )
    _experiment(sub, "e04", "attacco di troncamento")
    _experiment(sub, "e13", "il testo delle opzioni e' un canale di informazione")
    e15 = _experiment(sub, "e15", "la sicurezza sopravvive alla riduzione di precisione?")
    e15.add_argument(
        "--dataset",
        default=None,
        metavar="SLUG",
        help=(
            "misura tutte le famiglie su un solo dataset, es. `adv e15 --dataset banking77` per "
            "laya. Default: nessuno, cioe' il dataset competente di ciascuna famiglia (laya su "
            "massive-it, deberta-onnx su banking77). Il default era 'banking77' ed era innocuo "
            "solo perche' nessuno leggeva il flag: collegarlo avrebbe spostato laya su "
            "banking77, dove va a 0.03"
        ),
    )
    e15.add_argument(
        "--replicates",
        type=int,
        default=1,
        help=(
            "ripeti tutta la matrice con semi diversi, e riporta escursione e se le repliche "
            "attraversano la soglia 0.50 (default 1: nessuna replica). Il seed ridisegna le "
            "opzioni per riga, quindi la dispersione e' del benchmark, non del modello"
        ),
    )
    e15.add_argument(
        "--variant",
        choices=["none", "int8", "nf4", "fp32", "fp16", "q4", "q4f16"],
        default=None,
        help=(
            "una sola precisione. Accetta i nomi delle due famiglie e non li scambia: none/int8/"
            "nf4 sono di laya, fp32/fp16/q4/q4f16 di deberta-onnx, e una combinazione impossibile "
            "solleva elencando cosa c'e'. Default: tutte, che e' quello che vuole `adv e15` e "
            "quello che produce la tabella di confronto"
        ),
    )
    e15.add_argument(
        "--k",
        default="16",
        help="opzioni per domanda, o curva separate da virgola: 4,8,16,32 (default 16)",
    )
    e15.add_argument(
        "--only",
        default=None,
        metavar="LISTA",
        help="piu' precisioni per volta, separate da virgola (es. fp32,q4)",
    )
    e12s = _experiment(
        sub, "e12-surface",
        "superficie di attacco: otto operatori puri sullo stato, con archivio e braccio di controllo",
    )
    e12s.add_argument(
        "--operator",
        default=None,
        metavar="NOME",
        help="un solo operatore invece di tutti (vedi REGISTRY in e12_surface)",
    )
    e16 = _experiment(
        sub, "e16", "la calibrazione, come difesa, chiude il guardrail che l'attacco rompe?"
    )
    e17 = _experiment(
        sub, "e17", "esiste una configurazione di esecuzione che chiude il buco nel guardrail?"
    )
    e16.add_argument(
        "--replicates",
        type=int,
        default=None,
        help=(
            "repliche della difesa (default 1): ogni replica fitta la temperatura su un "
            "quarto diverso dei casi puliti e valuta sugli altri tre quarti, quindi la "
            "temperatura e gli intervalli di valutazione cambiano"
        ),
    )
    e16.add_argument("--template", choices=["prepend", "append"], default="append",
                     help="frase dell'attacco (default: append, braccio principale E03)")
    e16.add_argument("--position", choices=["head", "tail"], default=None,
                     help="posizione della frase (default: quella del template)")
    e16.add_argument("--split-seed", type=int, default=1000,
                     help="seme del primo split fit/held-out (default 1000)")
    e18 = _experiment(
        sub, "e18", "un detector sulla nominazione esplicita dell'opzione chiude il residuo?"
    )
    e18.add_argument(
        "--dataset", default=None, metavar="SLUG",
        help="dataset su cui applicare il detector (default: massive-it)",
    )
    e19 = _experiment(
        sub, "e19", "l'etichetta del detector conta piu' del fatto che sia un detector?"
    )
    e19.add_argument("--dataset", default=None, metavar="SLUG", help="default: massive-it")
    e19.add_argument("--only", action="append", help="variante (ripetibile; default: tutte)")
    e20 = _experiment(
        sub, "e20", "quante riformulazioni dell'attacco restano efficaci e non viste?"
    )
    e20.add_argument("--dataset", default=None, metavar="SLUG", help="default: massive-it")
    e20.add_argument("--only", action="append", help="variante (ripetibile; default: tutte)")
    e22 = _experiment(
        sub, "e22", "quanta attenzione prende la frase iniettata, in testa e in coda?"
    )
    e22.add_argument(
        "--arm", action="append", choices=["clean", "neutral", "attack"],
        help="braccio da misurare (ripetibile; default: tutti e tre)",
    )
    e22.add_argument(
        "--position", action="append", choices=["head", "tail"],
        help="posizione della frase (ripetibile; default: entrambe)",
    )
    e22.add_argument(
        "--dataset", default=None, metavar="SLUG",
        help="dataset su cui misurare (default: massive-it, come lo scope E03 congelato)",
    )
    e22b = _experiment(
        sub, "e22b", "la distanza dai marker decide cosa leggono i livelli locali?"
    )
    e22b.add_argument(
        "--filler", type=int, default=None, metavar="TOK",
        help=(
            "token di filler neutro messi davanti alla frase (default 200), per spostarla "
            "oltre la finestra ±64 dei livelli locali"
        ),
    )
    e22b.add_argument(
        "--dataset", default=None, metavar="SLUG", help="default: massive-it"
    )
    e23 = _experiment(
        sub, "e23",
        "a che profondita' l'iniezione entra negli stati che la testa legge ai marker?",
    )
    e23.add_argument(
        "--markers", action="append", choices=["all", "gold", "target"],
        help=(
            "sottoinsieme di marker da ripristinare (ripetibile; default: tutti e tre). "
            "'target' e' il marker dell'opzione nominata, 'gold' quello dell'opzione "
            "corretta: confrontarli separa «l'attacco scrive nel bersaglio» da «danneggia "
            "l'oro» (P2 della nota)"
        ),
    )
    e23.add_argument(
        "--dataset", default=None, metavar="SLUG",
        help="dataset su cui misurare (default: massive-it, come lo scope E03 congelato)",
    )
    e23b = _experiment(
        sub, "e23b",
        "quali teste dell'attenzione portano l'effetto dell'iniezione?",
    )
    e23b.add_argument(
        "--screen", type=int, default=None, metavar="N",
        help=(
            "casi su cui knockare tutte le teste (default 150, presi a passo fisso "
            "sull'intero dataset: i primi casi di massive-it sono i piu' difficili)"
        ),
    )
    e23b.add_argument(
        "--confirm", type=int, default=None, metavar="K",
        help="teste da confermare su N pieno (default 10)",
    )
    e23b.add_argument(
        "--dataset", default=None, metavar="SLUG", help="default: massive-it"
    )
    e23c = _experiment(
        sub, "e23c",
        "la nominazione conta solo nello stato, o in tutti i campi del formato?",
    )
    e23c.add_argument(
        "--field", action="append",
        choices=["state", "instructions", "criteria-target", "criteria-other"],
        help="campo del formato in cui mettere la frase (ripetibile; default: tutti e quattro)",
    )
    e23c.add_argument(
        "--dataset", default=None, metavar="SLUG", help="default: massive-it"
    )
    e23d = _experiment(
        sub, "e23d",
        "la nominazione è semantica (riferimento) o di stringa (token)?",
    )
    e23d.add_argument(
        "--form", action="append",
        choices=["exact", "reordered", "spaceless", "typo", "negated", "neutral"],
        help=(
            "forma della frase (ripetibile; default: tutte e sei). `negated` conserva la "
            "stringa e inverte il riferimento: è la forma che separa i due meccanismi"
        ),
    )
    e23d.add_argument(
        "--dataset", default=None, metavar="SLUG", help="default: massive-it"
    )
    e23e = _experiment(
        sub, "e23e",
        "un rivelatore causale che non legge le parole separa l'attacco dal legittimo?",
    )
    e23e.add_argument(
        "--dataset", action="append", metavar="SLUG",
        help="dataset su cui misurare (ripetibile; default: massive-it e massive-en)",
    )
    e17.add_argument("--only", action="append", help="configurazione (ripetibile; default: tutte)")
    e17.add_argument("--dataset", default=None, metavar="SLUG", help="default: massive-it")
    e17.add_argument(
        "--k", type=int, action="append", metavar="K", help="ripetibile; default: 16"
    )
    e17.add_argument(
        "--template", default=None, choices=["prepend", "append"], help="default: prepend"
    )
    e16.add_argument(
        "--dataset",
        default=None,
        metavar="SLUG",
        help=(
            "dataset su cui misurare la difesa. Default: massive-it. Vale anche "
            "`typed-decisions`, che ha cinque domande per stato: li usa la prima di tipo choice, "
            "perche' e' la primitiva su cui la temperatura e' calibrata"
        ),
    )
    e24 = _experiment(
        sub, "e24",
        "il verbalizer e' una superficie diversa? nominare il codice invece dell'etichetta",
    )
    e24.add_argument(
        "--dataset", default=None, metavar="SLUG",
        help="dataset su cui iniettare (default: massive-it, come lo scope E03 congelato)",
    )
    e24.add_argument(
        "--k",
        type=int,
        action="append",
        metavar="K",
        help="opzioni per domanda (ripetibile; default: sweep 4..40)",
    )

    datasets = sub.add_parser("datasets", help="gestione dataset (download, manifest)")
    ds_sub = datasets.add_subparsers(dest="subcommand", required=True)
    download = ds_sub.add_parser("download", help="scarica i dataset e rigenera il manifest")
    download.set_defaults(func=_handler("layaadv.datasets.download"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = args.func(args)
    except NotImplementedError as e:
        print(f"non implementato: {e}", file=sys.stderr)
        return 2
    # Un handler puo' restituire un dict (contratto degli esperimenti) o un int
    # (`datasets download` restituisce 1 se un download è fallito). Guardare solo il
    # dict faceva uscire 0 da un download fallito: indistinguibile dal successo.
    if isinstance(payload, bool):
        return 0 if payload else 1
    if isinstance(payload, int):
        return payload
    if isinstance(payload, dict) and payload.get("gate_pass") is False:
        print("GATE FAIL", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
