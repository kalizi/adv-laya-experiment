#!/usr/bin/env python3
"""Download dei dataset SOTA per la study (vedi 30-dataset/).

Uso:
    export USE_TF=0 && uv run adv datasets download

Output:
    data/raw/<slug>/<file>   dati grezzi (immutable, gitignored)
    data/manifest.json|.md   sha256 + provenienza (tracciati nel repo)

Filtro: solo dataset citati nel vault o con fonte ufficiale.
Niente aggregator, niente mirror non verificati.
"""

from __future__ import annotations

import io
import json
import sys
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from layaadv import hashing, paths

RAW = paths.RAW
MANIFEST_JSON = paths.MANIFEST_JSON
MANIFEST_MD = paths.MANIFEST_MD

GH_RAW = "https://raw.githubusercontent.com"

# ---------------------------------------------------------------------------
# Dataset HF da load_dataset -> jsonl per split
# ---------------------------------------------------------------------------
HF_DATASETS = [
    # --- primari (vault: typed-decisions-dataset, dataset-secondari) ---
    dict(
        slug="typed-decisions",
        repo="LocalLLaMA/typed-decisions",
        config="all",
        license="apache-2.0",
        note="Benchmark ufficiale; config=all (train 1200, test 400).",
    ),
    # --- secondari classificazione (vault: dataset-secondari) ---
    dict(slug="ag-news", repo="fancyzhx/ag_news", config=None, license="unknown", note="AG News, 4 classi."),
    dict(
        slug="dair-emotion",
        repo="dair-ai/emotion",
        config="split",
        license="cc-by-4.0",
        note="DAIR Emotion, 6 emozioni (config 'split' = 16k/2k/2k del paper).",
    ),
    dict(
        # `mteb/banking77` e non `legacy-datasets/banking77`: il primo porta la colonna
        # `label_text` con il nome dell'intent, il secondo solo l'id intero. Senza i nomi le
        # opzioni diventano "63" e il modello (addestrato sui nomi) non ha niente su cui
        # decidere: la baseline collassa a caso. Stesse 10003/3080 righe.
        slug="banking77",
        repo="mteb/banking77",
        config=None,
        license="mit",
        note="Banking77 via mteb (con label_text): 10003/3080, nomi degli intent inclusi.",
    ),
    dict(slug="sst5", repo="SetFit/sst5", config=None, license="see-card", note="SST-5, distribuzione HF standard a 5 livelli."),
    dict(
        slug="xnli",
        repo="facebook/xnli",
        config=None,
        config_filter=lambda c: c not in ("all_languages",),  # 15 config lingua
        splits=["validation", "test"],  # train è solo EN, non duplicare 15 volte
        license="cc-by-4.0",
        note="XNLI 15 lingue (val+test per lingua), cross-lingual per il router.",
    ),
    # --- guardrail / injection (vault: dataset-secondari) ---
    dict(
        slug="hackaprompt",
        repo="hackaprompt/hackaprompt-dataset",
        config=None,
        license="mit",
        note="HackPrompt ufficiale (gated auto, accesso concesso).",
    ),
    dict(
        slug="notinject",
        repo="leolee99/NotInject",
        config=None,
        license="mit",
        note="NotInject ufficiale (over-defense; autore InjecGuard).",
    ),
    dict(
        slug="wildguardmix",
        repo="allenai/wildguardmix",
        config=None,
        license="odc-by",
        note="WildGuardMix ufficiale AllenAI; split test contiene i benigni (gated auto, accesso concesso).",
    ),
    # --- catalogo attacchi (vault: AdvBench / STRONGReject / SorryBench) ---
    dict(
        slug="strongreject",
        repo="AlignmentResearch/StrongREJECT",
        config=None,
        stream=True,  # train ufficiale vuoto: lo builder non lo prepara in streaming
        license="see-card",
        note="STRONGReject ufficiale (org autori, Souly et al. 2024); split val=313.",
    ),
    # --- italiano (vault: dataset-italiano) ---
    dict(
        slug="alert-it",
        repo="felfri/ALERT_it",
        config=None,
        license="cc-by-nc-sa-4.0",
        note="ALERT red-teaming tradotto EN->IT (+ sottoinsieme ALERTAdv).",
    ),
]

# File raw jsonl/npz (load_dataset CastError sulle feature della card)
HF_RAW_JSONL = [
    dict(
        slug="typed-decisions-v2",
        repo="pngwn/typed-decisions-v2",
        files=[
            "train_raw.jsonl",
            "train_meta.jsonl",
            "test_raw.jsonl",
            "test_meta.jsonl",
            "cal_raw.jsonl",
            "cal_meta.jsonl",
            "stats.json",
            "README.md",
        ],
        license="cc-by-sa-4.0",
        note="Corpus companion, label review/escalate corrette rispetto a v1; raw+meta jsonl (la card ha feature in cast-error).",
    ),
]

# ---------------------------------------------------------------------------
# File grezzi HF (csv/parquet non caricabili come dataset standard)
# ---------------------------------------------------------------------------
HF_FILES = [
    dict(
        slug="prompt-injections-multilingual",
        repo="Federico82/prompt_injections",
        files=["prompt_injections.csv", "LICENSE"],
        license="apache-2.0",
        note="1000+ prompt injection in 7 lingue incl. it.",
    ),
    dict(
        slug="sorry-bench-202406",
        repo="sorry-bench/sorry-bench-202406",
        files=["question.jsonl", "LICENSE", "README.md"],
        include_glob="question_*.jsonl",
        license="other",
        note="SorryBench ufficiale: 400 istruzioni harmful + varianti di obfuscation (gated auto, accesso concesso).",
    ),
]

# ---------------------------------------------------------------------------
# GitHub (fonti ufficiali, ungated)
# ---------------------------------------------------------------------------
GITHUB_FILES = [
    dict(
        slug="advbench",
        repo="llm-attacks/llm-attacks",
        branch="main",
        files=["data/advbench/harmful_behaviors.csv", "data/advbench/harmful_strings.csv"],
        license="see-repo",
        note="AdvBench originale (Zou et al. 2023).",
    ),
    dict(
        slug="compromesso-ItaMSJ",
        repo="fabiopernisi/ita-many-shots-jailbreaking",
        branch="main",
        files=[
            "data/ItaMSJ.csv",
            "data/README.md",
            "data/source_data/FilteredSimpleSafetyTests - test cases.csv",
            "data/source_data/FilteredStrongReject_dataset.csv",
        ],
        license="see-repo",
        note="'Compromesso!' ACL 2024 SRW: 418 QA unsafe italiane many-shot.",
    ),
    dict(
        slug="bipia",
        repo="microsoft/BIPIA",
        branch="main",
        files=[
            "benchmark/README.md",
            "benchmark/email/train.jsonl",
            "benchmark/email/test.jsonl",
            "benchmark/table/train.jsonl",
            "benchmark/table/test.jsonl",
            "benchmark/code/train.jsonl",
            "benchmark/code/test.jsonl",
            "benchmark/code_attack_train.json",
            "benchmark/code_attack_test.json",
            "benchmark/text_attack_train.json",
            "benchmark/text_attack_test.json",
        ],
        license="microsoft license (vedi NOTICE.md)",
        note="BIPIA ufficiale Microsoft; task qa/abstract esclusi (richiedono NewsQA/XSum per licenza).",
    ),
]

# ---------------------------------------------------------------------------
# URL diretti
# ---------------------------------------------------------------------------
URL_FILES = [
    dict(
        slug="massive-1.0",
        url="https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz",
        extract_members=["data/it-IT.jsonl", "data/en-US.jsonl", "LICENSE"],
        license="cc-by-4.0",
        note="MASSIVE 1.0 ufficiale Amazon (S3): solo it-IT + en-US, 60 intent.",
    ),
]

# ---------------------------------------------------------------------------
# Non scaricabili (registrati nel manifest, non inventati)
# ---------------------------------------------------------------------------
BLOCKED = [
    dict(
        slug="pint",
        source="Lakera PINT (github.com/lakeraai/pint-benchmark)",
        license="richiesta accesso via form Lakera",
        note="Gated form esterno: nessuna copia pubblica su HF. Richiedere: share-eu1.hsforms.com/1TwiBEvLXRrCjJSdnbnHpLwfdfs3",
    ),
    dict(
        slug="injecguard",
        source="InjecGuard (Li et al. 2024)",
        license="n/d",
        note="Nessun rilascio pubblico (repo leolee99/InjecGuard_private). NotInject ufficiale è scaricato; PIGuard va dal paper.",
    ),
]

# Richieste inviate, in attesa di approvazione (ri-lanciare lo script quando approvato)
REQUESTED = [
    dict(
        slug="jailbreak-corpus",
        source="Cogensec/jailbreak-corpus (HF)",
        license="odc-by/mit/apache (misto)",
        note="59k attacchi 15 lingue incl. it. Gate manuale: richiesta inviata, in attesa approvazione. Poi rimuovere da qui e da HF_DATASETS.",
        repo="Cogensec/jailbreak-corpus",
        config=None,
    ),
]


def write_manifest(entries: list[dict]) -> None:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "30-dataset/protocollo-dataset-valido",
        "datasets": entries,
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    lines = [
        "# Manifest dataset",
        "",
        f"Generato: {manifest['generated_at']}",
        "",
        "| slug | stato | file | sha256 (primi 16) | licenza | note |",
        "|---|---|---|---|---|---|",
    ]
    for e in entries:
        if e["status"] == "ok":
            files = e["files"]
            fh = "<br>".join(f"`{f['name'][:40]}` {f['sha256'][:16]}" for f in files)
            nfiles = str(len(files))
        elif e["status"] == "needs_gate":
            fh, nfiles = "gate da accettare", "0"
        elif e["status"] == "requested":
            fh, nfiles = "richiesta in approvazione", "0"
        elif e["status"] == "blocked":
            fh, nfiles = "non scaricabile", "0"
        else:
            fh, nfiles = "ERRORE", "0"
        lines.append(
            f"| {e['slug']} | {e['status']} | {nfiles} | {fh} | {e.get('license', '?')} | {e.get('note', '')} |"
        )
    lines.append("")
    MANIFEST_MD.write_text("\n".join(lines))


def download_hf_dataset(spec: dict) -> list[Path]:
    from datasets import Dataset, get_dataset_config_names, load_dataset

    repo, slug = spec["repo"], spec["slug"]
    out = RAW / slug
    out.mkdir(parents=True, exist_ok=True)

    configs = get_dataset_config_names(repo)
    if spec.get("config") is not None:
        chosen = [spec["config"]]
    elif "config_filter" in spec:
        chosen = [c for c in configs if spec["config_filter"](c)]
    elif len(configs) > 1 and "all" in configs:
        chosen = ["all"]
    else:
        chosen = configs

    stream = spec.get("stream", False)
    written: list[Path] = []
    for cfg in chosen:
        ds = load_dataset(repo, cfg, streaming=stream)
        splits = list(ds.keys())
        for split in splits:
            part = ds[split]
            if stream:
                rows = list(part)
                if not rows:
                    continue  # train ufficiale vuoto (StrongREJECT)
                part = Dataset.from_list(rows)
            elif len(part) == 0:
                continue
            if spec.get("splits") and split not in spec["splits"]:
                continue
            path = out / (f"{cfg}__{split}.jsonl" if len(chosen) > 1 else f"{split}.jsonl")
            part.to_json(str(path), force_ascii=False, orient="records")
            written.append(path)
    return written


def download_hf_files(spec: dict) -> list[Path]:
    from huggingface_hub import hf_hub_download, list_repo_files

    repo, slug = spec["repo"], spec["slug"]
    out = RAW / slug
    out.mkdir(parents=True, exist_ok=True)

    paths_list = list(spec["files"])
    if spec.get("include_glob"):
        import fnmatch

        paths_list += [
            f
            for f in list_repo_files(repo, repo_type="dataset")
            if fnmatch.fnmatch(Path(f).name, spec["include_glob"])
        ]

    return [Path(hf_hub_download(repo, rel, repo_type="dataset", local_dir=str(out))) for rel in paths_list]


def download_github_files(spec: dict) -> list[Path]:
    out = RAW / spec["slug"]
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for rel in spec["files"]:
        url = f"{GH_RAW}/{spec['repo']}/{spec['branch']}/{urllib.request.quote(rel)}"
        dest = out / Path(rel).name
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                dest.write_bytes(resp.read())
        except Exception as e:  # noqa: BLE001
            print(f"  !! {spec['slug']}/{rel}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        written.append(dest)
    return written


def download_url_extract(spec: dict) -> list[Path]:
    out = RAW / spec["slug"]
    out.mkdir(parents=True, exist_ok=True)
    written = []
    print(f"  streaming {spec['url']} ...")
    with urllib.request.urlopen(spec["url"], timeout=120) as resp:
        with tarfile.open(fileobj=io.BytesIO(resp.read()), mode="r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                name = member.name
                base = name.split("/", 1)[-1]  # 1.0/data/it-IT.jsonl -> data/it-IT.jsonl
                if base in spec["extract_members"] or name in spec["extract_members"]:
                    dest = out / Path(name).name
                    src = tar.extractfile(member)
                    assert src is not None
                    dest.write_bytes(src.read())
                    written.append(dest)
    return written


def _record(entries: list[dict], slug: str, files: list[Path], spec: dict, status: str | None = None) -> None:
    rec = {
        "slug": slug,
        "status": status or ("ok" if files else "failed"),
        "source": spec.get("repo") or spec.get("url") or spec.get("source") or "",
        "license": spec.get("license", "da verificare"),
        "note": spec.get("note", ""),
        "files": [
            {
                "name": str(f.relative_to(RAW)),
                "bytes": f.stat().st_size,
                "sha256": hashing.sha256_file(f),
            }
            for f in sorted(files)
        ],
    }
    entries.append(rec)
    print(f"[{rec['status']}] {slug}: {len(rec['files'])} file")


def _fetch(entries: list[dict], spec: dict, fn) -> None:
    """Un solo gestore d'errore per tutte le sorgenti (prima duplicato)."""
    slug = spec["slug"]
    print(f"== {slug} ({spec.get('repo') or spec.get('url') or spec.get('source', '')})")
    try:
        files = fn(spec)
    except Exception as e:  # noqa: BLE001
        print(f"  !! {type(e).__name__}: {e}", file=sys.stderr)
        if "gated" in str(e).lower():
            note = (
                f"{spec.get('note', '')} Gate auto: aprire "
                f"https://huggingface.co/datasets/{spec.get('repo', '')} e accettare, poi rilanciare."
            )
            _record(entries, slug, [], {**spec, "note": note}, status="needs_gate")
        else:
            _record(entries, slug, [], spec, status="failed")
        return
    _record(entries, slug, files, spec)


def run(args=None) -> int:
    """Il ponte che il CLI chiama. Il contratto qui è un `int`, non il dict degli esperimenti.

    `cli.main()` gestisce gia' i due casi, e il suo commento nomina esplicitamente questo
    comando; mancava solo questa funzione, quindi `adv datasets download` moriva con
    `AttributeError: module ... has no attribute 'run'`. Il modulo resta eseguibile da solo
    (`python -m layaadv.datasets.download`), che è come si usa durante uno sviluppo.
    """
    return main()


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []

    for spec in HF_DATASETS:
        _fetch(entries, spec, download_hf_dataset)
    for spec in HF_FILES + HF_RAW_JSONL:
        _fetch(entries, spec, download_hf_files)
    for spec in GITHUB_FILES:
        _fetch(entries, spec, download_github_files)
    for spec in URL_FILES:
        _fetch(entries, spec, download_url_extract)

    for b in BLOCKED:
        _record(entries, b["slug"], [], b, status="blocked")
    for b in REQUESTED:
        _record(entries, b["slug"], [], b, status="requested")

    write_manifest(entries)
    by_status: dict[str, list[str]] = {}
    for e in entries:
        by_status.setdefault(e["status"], []).append(e["slug"])
    print(
        "\nOK=%d FAILED=%s NEEDS_GATE=%s REQUESTED=%s BLOCKED=%s"
        % (
            len(by_status.get("ok", [])),
            by_status.get("failed") or "-",
            by_status.get("needs_gate") or "-",
            by_status.get("requested") or "-",
            by_status.get("blocked") or "-",
        )
    )
    print(f"manifest: {MANIFEST_JSON.relative_to(paths.ROOT)}")
    return 1 if by_status.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
