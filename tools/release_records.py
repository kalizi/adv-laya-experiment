"""Crea e verifica gli archivi dei preds citati dai risultati verificabili.

Uso: python tools/release_records.py build|verify
Gli archivi contengono soltanto `preds.jsonl`, senza testi dei dataset grezzi.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import tarfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RELEASE = RESULTS / "release"
MANIFEST = RELEASE / "records.manifest.json"
MAX_RAW_BYTES = 80 * 1024 * 1024

# Esperimenti i cui record **non** entrano negli archivi di release, con il motivo.
# E22 e' un'estensione fuori dal percorso critico e dal paper (vedi
# 40-esperimenti/E22-attenzione-laya.md): i suoi `preds.jsonl` portano un profilo
# d'attenzione per livello per caso, cioe' ~40 MB che nessun numero del paper cita.
# I numeri della nota restano ricalcolabili dagli store in `results/E22_attention/runs/`,
# che stanno nel repository: l'esclusione e' dal candidato distribuibile, non dalla
# riproducibilita'.
# La lista e' corta e motivata di proposito: un'esclusione senza motivo sarebbe un record
# che nessuno recupera, che e' la classe di difetto che questo progetto rifiuta.
# E22b (stessa estensione di E22, geometria della distanza) ha lo stesso trattamento: i
# suoi record sono i piu' pesanti di tutti, un profilo d'attenzione per livello per caso.
RELEASE_SCOPE_EXCLUDED = {
    "E22_attention": "estensione fuori dal paper: i record restano nello store del repository",
    "E22b_attention_distance": "estensione fuori dal paper: i record restano nello store del repository",
    # E23 (meccanismo dell'iniezione) e' pensato per il paper, ma al 2026-10-09 non ha
    # ancora il run a N pieno: i record su disco sono degli smoke test. Metterlo nel
    # candidato vorrebbe dire archiviare misure che nessun numero pubblica. L'esclusione
    # va **tolta** quando il run pieno entra nel paper, non tenuta per comodita'.
    "E23_defense": (
        "estensione fuori dal paper: i record sono 14 bracci di probs per caso, ~25 MB che "
        "nessun numero del paper cita. Da rimuovere dall'esclusione se i numeri entrano nel "
        "paper"
    ),
    "E23_form": (
        "estensione fuori dal paper, stesso trattamento di E22: i record portano un profilo "
        "di patch per forma per caso, ~25 MB che nessun numero del paper cita. Da rimuovere "
        "dall'esclusione se i numeri entrano nel paper"
    ),
    "E23_heads": (
        "estensione fuori dal paper, stesso trattamento di E22: i record sono un censimento "
        "di 448 knockout per caso (~40 MB) che nessun numero del paper cita. Da rimuovere "
        "dall'esclusione se e quando i numeri entrano nel paper"
    ),
    "E23_channel": (
        "estensione fuori dal paper: i record sono 9 bracci di ASR per caso (stessa forma di "
        "E22), ~9 MB che nessun numero del paper cita. Da rimuovere dall'esclusione se i "
        "numeri entrano nel paper"
    ),
    "E23_mechanism": (
        "run a N pieno non ancora fatto (2026-10-09): i record su disco sono smoke test, "
        "e archiviare misure che nessun numero pubblica e' il difetto che questo test "
        "cerca. Da rimuovere dall'esclusione quando i numeri entrano nel paper"
    ),
}


def is_in_release_scope(experiment: str) -> bool:
    return experiment not in RELEASE_SCOPE_EXCLUDED


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cited_records(*, require_present: bool = True) -> list[pathlib.Path]:
    paths: set[pathlib.Path] = set()

    def collect(node: object, experiment: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "run_ids" and isinstance(value, dict):
                    for run_id in value.values():
                        add(experiment, run_id)
                elif (key == "run_id" or key.endswith("_run_id")) and isinstance(value, str):
                    add(experiment, value)
                else:
                    collect(value, experiment)
        elif isinstance(node, list):
            for value in node:
                collect(value, experiment)

    def add(experiment: str, run_id: str) -> None:
        if not run_id:
            return
        path = RESULTS / experiment / "runs" / run_id / "preds.jsonl"
        if require_present and not path.is_file():
            raise FileNotFoundError(path)
        paths.add(path)

    for summary in sorted(RESULTS.glob("*.json")):
        payload = json.loads(summary.read_text())
        experiment = payload.get("experiment", "E15_quantization")
        if not is_in_release_scope(experiment):
            continue
        collect(payload, experiment)
    return sorted(paths)


def chunks(paths: list[pathlib.Path]) -> list[list[pathlib.Path]]:
    groups: list[list[pathlib.Path]] = []
    current: list[pathlib.Path] = []
    size = 0
    for path in paths:
        n = path.stat().st_size
        if current and size + n > MAX_RAW_BYTES:
            groups.append(current)
            current, size = [], 0
        current.append(path)
        size += n
    if current:
        groups.append(current)
    return groups


def build() -> None:
    RELEASE.mkdir(parents=True, exist_ok=True)
    paths = cited_records()
    if not paths:
        raise ValueError("nessun record citato")
    archives = []
    for index, group in enumerate(chunks(paths), 1):
        name = f"records-{index:02d}.tar.zst"
        dest = RELEASE / name
        temp = RELEASE / f"{name}.tmp"
        proc = subprocess.Popen(["zstd", "-q", "-T0", "-6", "-o", str(temp)], stdin=subprocess.PIPE)
        assert proc.stdin is not None
        try:
            with tarfile.open(fileobj=proc.stdin, mode="w|") as archive:
                for path in group:
                    relative = path.relative_to(ROOT)
                    info = archive.gettarinfo(str(path), arcname=str(relative))
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)
        finally:
            proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"zstd fallito su {name}")
        os.replace(temp, dest)
        archives.append({
            "name": name,
            "bytes": dest.stat().st_size,
            "sha256": sha256(dest),
            "files": [
                {"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha256(p)}
                for p in group
            ],
        })
        print(f"{name}: {len(group)} record, {dest.stat().st_size} byte")
    payload = {"format": 1, "selection": "run_id citati da results/*.json", "archives": archives}
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{len(paths)} record in {len(archives)} archivi; manifest: {MANIFEST}")


def verify() -> None:
    payload = json.loads(MANIFEST.read_text())
    expected = {entry["path"] for archive in payload["archives"] for entry in archive["files"]}
    cited = {str(path.relative_to(ROOT)) for path in cited_records(require_present=False)}
    if expected != cited:
        raise ValueError(f"manifest e risultati divergono: {len(expected ^ cited)} percorsi")
    for archive in payload["archives"]:
        path = RELEASE / archive["name"]
        if path.stat().st_size != archive["bytes"] or sha256(path) != archive["sha256"]:
            raise ValueError(f"hash archivio errato: {path}")
        entries = {entry["path"]: entry for entry in archive["files"]}
        seen = set()
        proc = subprocess.Popen(["zstd", "-q", "-dc", str(path)], stdout=subprocess.PIPE)
        assert proc.stdout is not None
        try:
            with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
                for member in tar:
                    if not member.isfile() or member.name not in entries or member.name in seen:
                        raise ValueError(f"membro non atteso: {member.name}")
                    stream = tar.extractfile(member)
                    assert stream is not None
                    digest = hashlib.sha256()
                    count = 0
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        count += len(chunk)
                        digest.update(chunk)
                    entry = entries[member.name]
                    if count != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
                        raise ValueError(f"hash record errato: {member.name}")
                    seen.add(member.name)
        finally:
            proc.stdout.close()
        if proc.wait() != 0 or seen != set(entries):
            raise ValueError(f"archivio incompleto: {path}")
    print(f"verificati {len(expected)} record in {len(payload['archives'])} archivi")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "verify"])
    action = parser.parse_args().action
    if action == "build":
        build()
    else:
        verify()


if __name__ == "__main__":
    main()
