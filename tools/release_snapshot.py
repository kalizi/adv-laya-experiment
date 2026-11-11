"""Hash del candidato di release senza creare un commit Git.

Uso: python tools/release_snapshot.py build|verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "results" / "release" / "snapshot.json"
DIRECTORIES = (
    "10-modello/", "20-attack-surface/", "30-dataset/",
    "30-modelli/", "40-esperimenti/", "50-metodologia/", "90-risorse/",
    "src/", "tests/", "tools/", "results/", "data/uts39/",
)
ROOT_FILES = {
    ".gitignore", "CLAUDE.md", "README.md", "LICENSE",
    "todo-list.md", "pyproject.toml", "uv.lock", "data/manifest.json", "data/manifest.md",
    "NOTICE", "vault-indice.md",
}


def files() -> list[pathlib.Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
    )
    names = [pathlib.Path(name.decode()) for name in output.split(b"\0") if name]
    return sorted(
        ROOT / name for name in names
        if (str(name) in ROOT_FILES or str(name).startswith(DIRECTORIES))
        and name != SNAPSHOT.relative_to(ROOT)
    )


def digest(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build() -> None:
    rows = [
        {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest(path)}
        for path in files()
    ]
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(json.dumps({"format": 1, "files": rows}, indent=2) + "\n")
    print(f"snapshot: {len(rows)} file, {SNAPSHOT}")


def verify() -> None:
    rows = json.loads(SNAPSHOT.read_text())["files"]
    expected = {row["path"]: row for row in rows}
    actual = {str(path.relative_to(ROOT)): path for path in files()}
    if set(expected) != set(actual):
        raise ValueError(f"file aggiunti o mancanti: {sorted(set(expected) ^ set(actual))[:8]}")
    for name, path in actual.items():
        row = expected[name]
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError(f"file cambiato: {name}")
    print(f"snapshot verificato: {len(rows)} file")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "verify"])
    if parser.parse_args().action == "build":
        build()
    else:
        verify()


if __name__ == "__main__":
    main()
