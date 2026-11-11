"""Readers for the immutable raw datasets under ``data/raw/``.

Raw files are never modified (download script owns them; sha256 lives in the
manifest). Everything downstream reads through these helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from layaadv import paths


def dataset_path(slug: str, name: str) -> Path:
    """``data/raw/<slug>/<name>`` with an actionable error when missing."""
    path = paths.RAW / slug / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `uv run adv datasets download` first"
        )
    return path


def read_jsonl(path: Path) -> Iterator[dict]:
    """Stream records from a jsonl file."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def maybe_json(value: Any) -> Any:
    """Dataset columns ship JSON-encoded strings; decode them once, here.

    typed-decisions stores ``questions``/``gold``/``label_agreement`` as
    strings inside the jsonl (HF export quirk).
    """
    if isinstance(value, str):
        return json.loads(value)
    return value
