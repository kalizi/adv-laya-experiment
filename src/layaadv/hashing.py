"""Hashing helpers shared by the dataset downloader and the experiment logs.

sha256 of files goes into the dataset manifest (provenance) and into every
experiment result (reproducibility: same weights + same data => same numbers).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_CHUNK = 1 << 20  # 1 MiB


def sha256_file(path: Path) -> str:
    """Stream a file through sha256 (constant memory, works on multi-GB files)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_canonical_json(obj) -> str:
    """Deterministic hash of a JSON-ish object.

    Key order and whitespace must not change the hash: this signature keys both
    question grouping (engine batching) and run identity (checkpoint resume).
    """
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
