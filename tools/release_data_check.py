"""Verifica checksum e licenze dichiarate per i tre dataset dei claim centrali."""

from __future__ import annotations

import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXPECTED_LICENSES = {"typed-decisions": "apache-2.0", "banking77": "mit", "massive-1.0": "cc-by-4.0"}


def main() -> None:
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
    datasets = {entry["slug"]: entry for entry in manifest["datasets"]}
    count = 0
    for slug, license_name in EXPECTED_LICENSES.items():
        entry = datasets[slug]
        if entry["license"] != license_name or entry["status"] != "ok":
            raise ValueError(f"licenza o stato inatteso: {slug}")
        for file in entry["files"]:
            path = ROOT / "data" / "raw" / file["name"]
            if not path.is_file() or path.stat().st_size != file["bytes"]:
                raise ValueError(f"file mancante o dimensione diversa: {path}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != file["sha256"]:
                raise ValueError(f"checksum diverso: {path}")
            count += 1
    license_text = (ROOT / "data" / "raw" / "massive-1.0" / "LICENSE").read_text()
    if "Attribution 4.0 International" not in license_text:
        raise ValueError("licenza MASSIVE locale inattesa")
    print(f"{count} file core verificati, tre dataset con licenza dichiarata")


if __name__ == "__main__":
    main()
