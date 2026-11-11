"""Scarica solo i tre dataset core, senza toccare jailbreak-corpus o il manifest."""

from __future__ import annotations

from release_data_check import ROOT
from release_data_check import main as check

from layaadv.datasets.download import (
    HF_DATASETS,
    URL_FILES,
    download_hf_dataset,
    download_url_extract,
)

CORE = {"typed-decisions", "banking77", "massive-1.0"}


def main() -> None:
    for spec in HF_DATASETS:
        if spec["slug"] in CORE and not (ROOT / "data" / "raw" / spec["slug"]).exists():
            print(f"scarico {spec['slug']}")
            download_hf_dataset(spec)
    for spec in URL_FILES:
        if spec["slug"] in CORE and not (ROOT / "data" / "raw" / spec["slug"]).exists():
            print(f"scarico {spec['slug']}")
            download_url_extract(spec)
    check()


if __name__ == "__main__":
    main()
