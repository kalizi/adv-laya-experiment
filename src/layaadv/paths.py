"""Single source of truth for every filesystem path.

No other module is allowed to compute paths with ``Path(__file__)``:
everything resolves from :data:`ROOT` so moving the project never breaks
experiments (see 50-metodologia/configurazione-ambiente.md for the layout).
"""

from __future__ import annotations

from pathlib import Path

# src/layaadv/paths.py -> parents[0]=layaadv, [1]=src, [2]=repo root
ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"           # downloaded datasets, immutable + sha256 in manifest
ADV = DATA / "adv"           # generated adversarial states (jsonl)
RESULTS = ROOT / "results"   # metrics, predictions, logs

MANIFEST_JSON = DATA / "manifest.json"
MANIFEST_MD = DATA / "manifest.md"

# CLM-v0.1-8B: the quantised encoder plus the pooling/scale contract that no GGUF
# key carries. Outside the repo because the Q6_K weights are 6.5 GiB and the
# heads live in ~/.cache/clm (clm-serve downloads them itself).
MODELS = Path.home() / ".cache" / "layaadv"
CLM_DIR = MODELS / "clm"
CLM_CONFIG = CLM_DIR / "clm_config.json"
CLM_HEADS = Path.home() / ".cache" / "clm" / "CLM_v0.1-8B.pt"
