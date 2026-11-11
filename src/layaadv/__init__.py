"""adversarial-laya: tooling per attacchi avversari contro Laya (System One).

Convenzioni del pacchetto (vedi CLAUDE.md / 50-metodologia):
  - ``layaadv.model.engine`` è l'unico modulo che importa ``laya``/torch;
  - le metriche sono funzioni pure (``layaadv.metrics.core``);
  - ogni esperimento passa da ``layaadv.evaluate`` + ``layaadv.io.RunStore``.
"""

from __future__ import annotations

import os

# transformers/TF deadlock workaround: deve essere imposto PRIMA di qualsiasi
# import di transformers/torch (50-metodologia/configurazione-ambiente.md).
os.environ.setdefault("USE_TF", "0")

from layaadv.evaluate import evaluate  # noqa: E402
from layaadv.io import RunStore, write_result  # noqa: E402
from layaadv.model.engine import LayaEngine  # noqa: E402
from layaadv.schema import Case, Prediction  # noqa: E402

__all__ = ["Case", "LayaEngine", "Prediction", "RunStore", "evaluate", "write_result"]
