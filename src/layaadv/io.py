"""Result writing + resumable run checkpointing.

Every experiment ends with :func:`write_result`; long computations additionally
wrap their loop in a :class:`RunStore` so that Ctrl-C loses at most the batch
currently in flight (see the class docstring for the resume contract).
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from layaadv import hashing, paths


class ConfigMismatch(RuntimeError):
    """Resume requested on a run whose config differs — never mix runs."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def env_fingerprint(weight_sha256: str | None = None) -> dict:
    """Everything needed to reproduce a number: versions, hardware, weights.

    Written into every result file (protocollo: logga sha256 pesi + versioni).
    """
    import laya
    import torch

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "laya": getattr(laya, "__version__", "?"),
        "torch": torch.__version__,
        "transformers": _version_or_none("transformers"),
        "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu",
        "cuda_available": torch.cuda.is_available(),
        "weight_sha256": weight_sha256,
    }


def _version_or_none(pkg: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(pkg)
    except Exception:
        return None


def write_result(name: str, payload: dict) -> Path:
    """Write results/<name>.json (the file the vault notes expect)."""
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    out = paths.RESULTS / f"{name}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")
    return out


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    """Full rewrite of a jsonl file (used at finish, not during the loop)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: Path, *, repair_tail: bool = False) -> list[dict]:
    """Read JSONL; optionally remove a truncated tail before resuming writes."""
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open("r+b" if repair_tail else "rb") as f:
        while line := f.readline():
            start = f.tell() - len(line)
            if not line.strip():
                if repair_tail and not line.endswith(b"\n"):
                    f.write(b"\n")
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                if f.read(1):
                    raise  # an invalid line in the middle is corruption, not a cut tail
                if repair_tail:
                    f.truncate(start)
                break
            if repair_tail and not line.endswith(b"\n"):
                f.write(b"\n")
    return records


class RunStore:
    """Checkpoint of one run: ``results/<experiment>/runs/<run_id>/``.

    Layout::

        config.json   immutable: checkpoint + dataset hash + params
        preds.jsonl   one record per completed item, appended per batch
        latency.jsonl one forward-pass wall time per batch, appended with preds
        state.json    progress counters (observability only; preds.jsonl is truth)
        summary.json  written by finish()

    Resume contract:
    - run_id = sha256 of the canonical config: same params => same run.
    - re-running with a *different* config raises :class:`ConfigMismatch`
      instead of silently mixing two experiments.
    - items already present in preds.jsonl are never recomputed.
    - latency samples live in their own sidecar so a resumed run reports the
      percentiles of the whole run, not just of the final segment.
    """

    def __init__(self, experiment: str, run_id: str, directory: Path, config: dict, resume: bool):
        self.experiment = experiment
        self.run_id = run_id
        self.dir = directory
        self.config = config
        self._resumed = resume
        self._completed: set[str] | None = None

        self._preds_path = directory / "preds.jsonl"
        self._latency_path = directory / "latency.jsonl"
        self._state_path = directory / "state.json"

    # ------------------------------------------------------------------ create
    @classmethod
    def create(
        cls,
        experiment: str,
        config: dict,
        *,
        resume: bool = True,
        force_new: bool = False,
    ) -> "RunStore":
        paths.RESULTS.mkdir(parents=True, exist_ok=True)
        base = paths.RESULTS / experiment / "runs"
        run_id = hashing.sha256_canonical_json(config)[:16]

        if force_new:
            run_id = cls._free_run_id(base, run_id)

        directory = base / run_id
        if directory.exists():
            if not resume:
                raise ConfigMismatch(
                    f"run {run_id} exists and --no-resume was passed (use --force-new)"
                )
            stored = json.loads((directory / "config.json").read_text())
            if stored != config:
                raise ConfigMismatch(
                    f"run {run_id} was created with a different config; "
                    "use --force-new to start a separate run"
                )
        else:
            directory.mkdir(parents=True)
            (directory / "config.json").write_text(
                json.dumps(config, indent=2, ensure_ascii=False, default=str) + "\n"
            )
        return cls(experiment, run_id, directory, config, resume)

    @staticmethod
    def _free_run_id(base: Path, run_id: str) -> str:
        if not (base / run_id).exists():
            return run_id
        n = 2
        while (base / f"{run_id}-{n}").exists():
            n += 1
        return f"{run_id}-{n}"

    # ------------------------------------------------------------------- reads
    def completed_ids(self) -> set[str]:
        """Ids already checkpointed — the resume filter."""
        if self._completed is None:
            self._completed = {str(r.get("id")) for r in read_jsonl(self._preds_path, repair_tail=True)}
        return self._completed

    def read_records(self) -> list[dict]:
        return read_jsonl(self._preds_path, repair_tail=True)

    def read_latency_samples(self) -> list[dict]:
        """Every forward-pass sample ever recorded for this run, with its context.

        Replayed from disk so a resumed run reports percentiles over every batch it
        ever ran, not only the ones after the interruption.
        """
        return [r for r in read_jsonl(self._latency_path, repair_tail=True) if "ms" in r]

    def read_latencies(self) -> list[float]:
        return [float(r["ms"]) for r in self.read_latency_samples()]

    def append_latency(self, milliseconds: float, decisions: int = 0, **context) -> None:
        """Record one forward pass. Appended unconditionally, never deduped:
        a repeated batch after resume is a genuine second measurement.

        ``decisions`` and ``context`` are what make a speed number comparable.
        The decisions are counted *here* so ms/decision is exact even after a
        resume (dividing the samples by the run total would understate throughput
        by whatever fraction was reused). The context carries where the model
        actually was: laya demotes itself to CPU on OOM without raising, and that
        costs about 10x, so a sample without its device is not comparable with one
        that has it — and the run's own `env` block cannot tell them apart, because
        it reports the GPU whenever CUDA is available at all.
        """
        with self._latency_path.open("a", encoding="utf-8") as f:
            row = {"ms": round(float(milliseconds), 3), "decisions": decisions, **context}
            f.write(json.dumps(row) + "\n")
            f.flush()

    # ------------------------------------------------------------------ writes
    def append(self, records: list[dict], key: str = "id") -> int:
        """Append finished records (one call per batch), refresh progress state.

        Returns the number of records actually written (already-done items are
        skipped, so append is idempotent on resume).
        """
        if not records:
            return 0
        done = self.completed_ids()
        fresh = []
        seen = set(done)
        for record in records:
            if key not in record:
                raise KeyError(f"record senza chiave {key!r}")
            record_id = str(record[key])
            if record_id in seen:
                if record_id not in done:
                    raise ValueError(f"id duplicato nello stesso batch: {record_id}")
                continue
            fresh.append(record)
            seen.add(record_id)
        if not fresh:
            return 0
        self._preds_path.parent.mkdir(parents=True, exist_ok=True)
        with self._preds_path.open("a", encoding="utf-8") as f:
            for rec in fresh:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            f.flush()
        done.update(str(r[key]) for r in fresh)
        self._write_state(len(done), str(fresh[-1][key]))
        return len(fresh)

    def _write_state(self, completed: int, last_key: str) -> None:
        self._state_path.write_text(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "experiment": self.experiment,
                    "completed": completed,
                    "last_key": last_key,
                    "updated_at": _now(),
                },
                indent=2,
            )
        )

    def finish(self, summary: dict) -> Path:
        """Close the run: per-run summary inside the run directory."""
        out = self.dir / "summary.json"
        out.write_text(
            json.dumps(
                {
                    "experiment": self.experiment,
                    "run_id": self.run_id,
                    "config": self.config,
                    "env": env_fingerprint(self.config.get("weight_sha256")),
                    "finished_at": _now(),
                    "summary": summary,
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )
        return out
