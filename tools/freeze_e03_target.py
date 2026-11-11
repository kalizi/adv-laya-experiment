"""Verify and freeze the full E03 target curve from paired raw prediction stores.

Usage: uv run python tools/freeze_e03_target.py results/E03_option_injection__...json
The command refuses incomplete or mismatched scopes and writes a provenance
manifest next to the chosen scope. It never edits a measured result.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

from layaadv import hashing
from layaadv.data import loaders
from layaadv.metrics.core import aggregate, asr

EXPERIMENT = "E03_option_injection"
CHECKPOINT = "convaiinnovations/laya-typed-decisions"
DATASET = "massive-it"
KS = (4, 8, 16, 24, 32, 40)
TEMPLATES = {"prepend": "head", "append": "tail"}
N_CASES = 2974


def json_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def records(path: Path, expected_ids: set[str]) -> list[dict]:
    require(path.is_file(), f"predizioni assenti: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(row["id"]) for row in rows]
    require(len(rows) == N_CASES and len(set(ids)) == N_CASES and set(ids) == expected_ids,
            f"numero, unicità o subset errati: {path}")
    return rows


def close(actual: float, published: float, label: str) -> None:
    require(abs(actual - published) < 1e-10,
            f"{label}: ricalcolato {actual}, pubblicato {published}")


@lru_cache(maxsize=1)
def weight_revisions(weight_hash: str) -> tuple[str, list[str]]:
    cache = (Path.home() / ".cache/huggingface/hub"
             / "models--convaiinnovations--laya-typed-decisions")
    resolved_main = (cache / "refs/main").read_text(encoding="utf-8").strip()
    model_path = cache / "snapshots" / resolved_main / "model.safetensors"
    require(model_path.is_file() and hashing.sha256_file(model_path) == weight_hash,
            "la revisione main in cache non corrisponde all'hash dei pesi del run")
    candidates = sorted(snapshot.parent.name for snapshot in (cache / "snapshots").glob("*/model.safetensors")
                        if snapshot.resolve() == model_path.resolve())
    return resolved_main, candidates


def freeze(scope_path: Path) -> dict:
    payload = json_file(scope_path)
    rows = payload["results"]
    require(payload.get("checkpoint") == CHECKPOINT and payload.get("dataset") == DATASET,
            "scope non target")
    require(payload.get("seed") == 0 and payload.get("pad_tokens") == 0,
            "seed o pad non target")
    require(len(rows) == len(KS) * len(TEMPLATES), "curva E03 incompleta")
    require({(row["k"], row["template"]) for row in rows}
            == {(k, template) for k in KS for template in TEMPLATES},
            "griglia K/template incompleta o duplicata")
    identity = payload["scope_identity"]
    require(scope_path.stem.endswith(hashing.sha256_canonical_json(identity)[:16]),
            "hash dello scope errato")
    require(identity["backend"] == "laya" and identity["checkpoint"] == CHECKPOINT,
            "identità del backend errata")
    require(identity["limit"] is None and identity["seed"] == 0
            and identity["k_values"] == list(KS)
            and set(identity["templates"]) == set(TEMPLATES)
            and identity["run_identity"]["backend"] == "laya",
            "parametri dello scope diversi dal protocollo finale")
    require({(r["k"], r["template"], r["clean_run_id"], r["adv_run_id"]) for r in rows}
            == {(r["k"], r["template"], r["clean_run_id"], r["adv_run_id"])
                for r in identity["runs"]},
            "run ID diversi dall'identità dello scope")

    store_root = scope_path.parent / EXPERIMENT / "runs"
    by_k = {}
    for k in KS:
        cases, ds, _ = loaders.load_cases(DATASET, k=k, seed=0)
        require(len(cases) == N_CASES, f"N inatteso per K={k}")
        by_k[k] = (set(c.id for c in cases), ds.file_sha256(), ds.subset_hash(cases),
                   {c.id: c.gold for c in cases})

    frozen_rows = []
    for row in sorted(rows, key=lambda r: (r["k"], r["template"])):
        k, template = row["k"], row["template"]
        ids, dataset_hash, subset_hash, gold_by_id = by_k[k]
        require(row["position"] == TEMPLATES[template] and row["pad_tokens"] == 0,
                f"posizione o pad errati K={k} {template}")
        runs = []
        paired = []
        for kind, run_id in (("clean", row["clean_run_id"]),
                             ("adversarial", row["adv_run_id"])):
            directory = store_root / run_id
            config_path = directory / "config.json"
            pred_path = directory / "preds.jsonl"
            summary_path = directory / "summary.json"
            config = json_file(config_path)
            summary = json_file(summary_path)
            require(config["kind"] == kind and config["checkpoint"] == CHECKPOINT
                    and config["dataset"] == DATASET and config["k"] == k
                    and config["seed"] == 0 and config["n_cases"] == N_CASES
                    and config["dataset_sha256"] == dataset_hash
                    and config["subset_hash"] == subset_hash,
                    f"config incompatibile: {run_id}")
            require(config.get("backend") == "laya" and config.get("device") == "cuda"
                    and config.get("weight_sha256") == identity["weight_sha256"],
                    f"backend, device o pesi non certificati: {run_id}")
            require(config.get("batch_size") == 16 and config.get("chunk") == 32
                    and config.get("quantization") == "none"
                    and config.get("compiled") is False and config.get("fast") is False,
                    f"runtime non comparabile: {run_id}")
            if kind == "adversarial":
                require(config.get("template") == template
                        and config.get("position") == TEMPLATES[template]
                        and config.get("pad_tokens") == 0,
                        f"attacco non corrispondente: {run_id}")
            require(summary.get("run_id") == run_id and summary.get("config") == config,
                    f"summary/config diversi: {run_id}")
            actual_devices = summary["summary"]["metrics"]["latency"]["devices"]
            require(actual_devices == ["cuda"], f"device effettivo non CUDA: {run_id}")
            latency_path = directory / "latency.jsonl"
            latency = [json.loads(line) for line in latency_path.read_text(encoding="utf-8").splitlines()
                       if line.strip()]
            require(latency and {sample.get("device") for sample in latency} == {"cuda"}
                    and {sample.get("dtype") for sample in latency} == {"torch.bfloat16"},
                    f"precisione o device nei batch non coerenti: {run_id}")
            record_set = records(pred_path, ids)
            require(all(rec["gold"] == gold_by_id[str(rec["id"])] for rec in record_set),
                    f"gold non appaiato: {run_id}")
            paired.append(record_set)
            runs.append({
                "kind": kind, "run_id": run_id,
                "config": config,
                "config_sha256": hashing.sha256_file(config_path),
                "preds_sha256": hashing.sha256_file(pred_path),
                "latency_sha256": hashing.sha256_file(latency_path),
                "summary_sha256": hashing.sha256_file(summary_path),
                "runtime": summary["env"],
                "actual_devices": actual_devices,
                "precision": "torch.bfloat16",
                "n_batches": len(latency),
            })
        rate, denominator = asr(*paired)
        clean_metrics, adv_metrics = aggregate(paired[0]), aggregate(paired[1])
        require(rate is not None and denominator == row["asr_denominator"],
                f"denominatore ASR errato K={k} {template}")
        close(rate, row["asr"], f"ASR K={k} {template}")
        close(clean_metrics["accuracy"], row["accuracy_clean"], f"accuracy clean K={k}")
        close(adv_metrics["accuracy"], row["accuracy_adv"], f"accuracy adv K={k}")
        close(clean_metrics["ece"], row["ece_clean"], f"ECE clean K={k}")
        close(adv_metrics["ece"], row["ece_adv"], f"ECE adv K={k}")
        close(clean_metrics["mean_conf_wrong"], row["mean_conf_wrong_clean"],
              f"confidenza errori clean K={k}")
        close(adv_metrics["mean_conf_wrong"], row["mean_conf_wrong_adv"],
              f"confidenza errori adv K={k}")
        for tau in ("0.75", "0.85"):
            close(clean_metrics["oer"][tau], row["oer_clean"][tau], f"OER clean K={k} tau={tau}")
            close(adv_metrics["oer"][tau], row["oer_adv"][tau],
                  f"OER adv K={k} tau={tau}")
        unchanged = sum(1 for clean, adv in zip(*paired) if clean["labels"] == adv["labels"])
        require(unchanged == row["unchanged_labels"], f"label immutate errate K={k} {template}")
        require(round(100 * (clean_metrics["accuracy"] - adv_metrics["accuracy"]), 2)
                == row["drop_pp"], f"drop errato K={k} {template}")
        frozen_rows.append({
            "k": k, "template": template, "position": TEMPLATES[template],
            "n_cases": N_CASES, "asr": rate, "asr_denominator": denominator,
            "accuracy_clean": clean_metrics["accuracy"],
            "accuracy_adv": adv_metrics["accuracy"],
            "dataset": DATASET, "split": "test", "dataset_sha256": dataset_hash,
            "subset_hash": subset_hash, "seed": 0,
            "backend": "laya", "batch_size": 16, "chunk": 32,
            "precision": "torch.bfloat16", "device": "cuda",
            "weight_revision_main_at_freeze": weight_revisions(identity["weight_sha256"])[0],
            "weight_revision_candidates_same_blob": weight_revisions(identity["weight_sha256"])[1],
            "weight_revision_note": (
                "Il run salva SHA256 dei pesi, non il commit HF; refs/main e le revisioni "
                "compatibili sono documentate al congelamento."
            ),
            "weight_sha256": identity["weight_sha256"],
            "runs": runs,
        })
    return {
        "experiment": EXPERIMENT,
        "status": "verified_from_paired_raw_stores",
        "scope_file": scope_path.name,
        "scope_sha256": hashing.sha256_file(scope_path),
        "command": (
            "uv run adv e03 --checkpoint convaiinnovations/laya-typed-decisions "
            "--device cuda --batch-size 16"
        ),
        "rows": frozen_rows,
    }


def historical_incomplete(store_root: Path) -> dict:
    """Record missing identity without guessing their backend from a filename."""
    runs = []
    required = {"backend", "device", "weight_sha256"}
    for config_path in sorted(store_root.glob("*/config.json")):
        config = json_file(config_path)
        missing = sorted(required - config.keys())
        if not missing:
            continue
        pred_path = config_path.parent / "preds.jsonl"
        runs.append({
            "run_id": config_path.parent.name,
            "kind": config.get("kind"),
            "requested_checkpoint": config.get("checkpoint"),
            "dataset": config.get("dataset"),
            "k": config.get("k"),
            "template": config.get("template"),
            "pad_tokens": config.get("pad_tokens"),
            "n_cases_declared": config.get("n_cases"),
            "missing_identity_fields": missing,
            "config_sha256": hashing.sha256_file(config_path),
            "preds_sha256": hashing.sha256_file(pred_path) if pred_path.exists() else None,
        })
    return {
        "experiment": EXPERIMENT,
        "status": "historical_identity_incomplete",
        "note": "Il checkpoint richiesto non dimostra quale backend abbia prodotto i record.",
        "runs": runs,
    }


def write_new_or_same(path: Path, payload: dict) -> None:
    serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != serialized:
        raise FileExistsError(f"contenuto diverso già presente: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", type=Path)
    args = parser.parse_args()
    manifest = freeze(args.scope)
    destination = args.scope.parent / "provenance" / (args.scope.stem + ".manifest.json")
    write_new_or_same(destination, manifest)
    historical_path = destination.parent / "E03_historical_incomplete_identity.json"
    historical = historical_incomplete(args.scope.parent / EXPERIMENT / "runs")
    write_new_or_same(historical_path, historical)
    print(f"{destination}: {len(manifest['rows'])} righe verificate")
    print(f"{historical_path}: {len(historical['runs'])} run con identità incompleta")


if __name__ == "__main__":
    main()
