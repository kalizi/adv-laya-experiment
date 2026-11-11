"""I record della difesa E16 devono entrare negli archivi di release."""

import json
from pathlib import Path

from tools.release_records import (
    MANIFEST,
    RELEASE_SCOPE_EXCLUDED,
    ROOT,
    cited_records,
    is_in_release_scope,
)


def test_release_manifest_includes_nested_e16_and_e01_runs():
    manifest = json.loads(MANIFEST.read_text())
    archived = {entry["path"] for group in manifest["archives"] for entry in group["files"]}
    cited = {str(path.relative_to(ROOT)) for path in cited_records(require_present=False)}
    assert archived == cited
    assert any(path.startswith("results/E16_calibration_defense/runs/") for path in archived)
    assert any(path.startswith("results/E01_baseline/runs/") for path in archived)


def test_ogni_record_citato_e_archiviato_o_escluso_con_motivo():
    """Nessun record citato puo' sparire di mezzo: o è negli archivi, o il suo esperimento
    è escluso per un motivo scritto. Un record che nessuna delle due cose copre è un numero
    pubblicato che nessuno ricalcola, ed è il difetto per cui questo test esiste."""
    manifest = json.loads(MANIFEST.read_text())
    archived = {entry["path"] for group in manifest["archives"] for entry in group["files"]}
    results = ROOT / "results"
    for summary in sorted(results.glob("*.json")):
        payload = json.loads(summary.read_text())
        experiment = payload.get("experiment", "E15_quantization")
        if not is_in_release_scope(experiment):
            # L'esclusione deve essere motivata, non un silenzio.
            assert experiment in RELEASE_SCOPE_EXCLUDED
            assert RELEASE_SCOPE_EXCLUDED[experiment].strip()
            continue
        for path in cited_records_for(summary):
            assert str(path.relative_to(ROOT)) in archived, path


def cited_records_for(summary: Path) -> list[Path]:
    """I record citati da un singolo riepilogo, usando il criterio del tool."""
    from tools.release_records import RESULTS

    payload = json.loads(summary.read_text())
    experiment = payload.get("experiment", "E15_quantization")
    paths: set[Path] = set()

    def add(run_id: str) -> None:
        if run_id:
            paths.add(RESULTS / experiment / "runs" / run_id / "preds.jsonl")

    def collect(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "run_ids" and isinstance(value, dict):
                    for run_id in value.values():
                        add(run_id)
                elif (key == "run_id" or key.endswith("_run_id")) and isinstance(value, str):
                    add(value)
                else:
                    collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)

    collect(payload)
    return sorted(paths)
