"""Ricostruisce l'indice `results/E03_option_injection.json` dai file per scope.

Perche' esiste
--------------
Ogni combinazione di scope scrive il proprio file, cosi' girare un altro scope non
sovrascrive la tabella del precedente. L'indice dice quale file leggere.

Una funzione sola, `write_index`, ed è quella che anche `e03.run` chiama a fine
run: l'indice non è costruito a mano li', altrimenti ogni run lo riscriverebbe
dai propri scope e gli altri sparirebbero dall'indice (è successo: l'indice
riportava 1 scope mentre i file erano 4).

Uso:
    uv run python -m layaadv.experiments.e03_index
"""

from __future__ import annotations

import json
import sys

from layaadv import hashing, io, paths

STEM = "E03_option_injection"
PREFIX = f"{STEM}__"

NOTE = (
    "Indice dei file per scope. Ogni file ha le sue righe in 'results'; questo non le "
    "duplica, serve a sapere quale file leggere. Rigenerato dai file su disco, quindi "
    "un run nuovo non cancella gli scope precedenti."
)


def scope_problem(path, payload: dict) -> str | None:
    """Reject a misleading scope before it enters paper tables and figures."""
    rows = payload.get("results")
    if not isinstance(rows, list) or not rows:
        return "nessuna riga"
    checkpoint = payload.get("checkpoint")
    dataset = payload.get("dataset")
    identity = payload.get("scope_identity")
    if identity:
        digest = hashing.sha256_canonical_json(identity)[:16]
        if not path.stem.endswith(f"__{digest}"):
            return "hash dello scope diverso dal contenuto"
        if identity.get("checkpoint") != checkpoint or identity.get("dataset") != dataset:
            return "intestazione diversa dall'identità dello scope"
        actual = {(r.get("k"), r.get("template"), r.get("position"),
                   r.get("clean_run_id"), r.get("adv_run_id"))
                  for r in rows}
        # Gli scope congelati prima del 2x2 codificavano la posizione nella riga,
        # ma non nell'elenco dei run dell'identità. Mantenerli indicizzabili.
        expected = {(r.get("k"), r.get("template"),
                     r.get("position", {"prepend": "head", "append": "tail"}.get(r.get("template"))),
                     r.get("clean_run_id"), r.get("adv_run_id"))
                    for r in identity.get("runs", [])}
        if actual != expected or len(rows) != len(identity.get("runs", [])):
            return "run ID delle righe diversi dall'identità dello scope"
    for row in rows:
        if row.get("checkpoint") != checkpoint or row.get("dataset") != dataset:
            return "riga con checkpoint o dataset diverso dall'intestazione"
        backend = row.get("backend")
        if identity and backend != identity.get("backend"):
            return "backend della riga diverso dall'identità dello scope"
        # A historical smoke test requested a Laya checkpoint through the CLM
        # backend and overwrote a Laya-named result. Do not index it as Laya.
        if checkpoint.startswith("convaiinnovations/laya") and backend not in (None, "laya", "laya-onnx"):
            return f"backend {backend} incompatibile con il checkpoint Laya"
    return None


def collect(excluded: list[dict] | None = None) -> dict:
    """Scan results/ for the per-scope files and index them."""
    scopes: dict[str, dict] = {}
    for path in sorted(paths.RESULTS.glob(f"{PREFIX}*.json")):
        payload = json.loads(path.read_text())
        problem = scope_problem(path, payload)
        if problem:
            print(f"[E03-index] escluso {path.name}: {problem}", file=sys.stderr)
            if excluded is not None:
                excluded.append({"file": path.name, "reason": problem})
            continue
        rows = payload.get("results", [])
        scopes[path.stem[len(PREFIX) :]] = {
            "checkpoint": rows[0]["checkpoint"],
            "dataset": rows[0]["dataset"],
            "backend": rows[0].get("backend"),
            "n_rows": len(rows),
            "k_values": payload.get("k_values"),
            "pad_tokens": payload.get("pad_tokens"),
            "templates": sorted(payload.get("templates", {})),
            "head_budget_walls": [w["k"] for w in payload.get("head_budget_walls", [])],
        }
    return scopes


def write_index() -> dict:
    """Rebuild the index from every scope file present and return it."""
    excluded: list[dict] = []
    scopes = collect(excluded)
    if not scopes:
        raise SystemExit(f"nessun file {PREFIX}*.json in {paths.RESULTS}")
    summary = {
        "experiment": STEM,
        "note": NOTE,
        "scopes": scopes,
        "files": [f"{PREFIX}{s}.json" for s in sorted(scopes)],
        "n_rows": sum(s["n_rows"] for s in scopes.values()),
        "excluded": excluded,
    }
    io.write_result(STEM, summary)
    return summary


def main() -> None:
    summary = write_index()
    print(f"[E03-index] {len(summary['scopes'])} scope indicizzati, {summary['n_rows']} righe:")
    for name in sorted(summary["scopes"]):
        s = summary["scopes"][name]
        wall = f" (muro a K={s['head_budget_walls']})" if s["head_budget_walls"] else ""
        pad = f", pad {s['pad_tokens']}" if s["pad_tokens"] else ""
        print(f"  {PREFIX}{name}.json  {s['n_rows']} righe{pad}{wall}")


if __name__ == "__main__":
    main()
