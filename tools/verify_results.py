"""I numeri pubblicati sono ricalcolabili dagli store grezzi?

Verifica di riproducibilita': per ogni riga di un file di risultati, ricalcola l'ASR dai
`preds.jsonl` con `metrics.core.asr` e confronta con il valore pubblicato, e controlla che il
denominatore coincida.

Il perche' esiste: i `results/*.json` si possono scrivere a mano, o sopravvivere a una
correzione del codice che li rende sbagliati senza che nessuno se ne accorga. Il `run_id` è un
hash della config, quindi i record non si mescolano da soli, ma nessuno garantisce che la
config sia quella che pensi. Qui il controllo è sui dati, non sulla fiducia.

Uso:
    uv run python tools/verify_results.py
    uv run python tools/verify_results.py results/E15_kcurve__deberta-onnx.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from layaadv.metrics.core import accuracy, asr, iter_decisions

ROOT = pathlib.Path("results")
TOLERANCE = 5e-5  # i valori pubblicati sono arrotondati a 4 decimali
EXPLORATORY_WITHOUT_RUNS = {
    "E12d_homoglyph_dose.json": 11,
    "E12f_homoglyph_tiers.json": 17,
    "E12i_underscore_dose.json": 13,
}


def store_for(experiment: str, run_id: str) -> pathlib.Path:
    return ROOT / experiment / "runs" / run_id


def records(experiment: str, run_id: str) -> list[dict]:
    path = store_for(experiment, run_id) / "preds.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check_identity(row: dict, experiment: str) -> list[str]:
    """Il config del run deve dire la stessa cosa che la riga pubblica.

    È il controllo che chiude la classe di bug `device` fuori dal `run_id`: se la riga dice
    CUDA e il run è stato fatto su CPU, i due non possono essere la stessa misura.
    """
    problems = []
    for key in ("clean_run_id", "adv_run_id"):
        cfg_path = store_for(experiment, row[key]) / "config.json"
        if not cfg_path.exists():
            problems.append(f"{row[key]}: config.json assente")
            continue
        cfg = json.loads(cfg_path.read_text())
        for field in ("backend", "variant", "quantization", "device", "k", "seed"):
            # Solo i campi che la riga dichiara davvero. Un esperimento che non registra
            # `seed` non ha un disallineamento: è una verifica che qui non è possibile, e
            # segnalarla come errore renderebbe il segnale illeggibile.
            if field not in row or field not in cfg:
                continue
            if cfg[field] != row[field]:
                problems.append(f"{row[key]}: {field} run={cfg[field]!r} riga={row[field]!r}")
    return problems


def verify(path: pathlib.Path) -> tuple[int, int, int, float, list[str]]:
    payload = json.loads(path.read_text())
    experiment = payload.get("experiment", "E15_quantization")
    rows = payload.get("results") or []
    if path.name in EXPLORATORY_WITHOUT_RUNS and len(rows) != EXPLORATORY_WITHOUT_RUNS[path.name]:
        return 0, 0, 1, 0.0, [f"{path.name}: numero di sonde esplorative cambiato"], 0
    worst = 0.0
    checked = 0
    skipped = 0
    paired_shape = 0
    problems: list[str] = []
    baseline = next((row for row in rows if "run_id" in row and "asr" not in row), None)
    for row in rows:
        if "run_id" in row:
            checked += 1
            try:
                current = records(experiment, row["run_id"])
                decs = list(iter_decisions(current))
                acc = accuracy(current)
                if len(decs) != row["n_decisions"] or abs(acc - row["accuracy"]) > TOLERANCE:
                    problems.append(f"{path.name} {row['run_id']}: accuracy/conteggio divergenti")
                if "asr" in row:
                    if baseline is None:
                        problems.append(f"{path.name}: baseline senza run_id")
                    else:
                        rate, denom = asr(records(experiment, baseline["run_id"]), current)
                        if denom != row["asr_denominator"] or abs(rate - row["asr"]) > TOLERANCE:
                            problems.append(f"{path.name} {row['run_id']}: ASR/denominatore divergenti")
            except (FileNotFoundError, KeyError, TypeError) as e:
                problems.append(f"{path.name} {row['run_id']}: record non verificabile ({e})")
            continue
        # Le sonde E12 senza record grezzi sono ammesse soltanto con nome esplicito
        # nell'allowlist esplorativa; ogni nuova riga senza provenienza fa fallire il gate.
        if "clean_run_id" not in row or "adv_run_id" not in row:
            skipped += 1
            if "run_id" in row:
                paired_shape += 1
            if path.name not in EXPLORATORY_WITHOUT_RUNS:
                problems.append(f"{path.name}: riga senza provenienza e senza classificazione esplorativa")
            continue
        # Una riga che dichiara la coppia dei run ma **non** dichiara `asr` non e' una riga da
        # ricalcolare: E19 per esempio ha la coppia `clean_run_id`/`adv_run_id` ma il suo
        # risultato e' il tasso di rilevazione del detector, non un ASR. Ricalcolare l'ASR su
        # quelle righe appaia due sottoinsiemi diversi (puliti contro attaccati) e crasha.
        if "asr" not in row:
            if experiment == "E19_detector_labels":
                checked += 1
                try:
                    clean = records(experiment, row["clean_run_id"])
                    adv = records(experiment, row["adv_run_id"])
                    positive = payload["variants"][row["variant"]]["malevolo"]
                    def fire(rs: list[dict]) -> int:
                        return sum(r["labels"].get("guardrail") == positive for r in rs)
                    fp = fire(clean) / len(clean)
                    det = fire(adv) / len(adv)
                    ratio = fp / det if fp and det else None
                    if (len(clean) != row["n_clean"] or len(adv) != row["n_attacked"]
                            or abs(fp - row["false_block_rate"]) > TOLERANCE
                            or abs(det - row["detection_rate"]) > TOLERANCE
                            or ratio != row["ratio_cost_to_benefit"]):
                        problems.append(f"{path.name} {row['variant']}: detector divergente")
                except (FileNotFoundError, KeyError, ZeroDivisionError) as e:
                    problems.append(f"{path.name} {row.get('variant')}: record non verificabile ({e})")
            else:
                skipped += 1
                problems.append(f"{path.name}: riga con run appaiati non verificata")
            continue
        checked += 1
        problems += [f"{row.get('config')} K={row.get('k')}: {p}" for p in check_identity(row, experiment)]
        try:
            rate, denom = asr(records(experiment, row["clean_run_id"]),
                              records(experiment, row["adv_run_id"]))
        except FileNotFoundError as e:
            problems.append(f"{row.get('config')} K={row.get('k')}: store assente ({e})")
            continue
        if denom != row["asr_denominator"]:
            problems.append(
                f"{row.get('config')} K={row.get('k')} seed={row.get('seed')}: denominatore "
                f"pubblicato {row['asr_denominator']} ricalcolato {denom}"
            )
        if rate is None:
            problems.append(f"{row.get('config')} K={row.get('k')}: ASR ricalcolata vuota")
            continue
        dev = abs(rate - row["asr"])
        worst = max(worst, dev)
        if dev > TOLERANCE:
            problems.append(
                f"{row.get('config')} K={row.get('k')} seed={row.get('seed')}: ASR pubblicata "
                f"{row['asr']} ricalcolata {rate:.6f}"
            )
    return checked, skipped, len(problems), worst, problems, paired_shape


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", help="file di risultati (default: tutti i per-famiglia)")
    args = ap.parse_args()
    # Solo i file che hanno davvero righe: l'indice di E15 elenca gli altri senza ripederli.
    files = [pathlib.Path(f) for f in args.files] or sorted(
        p for p in ROOT.glob("*.json")
        if p.name != "E15_quantization.json" and json.loads(p.read_text()).get("results")
    )
    if not files:
        print("nessun file di risultati trovato")
        return 1
    total_rows = total_skipped = total_bad = total_paired = 0
    for path in files:
        checked, skipped, bad, worst, problems, paired = verify(path)
        total_rows += checked
        total_skipped += skipped
        total_paired += paired
        total_bad += bad
        bits = []
        if paired:
            bits.append(f"{paired} righe con run_id unico non verificate")
        if skipped - paired:
            bits.append(f"{skipped - paired} sonde esplorative E12 senza record grezzi")
        extra = f"  ({'; '.join(bits)}, non verificate qui)" if bits else ""
        print(f"{path.name:44} {checked:3} righe  {bad} problemi  dev.max {worst:.2e}{extra}")
        for p in problems[:8]:
            print(f"    {p}")
    print(
        f"\n{total_rows} righe verificate dagli store grezzi, {total_bad} problemi"
        + (f", {total_paired} righe con run_id unico non verificate e "
           f"{total_skipped - total_paired} sonde esplorative E12 senza record grezzi"
           if total_skipped else "")
    )
    return 1 if total_bad else 0


if __name__ == "__main__":
    sys.exit(main())
