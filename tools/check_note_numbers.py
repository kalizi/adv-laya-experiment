"""Verifica i numeri di ogni nota contro **il suo** store, non contro tutti.

`check_claim_numbers.py` risponde a "questo numero esiste da qualche parte nel progetto".
Questo risponde alla domanda che serve per una revisione: "questo numero esiste **nell'archivio
di cui questa nota parla**". Uno `0.1858` in una nota E12 che combacia con un valore di E03 e'
un falso verde, e' il modo piu' economico che ha un documento di mentire.

Ogni nota dichiara i suoi store in frontmatter o nel corpo; le associo per nome file, e le
note senza archivio proprio vengono segnalate come non verificabili, che e' un esito
diverso da "numero sbagliato" e va detto.

Uso: `uv run python tools/check_note_numbers.py [nome-nota ...]`
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CLAIM = re.compile(r"(?<![\w.])(\d+\.\d{3,4})(?![\w])")

# Famiglia di esperimento -> store che le appartengono. La chiave e' il token che compare
# nel nome della nota e nel nome del file di risultato.
FAMILY = {
    "E00": ["E00_setup"],
    "E01": ["E01_baseline", "e01_third_checkpoint"],
    "E02": ["E02_option_count"],
    # Solo i file **per scope con nome nuovo** (`...__pad<N>__<template>`). Fino al
    # 2026-10-01 esistevano anche file col nome privo di `pad` e template, prodotti dal
    # codice precedente, con lo stesso prefisso e **numeri diversi** (il template `append`
    # non metteva niente in coda). Cercarli insieme rende il check incapace di distinguere
    # una citazione aggiornata da una superata: e' il falso verde peggiore, perche' il
    # numero stale sembra confermato. I legacy si escludono dal confronto e si contano.
    "E03": ["E03_option_injection__"],
    "E04": ["E04_truncation"],
    "E12": ["E12_surface", "E12b_rare_synonym", "e12_probe_filler_position"],
    "E13": ["E13_label_semantics"],
    "E14": ["E14_self_guardrail"],
    "E15": ["E15_quantization", "E15_kcurve"],
    "E16": ["E16_calibration_defense"],
    # E17 cita la temperatura grezza che dichiara E16: senza quella fonte il numero non si
    # ricalcola e la nota sembra avere un valore senza archivio.
    "E17": ["E17_configuration", "E16_calibration_defense"],
    # E18 non ha `results[]` (il detector non ha bisogno del modello): il suo archivio e' il
    # payload stesso, quindi si legge da li'.
    "E18": ["E18_nomination_detector", "E16_calibration_defense", "E19_detector_labels"],
    # E20 cita i falsi positivi che dichiara E18: senza quell'archivio il costo
    # dell'allentamento non si ricalcola e la tabella sembra piena di numeri inventati.
    "E20": ["E20_paraphrase_evasion", "E18_nomination_detector"],
    # E22 cita i numeri di E03 per dire che li riproduce alla quarta cifra: senza
    # quell'archivio lo sarebbero valori suoi e non un confronto.
    # E22 cita anche la geometria della distanza (E22b), che sta in un file suo.
    "E22": ["E22_attention", "E22b_attention_distance", "E03_option_injection"],
    # E23 (disegno, poi misura) cita le baseline congelate di E22/E03 su cui appoggia le
    # sue predizioni, e i numeri della propria misura. Registrarli qui significa che la nota
    # non puo' citare un numero che quegli archivi non contengono.
    "E23": ["E23_mechanism", "E23_heads", "E23_channel", "E23_form", "E23_defense",
            "E22_attention",
            "E22b_attention_distance", "E03_option_injection"],
}

# Note che parlano di piu' famiglie, o che non hanno un archivio proprio.
EXTRA = {
    "E03xE04-composizione": ["E03xE04", "E03_option_injection", "E04_truncation"],
    "E03xE04-laya-multilingual": ["E03xE04", "E03_option_injection"],
    "E12-sintesi-superficie": ["E12_surface", "E12b_rare_synonym"],
    "E12c-simbolo": ["E12_surface"],
    "E12d-omoglifi": ["E12_surface"],
    "E12e-omoglifi-uts39": ["E12_surface"],
    "E12f-omoglifi-livelli": ["E12_surface"],
    "E12i-underscore": ["E12_surface"],
    "E12l-zerowidth": ["E12_surface"],
    "E12m-swap-tasti": ["E12_surface"],
    "E12n-multicarattere-bidi": ["E12_surface"],
}


def is_current_e03(stem: str) -> bool:
    """Include both current E03 scope names, with or without the identity hash."""
    if not stem.startswith("E03_option_injection__"):
        return True
    return any(part.startswith("pad") and part[3:].isdigit()
               for part in stem.split("__")[1:])


def numbers_in(stems: list[str]) -> set[float]:
    vals: set[float] = set()
    for f in sorted((ROOT / "results").glob("*.json")):
        if not any(f.stem.startswith(s) for s in stems):
            continue
        if not is_current_e03(f.stem):
            continue
        data = json.loads(f.read_text())

        def walk(node: object) -> None:
            if isinstance(node, dict):
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
            elif isinstance(node, bool) or node is None:
                return
            elif isinstance(node, (int, float)):
                for nd in (3, 4):
                    vals.add(round(float(node), nd))
                vals.add(round(float(node), 6))

        walk(data)
    return vals


def stems_for(note: str) -> list[str]:
    if note in EXTRA:
        return EXTRA[note]
    for fam, stems in FAMILY.items():
        if note.startswith(fam):
            return stems
    return []


def main(argv: list[str]) -> int:
    notes = sorted((ROOT / "40-esperimenti").glob("*.md"))
    if argv:
        notes = [n for n in notes if any(a in n.name for a in argv)]

    total_bad = 0
    for note in notes:
        key = note.stem
        stems = stems_for(key)
        if not stems:
            print(f"\n== {key}: nessun archivio associabile (non verificabile)")
            continue
        known = numbers_in(stems)
        bad: dict[str, list[int]] = {}
        for lineno, line in enumerate(note.read_text().splitlines(), 1):
            for raw in CLAIM.findall(line):
                v = float(raw)
                if round(v, 4) in known or round(v, 3) in known or round(v, 6) in known:
                    continue
                bad.setdefault(raw, []).append(lineno)
        if bad:
            total_bad += len(bad)
            print(f"\n== {key}  ({', '.join(stems)}): {len(bad)} numeri fuori archivio")
            for raw in sorted(bad):
                print(f"   {raw:>9}  righe {','.join(map(str, bad[raw]))}")
        else:
            print(f"\n== {key}: tutti i numeri combaciano ({len(known)} valori nello store)")

    print(f"\n{total_bad} numeri fuori archivio, su {len(notes)} note")
    return 1 if total_bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
