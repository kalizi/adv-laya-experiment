"""Classifica le eccezioni numeriche della nota E03 prima del gate release."""

from __future__ import annotations

import collections
import json

from check_note_numbers import CLAIM, ROOT, numbers_in

NOTE = ROOT / "40-esperimenti" / "E03-option-injection.md"
REPORT = ROOT / "results" / "release" / "e03_note_exceptions.json"


def main() -> None:
    known = numbers_in(["E03_option_injection__"])
    unmatched = collections.Counter()
    for line in NOTE.read_text().splitlines():
        for raw in CLAIM.findall(line):
            value = float(raw)
            if not any(round(value, n) in known for n in (3, 4, 6)):
                unmatched[raw] += 1
    report = json.loads(REPORT.read_text())
    allowed = {entry["value"]: entry for entry in report["exceptions"]}
    if set(unmatched) != set(allowed):
        raise ValueError(f"eccezioni aggiunte/rimosse: {sorted(set(unmatched) ^ set(allowed))}")
    for value, count in unmatched.items():
        entry = allowed[value]
        if count != entry["occurrences"] or not entry["reason"] or not entry["source"]:
            raise ValueError(f"eccezione cambiata: {value}")
    print(f"nota E03: {len(unmatched)} valori eccezionali classificati, nessuna nuova eccezione")


if __name__ == "__main__":
    main()
