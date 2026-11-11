"""Metti gli accenti al testo italiano, ma solo dove non serve contesto.

Le correzioni "è -> è", "piu' -> più", "perche' -> perché" sono unambigue: in italiano `e`
accentato non esiste altra interpretazione, e in Python `è` non è un token valido, quindi
quando compare è prosa dentro un commento o una docstring.

Le correzioni che **richiedono contesto** non si toccano: `e`/`è`, `si`/`sì`, `da`/`dà`,
`la`/`là`, `po`/`può`, `se`/`sé`, `ne`/`né`. Sono decisioni di significato, non di ortografia,
e sbagliarle cambia la frase. Restano elencate in `--report` e vanno corrette a mano.

Dove non si applica: il codice. In `.py` si riscrivono solo i token `COMMENT` e `STRING` (che
in questo progetto sono prosa e messaggi d'errore), mai gli identificatori. Nei `.md` si
saltano i blocchi delimitati da tre apici inversi, che contengono codice.

Uso:
    uv run python tools/fix_accents.py            # applica
    uv run python tools/fix_accents.py --report   # solo il conto, nessuna scrittura
"""

from __future__ import annotations

import argparse
import io
import pathlib
import re
import tokenize

# Non serve contesto: accento obbligatorio, e la forma senza accento non è un'altra parola.
UNAMBIGUOUS = {
    # Con apostrofia apostrofata (e') e senza: "e' un" -> "e' un", "e' un" senza -> "e' un".
    # Nota: le chiavi sono SEMPRE le forme senza accento, i valori le forme accentate. Il
    # dizionario e' quindi idempotente e soprattutto auto-consistente: se questo file venisse
    # accentato da se stesso le chiavi diventerebbero identiche ai valori e il tool smetterebbe
    # di correggere, senza rompere nulla. Per questo il tool salta anche se stesso.
    "e'": "\u00e8",
    "piu": "pi\u00f9",
    "piu'": "pi\u00f9",
    "perche": "perch\u00e9",
    "perche'": "perch\u00e9",
    "puo": "pu\u00f2",
    "puo'": "pu\u00f2",
    "gia": "gi\u00e0",
    "gia'": "gi\u00e0",
    "cosi": "cos\u00ec",
    "cosi'": "cos\u00ec",
    "cioe": "cio\u00e8",
    "cioe'": "cio\u00e8",
    "pero'": "per\u00f2",
    "percio'": "perci\u00f2",
    "percio": "perci\u00f2",
    "cio'": "ci\u00f2",
    "cio": "ci\u00f2",
    "ne'": "n\u00e9",
    "c'e'": "c'\u00e8",
    "sara": "sar\u00e0",
    "sara'": "sar\u00e0",
    "verra": "verr\u00e0",
    "potra": "potr\u00e0",
    "dovra": "dovr\u00e0",
    "fara": "far\u00e0",
    "avra": "avr\u00e0",
    "stara": "star\u00e0",
    "meta'": "met\u00e0",
    "realta'": "realt\u00e0",
    "verita'": "verit\u00e0",
    "quantita'": "quantit\u00e0",
    "qualita'": "qualit\u00e0",
    "identita'": "identit\u00e0",
    "affidabilita'": "affidabilit\u00e0",
    "capacita'": "capacit\u00e0",
    "periodicita'": "periodicit\u00e0",
    "disponibilita'": "disponibilit\u00e0",
    "saro'": "sar\u00f2",
    "avro'": "avr\u00f2",
    "potro'": "potr\u00f2",
    "faro'": "far\u00f2",
    "dovro'": "dovr\u00f2",
    "usciro'": "uscir\u00f2",
    "misurero'": "misurer\u00f2",
    "rimarro'": "rimarr\u00f2",
    "entrero'": "entrer\u00f2",
    "partiro'": "partir\u00f2",
    "tornero'": "torner\u00f2",
    "vedero'": "veder\u00f2",
    "stabiliro'": "stabilir\u00f2",
    "restauro'": "restaur\u00f2",
}

# Servono contesto: NON si toccano, si contano e si correggono a mano.
AMBIGUOUS_PATTERNS = {
    r"\b(il|lo|la|un|questo|quello)\s+e\s+(che|un|una|il|lo)\b": "e vs è",
    r"\be\s+stato\b|\be\s+una\s+(misura|scelta|correzione)": "e vs è",
    r"\bsi\s+(?:è|è)\b": "si/sì",
    r"\bpo\b\s+(?:il|lo|la|un|essere|avere)": "po/può",
    r"\bda\s+(?:solo|solo|zero|se|stesso)\b": "da/dà",
    r"\bse\s+(?:stesso|sé|quindi)\b": "se/sé",
    r"\bla\s+(?:sua|sue|così|come)\b": "la/là",
    r"\bne\b\s+(?:era|sono|hanno)": "ne/né",
}

WORD = re.compile("[A-Za-zÀ-ÿ]+'")


def fix_text(text: str) -> tuple[str, int]:
    n = 0

    def sub(m: re.Match) -> str:
        nonlocal n
        w = m.group(0)
        key = w.lower()
        repl = UNAMBIGUOUS.get(key)
        if repl and repl.lower() != key:
            n += 1
            return repl if w[0].islower() else repl.capitalize()
        return w

    return WORD.sub(sub, text), n


def fix_markdown(text: str) -> tuple[str, int]:
    """Salta i blocchi di codice: dentro ci sono identificatori, non prosa."""
    out, total, in_fence = [], 0, False
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        new, n = fix_text(line)
        total += n
        out.append(new)
    return "".join(out), total


def fix_python(text: str) -> tuple[str, int]:
    """Riscrive solo COMMENT e STRING. Gli identificatori non li vede."""
    lines = text.splitlines(keepends=True)
    spans: list[tuple[int, int]] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                for ln in range(tok.start[0], tok.end[0] + 1):
                    spans.append((ln, len(lines[ln - 1])))
    except tokenize.TokenError:
        return text, 0
    total = 0
    for ln, end in spans:
        line = lines[ln - 1]
        new, n = fix_text(line[:end])
        if n:
            lines[ln - 1] = new + line[end:]
            total += n
    return "".join(lines), total


def ambiguous_report(text: str) -> list[str]:
    found = []
    for pattern, label in AMBIGUOUS_PATTERNS.items():
        for m in re.finditer(pattern, text, re.I):
            start = max(0, m.start() - 40)
            found.append(f"    [{label}] ...{' '.join(text[start:m.end() + 30].split())}...")
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", action="store_true", help="non scrive nulla")
    ap.add_argument("paths", nargs="*", default=["**/*.md", "src/**/*.py", "tools/*.py"])
    args = ap.parse_args()

    seen, changed, ambig = set(), [], []
    for pattern in args.paths:
        for f in sorted(pathlib.Path(".").glob(pattern)):
            if ".venv" in str(f) or str(f) in seen:
                continue
            if f.resolve() == pathlib.Path(__file__).resolve():
                continue  # le chiavi del dizionario sono forme senza accento: accentarle qui
                #       trasformerebbe il tool in una mappa identitaria
            seen.add(str(f))
            try:
                raw = f.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            new, n = fix_python(raw) if f.suffix == ".py" else fix_markdown(raw)
            if f.suffix == ".py":
                ambig += [f"{f}: {a}" for a in ambiguous_report(new)]
            elif f.suffix in (".md", ".tex"):
                ambig += [f"{f}: {a}" for a in ambiguous_report(new)]
            if n:
                changed.append((str(f), n))
                if not args.report:
                    f.write_text(new)

    for f, n in changed:
        print(f"  {n:4}  {f}")
    print(f"\n{sum(n for _, n in changed)} correzioni in {len(changed)} file"
          + ("  (dry run)" if args.report else ""))
    if ambig:
        print(f"\nDa correggere a mano ({len(ambig)}):")
        for a in ambig[:40]:
            print(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
