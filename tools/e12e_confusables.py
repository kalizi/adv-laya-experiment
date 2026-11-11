"""Confusabili normative: UTS #39 al posto della tabella scritta a mano.

`confusables.txt` mappa sorgente -> prototipo:
    03B1 ; 0061 ; MA   # ( α -> a )
L'attacco ha bisogno dell'inversa: **quali caratteri un umano confonde con `a`**.

Perche' questo e non una tabella scritta a mano: è lo standard che usano browser
e IDN, quindi la pretesa "indistinguibile per un umano" diventa verificabile
contro un dato normativo invece che essere una mia assunzione.

Il dato è solo testo (codici e commenti), nessun font e nessuna immagine: la
forma del glifo la decide il font, non il dato Unicode, quindi rasterizzare
misurerebbe il font installato invece che la proprieta' dei caratteri.
"""

import unicodedata
from pathlib import Path

# La specifica sta in tools/data/confusables.txt: 748K di dato normativo, non una
# libreria. Il path è configurabile perche' a volte si rigenera in /tmp.
CONFUSABLES_CANDIDATES = (
    Path(__file__).parent / "data" / "confusables.txt",
    Path("/tmp/confusables.txt"),
)


def _resolve(path: Path | str | None) -> Path:
    if path is not None:
        return Path(path)
    for cand in CONFUSABLES_CANDIDATES:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "confusables.txt non trovato: scaricalo da "
        "https://www.unicode.org/Public/security/latest/confusables.txt "
        "in tools/data/ oppure in /tmp/"
    )


CONFUSABLES = _resolve(None)

# Famiglie di intereste. "LETTER" = glifo di lettera, che è l'unica cosa che
# possiamo sostituire senza rompere il JSON (chiavi e valori sono stringhe).
LATIN = "LATIN"
GREEK = "GREEK"
CYRILLIC = "CYRILLIC"
MATHEMATICAL = "MATHEMATICAL"
FULLWIDTH = "FULLWIDTH"
LETTER_FAMILIES = {LATIN, GREEK, CYRILLIC, MATHEMATICAL, FULLWIDTH}


def load_inverted(path: Path | str = CONFUSABLES) -> dict[str, set[str]]:
    """{prototipo latino minuscolo: {caratteri che vi si confondono}}.

    Le righe possono avere piu' sorgenti e piu' prototipi (es. `0301 0301`).
    Per l'attacco serve il caso semplice: sorgente singola, prototipo singola.
    Le combinazioni multi-carattere sono skel di grafemi, non sostituzioni che
    possiamo fare lettera per lettera, quindi si scartano esplicitamente invece
    di tirare fuori un errore silenzioso.
    """
    out: dict[str, set[str]] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = [f.strip() for f in line.split(";") if f.strip()]
        if len(fields) < 2:
            continue
        srcs, dsts = fields[0].split(), fields[1].split()
        if len(srcs) != 1 or len(dsts) != 1:
            continue  # skel di grafemi: non sostituibile lettera per lettera
        src, dst = int(srcs[0], 16), int(dsts[0], 16)
        # il prototipo deve essere una lettera latina minuscola
        if not ("a" <= chr(dst) <= "z"):
            continue
        ch = chr(src)
        if unicodedata.combining(ch):
            continue
        out.setdefault(chr(dst), set()).add(ch)
    return out


def family_of(ch: str) -> str:
    """Famiglia Unicode del carattere, dal nome ufficiale."""
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return "UNKNOWN"
    for fam in LETTER_FAMILIES:
        # es. "GREEK SMALL LETTER ALPHA", "MATHEMATICAL BOLD SMALL A"
        if name.startswith(fam) or f" {fam} " in name:
            return fam
    return name.split()[0]


def is_letter(ch: str) -> bool:
    return unicodedata.category(ch).startswith("L")


def build_tables(path: Path | str = CONFUSABLES) -> dict[str, dict[str, list[str]]]:
    """Tabelle normative per famiglia: cross-script e same-script.

    cross-script (greco, cirillico): il carattere appartiene a un altro
    alfabeto. È la versione forte della tesi: il testo "appare" latino ma non
    è latino, né per byte né per lo script.

    same-script (matematico, fullwidth): stessa origine alfabetica, quindi la
    differenza è di presentazione. Sono attacchi piu' deboli da questo punto di
    vista e vanno tenuti separati: E12d li aveva mescolati.
    """
    inv = load_inverted(path)
    tables: dict[str, dict[str, list[str]]] = {
        "cross": {k: [] for k in inv},
        "same": {k: [] for k in inv},
    }
    for proto, sources in inv.items():
        for ch in sorted(sources):
            if ch == proto or not is_letter(ch):
                continue
            fam = family_of(ch)
            if fam in (GREEK, CYRILLIC):
                tables["cross"][proto].append(ch)
            elif fam in (MATHEMATICAL, FULLWIDTH):
                tables["same"][proto].append(ch)
    return tables


def load(path: Path | str = CONFUSABLES) -> dict[str, dict[str, list[str]]]:
    tables = build_tables(path)
    # tiene solo le lettere con almeno un omoglifo utilizzabile
    tables["cross"] = {k: v for k, v in tables["cross"].items() if v}
    tables["same"] = {k: v for k, v in tables["same"].items() if v}
    return tables


if __name__ == "__main__":
    tables = load()
    cross, same = tables["cross"], tables["same"]

    print(f"lettere latine con omoglifi cross-script (greco/cirillico): {len(cross)}")
    print(f"lettere latine con varianti same-script (math/fullwidth):    {len(same)}")
    print()
    print("cross-script, lettere con piu' omoglifi:")
    for k in sorted(cross, key=lambda c: -len(cross[c]))[:14]:
        print(f"  {k} -> {len(cross[k]):>2}  {''.join(cross[k])}")
    print()

    # confronto con la tabella scritta a mano che E12d usava
    hand = {
        "a": "α", "b": "β", "c": "ϲ", "d": "ԁ", "e": "ε", "i": "ι",
        "k": "κ", "l": "λ", "m": "μ", "n": "η", "o": "ο", "p": "ρ",
        "r": "г", "s": "ѕ", "t": "τ", "u": "υ", "v": "ν", "x": "χ",
        "y": "γ", "z": "ζ",
    }
    agree = dis = invented = 0
    print("verifica della tabella a mano di E12d contro UTS #39:")
    for k, v in hand.items():
        official = cross.get(k, [])
        if v in official:
            agree += 1
        elif v in same.get(k, []):
            dis += 1
        else:
            invented += 1
            print(f"  {k}: {v!r} NON e' in UTS #39 (inventata)")
    print(f"  confermate {agree}, di famiglia diversa {dis}, non presenti {invented}")
    print()
    missing = [k for k in hand if k not in cross and k not in same]
    print(f"lettere della tabella a mano con zero omoglifi ufficiali: {missing or 'nessuna'}")
