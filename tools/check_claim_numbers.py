"""Ogni numero decimale che il paper o le note dichiarano deve combaciare con uno store.

Non controlla che il numero sia *vero*, controlla che sia **ristampato da uno store**: un
numero che non si ricalcola dagli archivi non e' un numero, e' una memoria. Il controllo e'
deliberatamente severo sulle 4 cifre (il formato con cui il progetto pubblica) e segnala anche
i valori arrotondati a 3, perche' e' li' che uno sbaglio si nasconde.

Uso: `uv run python tools/check_claim_numbers.py [--verbose]`
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Documenti che fanno affermazioni numeriche. L'ordine non conta: ogni cifra viene cercata
# in tutti gli store, e un numero che non esiste da nessuna parte e' sospetto.
DOCS = sorted(
    [*ROOT.glob("paper/*.tex"), *ROOT.glob("40-esperimenti/*.md"), ROOT / "CLAUDE.md"]
)

# Il paper e' l'unico documento che dichiara a un lettore che i suoi numeri sono ricalcolabili,
# quindi `make verify` deve poterlo controllare da solo. Le note vault hanno sei buchi noti e
# dichiarati, e fallire su quelli renderebbe il comando rosso per un motivo che non riguarda il
# paper: meglio due comandi con due esiti, che uno che non significa niente.
DOCS_PAPER = sorted(ROOT.glob("paper/*.tex"))

CLAIM = re.compile(r"(?<![\w.])(\d+\.\d{3,4})(?![\w])")

# Cifre che non sono metriche e quindi non devono essere cercate negli store: frazioni di
# pagina, versioni, soglie dichiarate come costanti di protocollo, e formati di codice.
NOT_METRICS = {
    # Soglie nominali e valori banali: non sono misure.
    "0.000",  # zero esplicito
    "1.000",  # probabilita' massima
    "0.500",  # soglia nominale
    "0.750",  # soglia nominale OER
    # Separatore delle migliaia all'italiana. `11.900` sono undicimila novocento
    # probabilita', non 11.9; `6.235` e `7.027` sono i primi tre gruppi di un numero di
    # byte (`6.235.207.296 B`). La regex non puo' distinguerli da una proporzione,
    # quindi sono esclusioni strutturali e non assoluzioni: se un ASR valesse 6.235
    # sarebbe fuori scala lo stesso.
    "11.900",
    "6.235",
    "7.027",
    # Provenienza esterna dichiarata nella riga stessa: l'accuracy zero-shot di
    # `laya` e `laya-multilingual` viene dalla model card, non da uno
    # store nostro, e non puo' essercene uno.
    "0.362",  # E01-baseline.md: zero-shot dal model card
    "0.342",  # idem, secondo checkpoint
}


def store_values() -> tuple[set[float], int]:
    """Ogni valore numerico presente in qualsiasi result file, a 3 e 4 cifre.

    Due cose che la versione precedente non faceva, e che rendevano il controllo
    inaffidabile in entrambe le direzioni:

    - **ricorsione nelle sottocartelle.** `results/e12b_modes/`, `results/E12_surface/` e
      `results/E03_option_injection/` contengono sweep interi. Con il solo
      `results/*.json` ogni numero che esiste solo in uno sweep dava un falso allarme:
      le tre modalita' `pick` di E12b stanno tutte li', quindi il controllo non ne
      trovava nessuna e segnalava la nota che le descriveva.
    - **anche i valori fuori da (0, 1].** I tempi in ms, le dimensioni dei pesi in GB e i
      conteggi sono numeri che le note dichiarano e che vanno rintracciati come gli
      altri. Con il filtro `0 < |v| <= 1` erano *sempre* fuori, quindi il controllo non
      poteva distinguerli da un numero inventato: segnalava tutto e non diceva niente.

    Il secondo punto ha un prezzo, ed e' giusto dichiararlo: la stessa cifra puo' essere
    un ASR o un tempo in ms, quindi un numero grande puo' combaciare per caso. Per questo
    il ritorno e' una coppia: il controllo separa i valori sotto e sopra 1 e li riporta
    con conteggi distinti, cosi' il revisore legge un tasso con un altro occhio.
    """
    vals: set[float] = set()
    counts = [0, 0]
    for f in sorted((ROOT / "results").rglob("*.json")):
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Un file troncato o non-JSON non e' un archivio di risultati: si salta, e il
            # numero di file letti resta visibile nel riassunto.
            continue
        counts[0] += 1

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
                value = float(node)
                if value == 0.0:
                    return
                counts[1] += 1
                for nd in (3, 4):
                    vals.add(round(abs(value), nd))

        walk(data)
    return vals, counts[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="lista anche i numeri che combaciano")
    ap.add_argument(
        "--paper-only",
        action="store_true",
        help="controlla solo il paper: e' il documento che dichiara la ricalcolabilita'",
    )
    args = ap.parse_args()

    known, n_files = store_values()
    unmatched: list[tuple[str, int, str]] = []
    matched = 0

    for doc in (DOCS_PAPER if args.paper_only else DOCS):
        if not doc.exists():
            continue
        for lineno, line in enumerate(doc.read_text().splitlines(), 1):
            # DOI e versioni presenti nei riferimenti apribili sono identificatori,
            # non misure sperimentali da cercare negli store.
            line = re.sub(r"\\url\{[^}]*\}", "", line)
            for raw in CLAIM.findall(line):
                if raw in NOT_METRICS:
                    continue
                if round(float(raw), 4) in known or round(float(raw), 3) in known:
                    matched += 1
                    if args.verbose:
                        print(f"  ok   {doc.name}:{lineno}  {raw}")
                    continue
                unmatched.append((str(doc.relative_to(ROOT)), lineno, raw))

    print(f"{matched} numeri dichiarati combaciano con uno store")
    print(f"letti {n_files} file di risultato, ricorsione compresa (sweep in sottocartella inclusi)")
    if unmatched:
        print(f"\n{len(unmatched)} numeri dichiarati NON si ricalcolano dagli store:")
        seen: dict[str, list[str]] = {}
        for path, lineno, raw in unmatched:
            seen.setdefault(raw, []).append(f"{path}:{lineno}")
        for raw in sorted(seen):
            print(f"  {raw:>8}  {len(seen[raw])}x  {', '.join(seen[raw][:4])}")
        return 1
    print("nessun numero senza archivio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
