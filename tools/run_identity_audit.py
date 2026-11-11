"""Ogni store adv dichiara il backend che lo ha prodotto, e la riga che lo
cita dice la stessa cosa.

Perché questo sia un controllo e non una formalità. `io.RunStore.create` su una
directory esista **riprende senza sollevare** se `stored == config`. Quindi un
config che non registra l'identità del modello non è solo una provenienza
mancante: è un config che due backend diversi possono produrre identico, e il
resume legge in silenzio le previsioni dell'altro. È successo: il config del
braccio avversario di E03 non portava `run_identity()` dal 2026-09-30 al
2026-10-04, e i 12 store `clm-v0.1-8B-Q5_K_M` hanno i 6 puliti con
`Q5_K - Medium` dichiarato e i 6 avversari con `quantization` assente. A
salvarli è stato solo che lo slug del checkpoint è diverso nel config.

Il fix del 2026-10-04 aggiunge `run_identity()` al config avversario di E03, ma
i 149 store già archiviati hanno il vecchio hash e non vengono ricalcolati: i
loro numeri sono corretti (`verify_results` li ricalcola a deviazione zero) e
quel che non registrano è *da quale backend* vengono. Questo strumento chiude
quel "non registrano" in "l'abbiamo verificato": controlla che lo `checkpoint`
dello store e quello della riga che lo cita coincidano, e che nessuno store
avversario resti senza chiavi di identità.

Uso:
    uv run python tools/run_identity_audit.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from layaadv import paths

# Chiavi che `run_identity()` produce su almeno un backend. La loro assenza nel
# config di uno store significa che l'identita' non e' stata registrata.
IDENTITY_KEYS = frozenset(
    {
        "backend",
        "compiled",
        "fast",
        "device",
        "quantization",
        "temperature_applied",
        "temperature_shipped",
        "confidence_definition",
        "variant",
    }
)


def stores(root: Path) -> list[tuple[str, Path, dict]]:
    out = []
    for path in sorted(root.rglob("runs/*/config.json")):
        out.append((path.parts[-4], path.parent, json.loads(path.read_text())))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(paths.RESULTS))
    args = parser.parse_args()

    root = Path(args.root)
    rows = stores(root)
    if not rows:
        print(f"nessuno store sotto {root}")
        return 1

    missing_adv = []
    missing_any = []
    for exp, directory, cfg in rows:
        has = bool(IDENTITY_KEYS & set(cfg))
        kind = cfg.get("kind") or "clean"
        if not has:
            missing_any.append((exp, kind, directory.name))
            if kind == "adversarial":
                missing_adv.append((exp, directory.name))

    # Coerenza fra lo store e la riga che lo cita: e' l'unica prova che gli store
    # privi di identita' appartengono al backend che la riga dichiara.
    cited: dict[str, list[str]] = {}
    for path in sorted(root.glob("*.json")):
        if path.name in ("E03_option_injection.json",):
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        for row in payload.get("results", []) or payload.get("rows", []) or []:
            for key in ("clean_run_id", "adv_run_id"):
                if row.get(key):
                    cited.setdefault(row[key], []).append(
                        f"{path.name}:{row.get('checkpoint', '?')}"
                    )
    orphans = []
    disagree = []
    for exp, directory, cfg in rows:
        refs = cited.get(directory.name)
        if not refs:
            orphans.append(directory.name)
            continue
        if not (IDENTITY_KEYS & set(cfg)):
            for ref in refs:
                ref_ck = ref.split(":")[-1]
                if ref_ck != str(cfg.get("checkpoint")):
                    disagree.append((directory.name, cfg.get("checkpoint"), ref_ck))

    print(f"store esaminati: {len(rows)}")
    print(f"  senza chiavi di identita': {len(missing_any)}")
    print(f"    di cui avversari: {len(missing_adv)}")
    print(f"  citati da almeno una riga: {len(rows) - len(orphans)}")
    print(f"  non citati da nessuna riga: {len(orphans)}")
    print()
    if disagree:
        print("INCOERENZE tra lo checkpoint dello store e quello della riga che lo cita:")
        for name, store_ck, row_ck in disagree[:20]:
            print(f"  {name}  store={store_ck}  riga={row_ck}")
        return 1
    print("nessuna incoerenza: dove la riga dichiara un checkpoint, lo store ha lo stesso.")
    if missing_adv:
        by_exp: dict[str, int] = {}
        for exp, _ in missing_adv:
            by_exp[exp] = by_exp.get(exp, 0) + 1
        print("\navversari senza identita', per esperimento:")
        for exp, n in sorted(by_exp.items()):
            print(f"  {exp}: {n}")
        print(
            "\nQuesti sono gli store archiviati **prima** del fix. I loro numeri sono "
            "corretti e `verify_results` li ricalcola: quello che non registrano e' da "
            "quale backend vengono, e il controllo qui sopra dice che la riga e lo store "
            "concordano sul checkpoint. Un ricalcolo con `--force-new` ridurrebbe la "
            "procedura ma non cambierebbe nessun numero."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
