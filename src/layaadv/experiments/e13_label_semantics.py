"""E13 — Label semantics: le opzioni sono un canale di informazione?

Claim da verificare
--------------------
Le note riportano "~22pp a K=4 da `calendar_query` a `I7`", senza alcuna run che lo
sostenga. Questo esperimento lo misura.

Why it matters
--------------
Il canale delle opzioni è l'attacco piu' economico in assoluto: se rinominare le
etichette basta a spostare l'accuracy, allora il modello non classifica fra concetti ma
lega una stringa. Ma **le etichette sono backend**: toccarle sposta il goalpost, quindi
questa non è una submission, è una misura diagnostica (→ 20-attack-surface/
superficie-attacco.md, "cosa NON è attaccabile"). Il valore che produce è sapere se il
canale è informativo, non un attacco.

Metodo
------
Stesso dataset, stesso K, stesso stato, stesse posizioni delle opzioni: cambia **solo**
il testo delle opzioni, che passa da `calendar_query` a `I7`. Il gold segue la rinomina,
altrimenti la domanda non avrebbe risposta. Tre braccia:

- ``real``   : etichette del dataset (baseline)
- ``short``  : ``I0..I{n}`` nello stesso ordine (numeri, 2 caratteri)
- ``prefix`` : ``I7: calendar_query`` — l'etichetta corta con la descrizione lunga

``prefix`` isola il contributo della *lunghezza* del testo da quello del *significato*:
se ``prefix`` recovers quasi tutto di ``real``, il modello legge la stringa lunga e la
abbreviazione non aggiunge informazione; se ``short`` regge, il formato non importa e il
canale è puramente posizionale.

Output: results/E13_label_semantics.json
"""

from __future__ import annotations

from layaadv import io
from layaadv.data import loaders
from layaadv.evaluate import evaluate
from layaadv.model.engine import DEFAULT_CHECKPOINT, open_from_args
from layaadv.schema import Case

DEFAULT_DATASET = "massive-it"
K_VALUES = [4, 8, 16]
SEED = 0
CHUNK = 32
SAMPLE = 1000  # sottoinsieme: l'effetto qui è grande (decine di pp), N=1000 basta

ARMS = ("real", "short", "prefix")


def rename_map(labels: list[str], arm: str) -> dict[str, str]:
    """New option text per arm, in a deterministic order."""
    ordered = sorted(labels)
    if arm == "real":
        return {lab: lab for lab in ordered}
    if arm == "short":
        return {lab: f"I{i}" for i, lab in enumerate(ordered)}
    if arm == "prefix":
        # I7 con la descrizione lunga: isola lunghezza del testo vs significato
        return {lab: f"I{i}: {lab}" for i, lab in enumerate(ordered)}
    raise ValueError(arm)


def apply_arm(case: Case, mapping: dict[str, str]) -> Case:
    """Same state, same option order, renamed option texts. Gold renamed to match."""
    qid = next(iter(case.questions))
    question = case.questions[qid]
    gold = case.gold[qid]
    criteria = {mapping[lab]: desc for lab, desc in question["criteria"].items()}
    return Case(
        id=case.id,
        state=case.state,
        questions={qid: {**question, "criteria": criteria}},
        gold={qid: mapping[gold]},
    )


def run(args) -> dict:
    limit = getattr(args, "limit", None) or SAMPLE
    k_values = getattr(args, "k", None) or K_VALUES
    # Vedi E03: il seed fissa il disegno delle opzioni per riga, quindi --seed è il modo
    # per misurare quanto pesa quella scelta invece di ereditarla.
    seed = getattr(args, "seed", None)
    seed = SEED if seed is None else seed
    arms = tuple(getattr(args, "arm", None) or ARMS)
    checkpoint = getattr(args, "checkpoint", None) or [DEFAULT_CHECKPOINT]
    checkpoint = checkpoint[0] if isinstance(checkpoint, list) else checkpoint
    log = lambda m: print(m, flush=True)  # noqa: E731

    engine = open_from_args(args, checkpoint)

    results = []
    for k in k_values:
        base_cases, ds, labels = loaders.load_cases(DEFAULT_DATASET, k=k, limit=limit, seed=seed)
        if not base_cases:
            raise SystemExit("nessun caso caricato")
        audit = loaders.check_head_budget(engine, base_cases, strict=False)
        if audit["over_budget"]:
            print(f"[E13] K={k} oltre budget della testa, sweep fermato", flush=True)
            break
        print(f"[E13] K={k}: {len(base_cases)} casi, head {audit['max_head_tokens']}", flush=True)

        for arm in arms:
            mapping = rename_map(labels, arm)
            cases = [apply_arm(c, mapping) for c in base_cases]
            # L'audit va fatto sulle questioni **del braccio**: `prefix` ha label
            # piu' lunghe di quelle reali, quindi l'audit sui casi base dichiarava
            # "0 opzioni perse" anche quando il braccio misurato era oltre budget.
            # Sulla base l'head era 50/73/120 token per i tre bracci, identico per
            # costruzione del test: non era l'audit di nessuno di loro.
            arm_audit = loaders.check_head_budget(engine, cases, strict=False)
            if arm_audit["over_budget"]:
                print(
                    f"[E13] braccio {arm} oltre budget della testa "
                    f"({arm_audit['head_max_len']}): opzioni o gold troncati, braccio saltato",
                    flush=True,
                )
                continue
            store = io.RunStore.create(
                "E13_label_semantics",
                {
                    "experiment": "E13_label_semantics",
                    "arm": arm,
                    "checkpoint": checkpoint,
                    "dataset": DEFAULT_DATASET,
                    "dataset_sha256": ds.file_sha256(),
                    "subset_hash": ds.subset_hash(base_cases),
                    "k": k,
                    "n_cases": len(cases),
                    "seed": seed,
                    "batch_size": engine.batch_size,
                    "chunk": CHUNK,
                    **engine.run_identity(),
                    "weight_sha256": engine.weight_sha256(),
                    # l'audit del braccio, non quello della base: le label di
                    # `prefix` sono piu' lunghe e il muro puo' stare li'.
                    "head_max_len": arm_audit["head_max_len"],
                    "max_head_tokens": arm_audit["max_head_tokens"],
                },
                resume=not getattr(args, "no_resume", False),
                force_new=getattr(args, "force_new", False),
            )
            result = evaluate(
                engine, cases, store=store, batch_size=engine.batch_size, chunk=CHUNK, log=log,
            )
            store.finish({"metrics": result.metrics})
            results.append(
                {
                    "arm": arm,
                    "k": k,
                    "chance": round(1 / k, 4),
                    # E13 non tocca lo stato: il Δtoken è 0 per costruzione, ed è
                    # proprio il controllo (le opzioni rinominate sono l'unica
                    # variabile, quindi l'accuracy si legge come canale, non come rumore).
                    "delta_tokens": 0.0,
                    "temperature": engine.decision_temperature("choice", k),
                    "run_id": store.run_id,
                    "head_audit": arm_audit,
                    "head_audit_base_questions": audit,
                    **result.metrics,
                }
            )
            print(f"[E13]   {arm:7s} acc={result.metrics['accuracy']:.4f}", flush=True)

    # delta vs etichette reali, a parita' di K
    by_k: dict[int, dict[str, float]] = {}
    for row in results:
        by_k.setdefault(row["k"], {})[row["arm"]] = row["accuracy"]
    for k, arms_acc in by_k.items():
        real = arms_acc.get("real")
        for arm, acc in arms_acc.items():
            if arm == "real" or real is None:
                continue
            row = next(r for r in results if r["k"] == k and r["arm"] == arm)
            row["drop_vs_real_pp"] = round(100 * (real - acc), 2)

    payload = {
        "experiment": "E13_label_semantics",
        "checkpoint": checkpoint,
        "dataset": DEFAULT_DATASET,
        "n_cases": limit,
        "k_values": k_values,
        "arms": list(arms),
        "seed": seed,
        "study_valid": False,
        "why_not": (
            "le opzioni sono backend: rinominarle cambia la domanda, quindi questa è una "
            "misura diagnostica del canale, non una submission (vedi superficie-attacco.md)"
        ),
        "results": results,
        "env": io.env_fingerprint(engine.weight_sha256()),
    }
    io.write_result("E13_label_semantics", payload)
    print(f"[E13] wrote results/E13_label_semantics.json ({len(results)} righe)", flush=True)
    engine.close()
    return payload
