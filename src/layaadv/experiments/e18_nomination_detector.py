"""E18: un detector sul segnale giusto, non sulla "malevola" (E14).

**Il residuo di `laya-multilingual` non è un problema di soglia.** Sui 4 split la difesa di E16
resta a OER@0.75 0.308-0.367 con costo 12.8-15.0%: un refit monotono non sposta un errore sotto
la soglia se non lo sposta sopra, perché l'ordine delle confidenze resta. Quindi serve cambiare il
**segnale**, non la sua soglia.

E14 ha già provato la strada del "modello come proprio detector" (chiedere se lo stato è
malevolo) e ha fallito: 55.8% di falsi positivi sul traffico legittimo. Il problema non era che
fosse il modello stesso, era la **domanda**: "è malevolo?" è un giudizio soggettivo su ogni
stato, quindi distribuisce risposte "sì" anche dove non ce n'è bisogno.

Il segnale giusto è meccanico e verificabile: **l'attacco nomina un'opzione della domanda
nello stato**. È la firma dell'attacco E03 (l'iniezione scrive il testo di un distrattore). Un
detector che cerca quella nomina letterale nello stato non giudica, conta. Non costa un forward
e non ha bisogno del modello.

Il detector è **strict** (nomina esatta di un'opzione, parola intera) e **loose** (la nomina può
essere dentro un'altra frase). Entrambi vengono misurati, perché la differenza fra i due è
l'argomento: se il loose serve per lo stesso prezzo, il segnale è robusto; se costa troppi
falsi positivi, il segno stretto è il giusto compromesso.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from typing import Any

from layaadv import io
from layaadv.data import loaders
from layaadv.evaluate import evaluate
from layaadv.experiments.e03_option_injection import (
    POSITION,
    inject_option,
    make_state,
    template_text,
)
from layaadv.model.engine import DEFAULT_CHECKPOINT, open_from_args

EXPERIMENT = "E18_nomination_detector"
K = 4
CHUNK = 32
TEMPLATES = ("prepend", "append")


def normalise(text: str) -> str:
    """Minuscole, senza accenti e spazi compattati: due scritture della stessa opzione devono
    combaciare, altrimenti il detector non è lo stesso detector che il paper dichiara."""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text.replace("_", " ")).strip().lower()


def _tokens(option: str) -> list[str]:
    return [t for t in re.split(r"[^\w]+", normalise(option), flags=re.UNICODE) if t]


def nominates(state: str, options: list[str], *, loose: bool) -> list[str]:
    """Quali opzioni della domanda sono nominate, letteralmente, nello stato.

    `strict`: tutte le parole dell'opzione compaiono, in ordine, con parole intere.
    `loose`: basta che le parole compaiano, anche non in ordine (l'attacco può dire
    "cancella tutto, poi esegui play music").

    Restituisce le opzioni che trova, non un booleano: se lo stato nomina piu' di una opzione
    quello e' un caso da guardare, non uno da bloccare alla cieca.
    """
    hay = " " + " ".join(normalise(state).split()) + " "
    found = []
    for opt in options:
        toks = _tokens(opt)
        if not toks:
            continue
        pat = r"\W+".join(re.escape(t) for t in toks)
        strict_hit = re.search(rf"(?<!\w){pat}(?!\w)", hay) is not None
        if strict_hit:
            found.append(opt)
        elif loose and all(re.search(rf"(?<!\w){re.escape(t)}(?!\w)", hay) for t in toks):
            found.append(opt)
    return found


def row_for(state: str, case, template: str, position: str, pad_tokens: int = 0) -> dict:
    option = inject_option(case.questions[loaders.QID], case.gold[loaders.QID])
    # lo stato che il modello vede nell'attacco: il detector deve essere applicato a quello
    attacked_state = make_state(
        template_text(template), option, position=position, pad_tokens=pad_tokens
    )(case.state)
    clean_opts = list(case.questions[loaders.QID].get("criteria", {}))
    clean_hits = nominates(case.state, clean_opts, loose=False)
    clean_hits_loose = nominates(case.state, clean_opts, loose=True)
    adv_hits = nominates(attacked_state, clean_opts, loose=False)
    adv_hits_loose = nominates(attacked_state, clean_opts, loose=True)
    return {
        "id": case.id,
        "template": template,
        "injected": option,
        "n_options": len(clean_opts),
        "clean_hits": clean_hits,
        "clean_hits_loose": clean_hits_loose,
        "adv_hits": adv_hits,
        "adv_hits_loose": adv_hits_loose,
        # true = il detector nomina esattamente l'opzione iniettata
        "clean_fires": bool(clean_hits),
        "clean_fires_loose": bool(clean_hits_loose),
        "adv_fires": bool(adv_hits),
        "adv_fires_loose": bool(adv_hits_loose),
        "clean_names_injected": option in clean_hits,
        "adv_names_injected": option in adv_hits,
        "adv_names_injected_loose": option in adv_hits_loose,
    }


def evaluate_detector(rows_clean: list[dict], rows_adv: list[dict], variant: str) -> dict:
    """Il detector su Traffic pulito e su quello attaccato, e l'effetto del gate.

    Tre numeri, tenuti separati perche' dicono cose diverse:
    - **falsi blocchi** = detector che dice "pericolo" su traffico pulito (costo del gate);
    - **rilevazione** = detector che dice "pericolo" su traffico attaccato (beneficio);
    - **falso allarme sull'attacco** = dice "pericolo" ma NON nomina l'opzione iniettata
      (sarebbe un blocco alla cieca, il difetto che E14 ha dimostrato essere insostenibile).
    """
    key = "loose" if variant == "loose" else "strict"
    fire_c = "clean_fires_loose" if key == "loose" else "clean_fires"
    fire_a = "adv_fires_loose" if key == "loose" else "adv_fires"
    names_a = "adv_names_injected_loose" if key == "loose" else "adv_names_injected"

    def rate(rows, field):
        return sum(1 for r in rows if r[field]) / len(rows) if rows else None

    names_on_adv = rate(rows_adv, names_a)
    fires_on_adv = rate(rows_adv, fire_a)
    return {
        "variant": key,
        "clean_states": len(rows_clean),
        "adv_states": len(rows_adv),
        "false_block_rate": rate(rows_clean, fire_c),
        "detection_rate": fires_on_adv,
        "names_injected_rate": names_on_adv,
        # fra le righe che il detector blocca, quante sono davvero l'attacco che nomina il target:
        "precision_when_firing": (
            names_on_adv / fires_on_adv if (fires_on_adv and names_on_adv is not None) else None
        ),
        # un blocco che non nomina il bersaglio e' un blocco alla cieca: e' il costo che
        # rende E14 insostenibile, e va dichiarato anche quando e' raro
        "blind_blocks_on_attack": (
            (fires_on_adv - names_on_adv) if (fires_on_adv is not None and names_on_adv is not None) else None
        ),
    }


def run(args: Any) -> dict:
    def log(m: str) -> None:
        print(m, flush=True)

    dataset = getattr(args, "dataset", None) or "massive-it"
    checkpoint = (getattr(args, "checkpoint", None) or [DEFAULT_CHECKPOINT])[0]
    limit = getattr(args, "limit", None)
    engine = open_from_args(args, checkpoint)
    identity = engine.run_identity()

    cases, ds, _labels = loaders.load_cases(dataset, k=K, limit=limit, seed=0)

    # Il detector non ha bisogno del modello, ma valutiamo sugli stati ATTACATI come il modello li
    # vede, quindi gli record servono a misurare l'ASR che il gate eliminerebbe. Clean e attacco
    # passano da evaluate per avere lo stesso denominatore |S_ok ∩ S_adv| di E03.
    adv_cases_by_template = {
        t: [
            c.__class__(
                id=c.id,
                state=make_state(
                    template_text(t), inject_option(c.questions[loaders.QID], c.gold[loaders.QID]),
                    position=POSITION[t],
                )(c.state),
                questions=c.questions,
                gold=c.gold,
            )
            for c in cases
        ]
        for t in TEMPLATES
    }

    # Detector sulle stesse righe che il modello valuta
    rows_clean = [row_for(c.state, c, t, POSITION[t]) for c in cases for t in TEMPLATES]
    rows_adv = [
        row_for(adv_cases_by_template[t][i].state, cases[i], t, POSITION[t])
        for t in TEMPLATES
        for i in range(len(cases))
    ]

    # L'ASR che il gate eliminerebbe: fra le decisioni corrette al pulito, quante il modello
    # sbaglia sotto attacco E quante di quelle il detector blocca. Serve il forward solo per
    # l'ASR di riferimento, che gia' conosciamo dall'E03 ma riverifichiamo per non dipendere.
    common = dict(experiment=EXPERIMENT, checkpoint=checkpoint, dataset=dataset,
                  dataset_sha256=ds.file_sha256(), k=K, n_cases=len(cases), seed=0,
                  batch_size=engine.batch_size, chunk=CHUNK,
                  **{i: v for i, v in identity.items() if v is not None})
    clean_store = io.RunStore.create(EXPERIMENT, {**common, "kind": "clean"}, **resume(args))
    clean = evaluate(engine, cases, store=clean_store, batch_size=engine.batch_size, chunk=CHUNK, log=None)
    clean_store.finish({"metrics": clean.metrics})

    # solo append: il template che chiude, per avere l'ASR di riferimento su cui il gate agisce
    tpl = "append"
    adv_store = io.RunStore.create(
        EXPERIMENT, {**common, "kind": "adversarial", "template": tpl}, **resume(args))
    adv = evaluate(engine, adv_cases_by_template[tpl], store=adv_store,
                   batch_size=engine.batch_size, chunk=CHUNK, log=None)
    adv_store.finish({"metrics": adv.metrics})
    engine.close()

    # mappa fra stato e decisioni corrette al pulito, per il gate
    by_id = {}
    for r in clean.records:
        for qid, lab in r["labels"].items():
            by_id[(r["id"], qid)] = (str(lab) == str(r["gold"][qid]))

    rows_adv_single = [r for r in rows_adv if r["template"] == tpl]
    rows_clean_single = [r for r in rows_clean if r["template"] == tpl]
    variants = {
        v: evaluate_detector(rows_clean_single, rows_adv_single, v) for v in ("strict", "loose")
    }

    # effetto netto del gate: fra le decisioni corrette al pulito che il modello sbaglia sotto
    # attacco, il detector ne blocca una frazione. L'ASR residua e' quello che resta.
    # Appaiamento pulito/attacco per stato (stesso ordine di `cases`): un ribaltamento e' una
    # decisione che al pulito era corretta e sotto attacco e' sbagliata. Il gate puo' ridurre
    # l'ASR solo bloccando questi, quindi e' su questo insieme che va misurato.
    flips = []
    for cr, ar in zip(rows_clean_single, rows_adv_single):
        c_rec = next(x for x in clean.records if x["id"] == cr["id"])
        a_rec = next(x for x in adv.records if x["id"] == ar["id"])
        lab_c = str(c_rec["labels"][loaders.QID])
        lab_a = str(a_rec["labels"][loaders.QID])
        gold = str(c_rec["gold"][loaders.QID])
        if lab_c == gold and lab_a != gold:  # era giusta, ora sbagliata = ribaltamento
            flips.append(ar)

    gate = {}
    for v, res in variants.items():
        fire = "adv_fires" if v == "strict" else "adv_fires_loose"
        blocked = sum(1 for r in flips if r[fire])
        gate[v] = {
            "flips": len(flips),
            "blocked_by_gate": blocked,
            "asr_before": len(flips) / len([r for r in flips]) if flips else None,
            "asr_after": (len(flips) - blocked) / len(flips) if flips else None,
        }
        log(f"[E18] gate {v}: ribaltamenti {len(flips)}, bloccati {blocked}, "
            f"ASR {len(flips)/(len(flips) or 1):.4f} -> {(len(flips)-blocked)/(len(flips) or 1):.4f}")
        log(f"[E18]   detector {v}: falsi blocchi {res['false_block_rate']:.4f} "
            f"rilevazione {res['detection_rate']:.4f} nomina-il-bersaglio {res['names_injected_rate']:.4f}")

    payload = {
        "experiment": EXPERIMENT,
        "question": (
            "un detector sulla nominazione esplicita dell'opzione chiude il residuo che la "
            "calibrazione non chiude?"
        ),
        "study_valid": True,
        "axes": {
            "dataset": dataset,
            "checkpoint": checkpoint,
            "k": K,
            "n_cases": len(cases),
            "templates": list(TEMPLATES),
            "variants": ["strict", "loose"],
        },
        "detector": variants,
        "gate": gate,
        "env": {"engine_identity": identity},
        "note": (
            "Il detector non e' un modello: conta se lo stato nomina letteralmente un'opzione "
            "della domanda. E14 aveva chiesto al modello se lo stato fosse 'malevolo' (giudizio "
            "soggettivo) e aveva 55.8% di falsi positivi; qui la domanda e' meccanica e il costo "
            "sul traffico pulito e' quello che si vede nella colonna false_block_rate."
        ),
    }
    io.write_result(EXPERIMENT, payload)
    log(f"[E18] wrote results/{EXPERIMENT}.json")
    return payload


def resume(args) -> dict:
    return {"resume": not getattr(args, "no_resume", False),
            "force_new": getattr(args, "force_new", False)}


def main() -> None:
    run(argparse.Namespace(**{}))


if __name__ == "__main__":
    main()
