"""E12b — superficie lessica per sinonimi rari, scelta posizione per posizione.

Perché questo modulo esiste separato
------------------------------------
Il primo operatore lessicale provato sostituiva bigrammi, cioè guardava l'ordine delle parole e non
usava WordNet. È risultato inerte (zero ribaltamenti) e il suo ramo è stato rimosso, insieme al
suo esperimento: un operatore inerte tenuto in libreria è solo codice morto. Qui l'operatore è
diverso per costruzione.

Cosa cambia rispetto all'operatore rimosso
------------------------------------------
Il vocabolario è **derivato dagli stati reali** con WordNet, non scritto a mano:

1. spaCy (`en_core_web_sm`) etichetta lo stato e tiene solo i token di
   contenuto: esclude connettivi (CCONJ/SCONJ), articoli (DET), preposizioni
   (ADP), ausiliari, punti e virgole (PUNCT/SYM), numeri e identificatori.
   Il modello **inglese** perché il dataset è inglese: con il modello IT 5
   token su 14 restano `X` e il filtro non avrebbe scelto nulla.
2. WordNet dà i sinonimi dello stesso POS e `Lemma.count()` dà la frequenza
   d'uso. Si tiene solo ciò che è **meno frequente** dell'originale: la
   perturbazione deve essere fuori distribuzione, non una para in più.
3. Ogni generazione attacca **una posizione nuova** e la mutazione si
   **accumula** sullo stato migliore finora. Questo è il punto: il GA vecchio
   ricombinava due individui e perdeva quasi sempre la mutazione utile.

Non è più un GA elitista ma una hill-climbing guidata dalla fitness con
popolazione locale ≥ 3 per posizione. Il termine "generazione" resta perché il
ciclo e i vincoli sono gli stessi.

Metodo
------
    per ogni caso:
      candidati = posizioni di contenuto con ≥3 sinonimi più rari
      ordina i candidati in modo casuale
      stato_corrente = testo pulito
      per gen in 1..G:                      # early stopping dentro
        pos = prossimo candidato non ancora usato
        popolazione = [stato_corrente con pos -> sinonimo] × ≥3   # candidati diversi
        valuta in un batch, prendi il migliore
        il campione avanza **anche a parita'** (con fitness 0 ovunque
        l'elitismo stretto congelerebbe il ciclo al primo passo)
        stop se c'è un flip, o se `patience` generazioni non migliorano
    fitness = stessa di E12 (flip, confidenza, KL) sulle decisioni corrette al pulito

Cosa è study-valid
----------------------
La mutazione tocca solo lo stato: `questions`, `criteria` e gold restano
identici. I numeri sono protetti esplicitamente perché su typed-decisions il
gold è portato dai numeri (codici fattura, `$50`, durate, date): il filtro
numerico è il primo controllo e non è negoziabile. Sostituire `exceed` con
`outstrip` non sposta il gold; sostituire `32.5` con `99.0` sì, e quel
candidato non entra mai nella popolazione.

Uso:
    uv run adv e12b --limit 30
    uv run adv e12b --limit 100 --generations 10 --patience 5   # taglio a plateau
"""

from __future__ import annotations

import random
import re

from layaadv import io, paths
from layaadv.data import typed_decisions as td
from layaadv.evaluate import evaluate
from layaadv.experiments.e12_ga import (
    Individual,
    protected_tokens,
    score_population,
)
from layaadv.model.engine import DEFAULT_CHECKPOINT, open_from_args

SEED = 0
CHUNK = 32
DEFAULT_POPULATION = 3  # il requisito: almeno 3 candidati diversi per posizione
DEFAULT_GENERATIONS = 10
DEFAULT_SPACY_MODEL = "en_core_web_md"

# POS tenuti. `X` escluso di proposito: su testo tecnico spaCy inglese lo assegna
# alle parole che non conosce, e non sapremmo se sono connettivi o contenuto.
# PROPN escluso perché sono identificatori (`trace_summary`), che il gold usa.
KEEP_POS = ("NOUN", "VERB", "ADJ", "ADV")
DROP_POS = frozenset(
    {"CCONJ", "SCONJ", "DET", "ADP", "AUX", "PART", "PRON", "INTJ", "PUNCT", "SYM", "NUM", "X", "PROPN"}
)
WN_POS = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}


def build_nlp(model: str = DEFAULT_SPACY_MODEL):
    """spaCy inglese. I POS sono il filtro, quindi il modello conta.

    `en_core_web_md` e non `sm`: sul testo tecnico degli stati il modello piccolo
    lasciava 56 token su X contro 38, e non vedeva nessuna entita' ORG (5 su md).
    Non è un abisso, ma il filtro è fatto di POS, quindi si paga.
    """
    import spacy

    return spacy.load(model)


def lemma_frequency(lemma_name: str, pos: str) -> int:
    """Frequenza d'uso da WordNet (`Lemma.count`), massima sui synset. -1 se assente.

    Il massimo è obbligatorio per coerenza: una parola puo' comparire in più
    sinnomi e con conteggi diversi, e prendere il primo darebbe due frequenze
    diverse per la stessa parola (misurato: `surpass` 0 in un sinnomo, 2 in
    un altro, quindi l'ordinamento per rarita' non era definito).
    """
    from nltk.corpus import wordnet as wn

    found = -1
    for syn in wn.synsets(lemma_name.replace(" ", "_"), pos):
        for lem in syn.lemmas():
            if lem.name().replace("_", " ") == lemma_name:
                found = max(found, lem.count())
    return found


def rare_synonyms(
    token, min_alternatives: int, max_freq_ratio: float = 0.0, pick: str = "rarest"
) -> list[str]:
    """Sinonimi dello stesso POS, dal piu' raro, entro una soglia di frequenza.

    `max_freq_ratio` è il parametro che rende l'ipotesi testabile:
      - 0 (default): stretto, `count(sostituto) < count(originale)`. È il
        protocollo di E12b: la perturbazione deve essere **fuori** distribuzione.
      - 1: accetta anche pari frequenza, e basta.
      - >1: accetta sinonimi piu' comuni, fino a `ratio × count(originale)`.

    `pick` sceglie **quale parte della pool** prendere:
      - `rarest`: i piu' rari entro soglia (default, il protocollo E12b)
      - `commonest`: i piu' comuni, che sono anche i piu' fedeli al contesto
      - `mid`: metà della distribuzione

    La ragione di `commonest`: alzare la soglia da sola non serve a niente, perché
    la pool è gia' dominata dai rari (misurato: 333 posizioni strette contro 393 con
    soglia aperta, e gli stessi sostituti). Il vero bivio è un altro, ed è quello che
    distingue una superficie d'attacco dalla distruzione di significato: i sinonimi
    *rari* sbagliano il senso (`internal` -> `intragroup`, `model` -> `manikin`),
    quelli *comuni* lo rispettano (`internal` -> `inner`, `steps` -> `step`). Se
    l'attacco funziona anche con i comuni, allora E12b non stava rompendo il testo.
    """
    from nltk.corpus import wordnet as wn

    wn_pos = WN_POS.get(token.pos_)
    if wn_pos is None:
        return []
    original = token.lemma_.lower()
    freq_original = lemma_frequency(original, wn_pos)
    if freq_original < 0:
        return []
    # Candidati: gli stessi vincoli di WordNet, ma filtrati. Servono tre filtri:
    #   - sinonimi multi-parola ("go beyond") frammentano la frase
    #   - le maiuscole iniziali indicano nomi propri, non sinonimi: `service`
    #     dava "Robert William Service" e "Service". Un nome proprio non
    #     preserva il significato, è rumore puro
    #   - idem per i trattini dentro la parola
    names: set[str] = set()
    for syn in wn.synsets(original.replace(" ", "_"), wn_pos):
        for lem in syn.lemmas():
            name = lem.name().replace("_", " ")
            if name == original:
                continue
            if not all(ch.isalpha() or ch == " " for ch in name):
                continue
            if not name[:1].islower():
                continue
            names.add(name)
    # La frequenza di un sostituto si misura sulla parola, non sui sinset che
    # condivide con l'originale: altrimenti due parole hanno due frequenze e
    # l'ordinamento per rarita' non è definito (misurato su `exceed`).
    scored = sorted((lemma_frequency(name, wn_pos), name) for name in names)
    if max_freq_ratio and max_freq_ratio > 0:
        limit = max(freq_original, int(freq_original * max_freq_ratio))
        pool = [(f, name) for f, name in scored if 0 <= f <= limit]
    else:
        pool = [(f, name) for f, name in scored if 0 <= f < freq_original]
    if pick == "commonest":
        # l'estremo opposto: i sinonimi piu' comuni, che sono anche i più
        # fedeli al contesto (`internal` -> `inner`, non `intragroup`). Serve a
        # separare "la superficie esiste" da "i flip erano distruzione di
        # significato": se l'attacco funziona anche così, è superficie.
        pool = sorted(pool, reverse=True)
    elif pick == "mid":
        # metà della distribuzione: perturbazione media, significato intatto.
        # Si ruota la pool ordinata per partire dal centro, cosi' i primi 3
        # presi sono quelli di frequenza mediana e non piu' i piu' rari.
        ordered = sorted(pool)
        take = max(min_alternatives, DEFAULT_POPULATION)
        start = max(0, len(ordered) // 2 - take // 2)
        pool = ordered[start:] + ordered[:start]
    return [name for _, name in pool][:max(min_alternatives, DEFAULT_POPULATION)]


def content_candidates(
    nlp,
    text: str,
    min_alternatives: int,
    max_freq_ratio: float = 0.0,
    pick: str = "rarest",
) -> list[dict]:
    """Posizioni sostituibili: contenuto, non protetto, con sinonimi rari.

    Un token è scartato se non è contenuto (POS in DROP_POS), se è stopword,
    se è un token protetto (numeri, importi, identificatori) o se non ha
    abbastanza sinonimi entro soglia per una popolazione utile.
    """
    doc = nlp(text)
    protected = protected_tokens(doc)
    out: list[dict] = []
    for tok in doc:
        if tok.i in protected or tok.pos_ in DROP_POS or tok.is_stop:
            continue
        if tok.pos_ not in KEEP_POS:
            continue
        alts = rare_synonyms(tok, min_alternatives, max_freq_ratio, pick)
        if len(alts) < min_alternatives:
            continue
        out.append(
            {
                "index": tok.i,
                "surface": tok.text,
                "lemma": tok.lemma_.lower(),
                "pos": tok.pos_,
                "original_freq": lemma_frequency(tok.lemma_.lower(), WN_POS[tok.pos_]),
                "alternatives": alts,
            }
        )
    return out


def substitute(doc, token_index: int, replacement: str) -> str:
    """Ricostruisce il testo con il token `token_index` sostituito.

    Si ricostruisce da `doc.text` percorrendo i token, non indicizzando una
    `split()`: gli indici di spaCy non coincidono con le parole separate da
    spazi (misurato: 100 token contro 37 su uno stato reale), e i trattini
    spezzano le parole. Lavorare sul doc corrente tiene gli offset corretti
    anche dopo che le mutazioni precedenti hanno cambiato la lunghezza del testo.
    """
    repl = replacement
    pieces: list[str] = []
    cursor = 0
    for tok in doc:
        pieces.append(doc.text[cursor : tok.idx])
        if tok.i == token_index:
            repl = replacement.capitalize() if tok.text[:1].isupper() else replacement
            pieces.append(repl)
        else:
            pieces.append(tok.text)
        cursor = tok.idx + len(tok.text)
    pieces.append(doc.text[cursor:])
    return "".join(pieces)


def passes_constraints(original: str, candidate: str) -> tuple[bool, str]:
    """Numeri invariati, token protetti identici, drift sotto soglia.

    Il controllo dei numeri viene prima di tutto: su typed-decisions il gold è
    portato dai numeri, ed è l'unico invariante che non si puo' aggirare.
    """
    if re.findall(r"\d+(?:[.,]\d+)*", original) != re.findall(r"\d+(?:[.,]\d+)*", candidate):
        return False, "numeri cambiati"
    so, sc = set(original.lower().split()), set(candidate.lower().split())
    if not so or not sc:
        return False, "stato vuoto"
    jac = len(so & sc) / len(so | sc)
    if jac < 0.5:
        return False, f"drift {jac:.2f}"
    return True, "ok"


def run(args) -> dict:
    limit = getattr(args, "limit", None) or 100
    pop_size = getattr(args, "population", None) or DEFAULT_POPULATION
    generations = getattr(args, "generations", None) or DEFAULT_GENERATIONS
    # patience >= generations: il ciclo visita tutte le posizioni e si ferma solo
    # sul flip. Con patience < generations diventa un taglio a plateau: ferma
    # prima, risparmia compute, ma copre meno posizioni. Il default è 0 = spento.
    patience = getattr(args, "patience", None) or 0
    min_alternatives = getattr(args, "min_alternatives", None) or DEFAULT_POPULATION
    # soglia di frequenza sui sinonimi: 0 = solo piu' rari dell'originale (E12b),
    # >1 = accetta anche piu' comuni fino a ratio × freq(originale). Vedi
    # rare_synonyms: è un parametro perche' le due scelte tirano in direzioni
    # opposte e non si sa quale vinca.
    max_freq_ratio = getattr(args, "max_freq_ratio", None) or 0.0
    pick = getattr(args, "pick", None) or "rarest"
    spacy_model = getattr(args, "spacy_model", None) or DEFAULT_SPACY_MODEL
    variant = getattr(args, "variant", None) or "V2"
    min_agreement = getattr(args, "min_agreement", None)
    log = lambda m: print(m, flush=True)  # noqa: E731

    from layaadv.experiments.e12_ga import VARIANTS

    weights = VARIANTS[variant]

    cases = td.load_cases("test", limit=limit, min_agreement=min_agreement)
    if not cases:
        raise SystemExit("nessun caso caricato")
    checkpoint = getattr(args, "checkpoint", None) or [DEFAULT_CHECKPOINT]
    checkpoint = checkpoint[0] if isinstance(checkpoint, list) else checkpoint
    engine = open_from_args(args, checkpoint)
    nlp = build_nlp(spacy_model)
    log(
        f"[E12b] sinonimi rari, N={len(cases)}, pop={pop_size}, {generations} generazioni, "
        f"patience={patience}, variante {variant} pesi {weights}"
    )
    log(f"[E12b] spacy={spacy_model}, pick={pick}, max_freq_ratio={max_freq_ratio or 'stretto'}")

    # Come E12: la baseline non cambia mai, quindi sta in un RunStore invece di essere
    # ricalcolata a ogni invocazione del ciclo.
    clean_store = io.RunStore.create(
        "E12b_rare_synonym_clean",
        {
            "experiment": "E12b_rare_synonym_clean",
            "checkpoint": checkpoint,
            "dataset": "typed-decisions/test",
            "dataset_sha256": td.dataset_sha256("test"),
            "subset_hash": td.subset_hash(cases),
            "n_cases": len(cases),
            "min_agreement": min_agreement,
            "batch_size": engine.batch_size,
            "chunk": CHUNK,
            **engine.run_identity(),
            "weight_sha256": engine.weight_sha256(),
        },
        resume=not getattr(args, "no_resume", False),
        force_new=getattr(args, "force_new", False),
    )
    clean_result = evaluate(
        engine, cases, store=clean_store, batch_size=engine.batch_size, chunk=CHUNK
    )
    clean_store.finish({"metrics": clean_result.metrics})
    clean_by_id = {r["id"]: r for r in clean_result.records}
    log(f"[E12b] baseline acc={clean_result.metrics['accuracy']:.4f}")

    rng = random.Random(f"{SEED}:{variant}")
    per_case: dict[str, dict] = {}
    no_surface = 0

    for case in cases:
        candidates = content_candidates(nlp, case.state, min_alternatives, max_freq_ratio, pick)
        if not candidates:
            no_surface += 1
            continue
        rng.shuffle(candidates)
        current = Individual(state=case.state, lineage=["clean"])
        current_doc = nlp(current.state)
        current_score = score_population(engine, [case], [current], clean_by_id, weights)[0]
        best = current_score
        history = []
        first_flip_gen = None
        used_positions: list[int] = []
        stale = 0

        for gen in range(generations):
            # una posizione nuova a ogni generazione: è la "gen successiva
            # altera un altro pezzo" del protocollo.
            candidate = next(
                (c for c in candidates if c["index"] not in used_positions), None
            )
            if candidate is None:
                break
            used_positions.append(candidate["index"])
            population = [
                Individual(
                    state=substitute(current_doc, candidate["index"], alt),
                    lineage=current.lineage + [f"{candidate['surface']}->{alt}"],
                )
                for alt in candidate["alternatives"][:pop_size]
            ]
            population = [
                ind for ind in population if passes_constraints(case.state, ind.state)[0]
            ]
            if not population:
                stale += 1
                if patience and stale >= patience:
                    break
                continue
            scored = score_population(engine, [case], population, clean_by_id, weights)
            champion = max(scored, key=lambda s: s["fitness"])
            # `best` è il massimo storico (quello che riportiamo). `current`
            # avanza **anche a parita'**: con fitness 0 ovunque, un elitismo
            # stretto congela il ciclo al primo passo e le 10 generazioni
            # diventano una sola mutazione. A parita' non si perde nulla: il
            # campione è comunque il massimo della popolazione.
            if champion["fitness"] > best["fitness"]:
                best = champion
                stale = 0
            else:
                stale += 1
            current = champion["individual"]
            current_doc = nlp(current.state)
            if champion["flips"] > 0 and first_flip_gen is None:
                first_flip_gen = gen
            history.append(
                {
                    "gen": gen,
                    "position": candidate["index"],
                    "surface": candidate["surface"],
                    "original_freq": candidate["original_freq"],
                    "best_fitness": champion["fitness"],
                    "best_flips": champion["flips"],
                    "stale": stale,
                }
            )
            # early stopping: il flip trovato chiude il caso. Il taglio a plateau
            # è opt-in (`--patience N`): con patience 0 il ciclo visita tutte le
            # posizioni candidate, che è il protocollo di misura.
            if champion["flips"] > 0 or (patience and stale >= patience):
                break

        final = score_population(engine, [case], [best["individual"]], clean_by_id, weights)[0]
        per_case[case.id] = {
            "id": case.id,
            "state": best["individual"].state,
            "lineage": best["individual"].lineage,
            "final": final,
            "first_flip_gen": first_flip_gen,
            "n_candidates": len(candidates),
            "n_positions_used": len(used_positions),
            "history": history,
            "clean_correct": sum(
                1 for qid, g in case.gold.items() if clean_by_id[case.id]["labels"][qid] == g
            ),
        }
        if len(per_case) % 10 == 0:
            done = list(per_case.values())
            flipped = sum(1 for d in done if d["final"]["flips"] > 0)
            log(f"[E12b]   {len(per_case)}/{len(cases)} casi, almeno un flip: {flipped}")

    done = list(per_case.values())
    total_correct = sum(d["clean_correct"] for d in done)
    flipped_any = sum(1 for d in done if d["final"]["flips"] > 0)
    total_flips = sum(d["final"]["flips"] for d in done)
    gens = [d["first_flip_gen"] for d in done if d["first_flip_gen"] is not None]
    conf = [d["final"]["mean_conf_on_wrong"] for d in done if d["final"]["flips"] > 0]
    payload = {
        "experiment": "E12b_rare_synonym",
        "variant": variant,
        "weights": {"flip": weights[0], "conf": weights[1], "kl": weights[2]},
        "checkpoint": checkpoint,
        "n_cases": len(cases),
        "n_cases_without_surface": no_surface,
        "population": pop_size,
        "generations": generations,
        "patience": patience,
        "min_alternatives": min_alternatives,
        "max_freq_ratio": max_freq_ratio,
        "pick": pick,
        "spacy_model": spacy_model,
        "seed": SEED,
        "min_agreement": min_agreement,
        "baseline_accuracy": clean_result.metrics["accuracy"],
        "summary": {
            "cases_with_a_flip": flipped_any,
            "rate_cases_flipped": round(flipped_any / len(done), 4) if done else None,
            "decision_level_asr": round(total_flips / total_correct, 4) if total_correct else None,
            "decisions_correct_at_clean": total_correct,
            "total_flips": total_flips,
            "median_gen_to_first_flip": sorted(gens)[len(gens) // 2] if gens else None,
            "mean_positions_used": (
                round(sum(d["n_positions_used"] for d in done) / len(done), 2) if done else None
            ),
            "mean_candidates_per_case": (
                round(sum(d["n_candidates"] for d in done) / len(done), 2) if done else None
            ),
            "mean_conf_on_wrong": round(sum(conf) / len(conf), 4) if conf else None,
        },
        "per_case": per_case,
        "env": io.env_fingerprint(engine.weight_sha256()),
    }
    io.write_result("E12b_rare_synonym", payload)
    io.write_jsonl(
        paths.RESULTS / "E12b_best.jsonl",
        [
            {
                "id": d["id"],
                "state": d["state"],
                "lineage": d["lineage"],
                "flips": d["final"]["flips"],
                "mean_conf_on_wrong": d["final"]["mean_conf_on_wrong"],
                "n_positions_used": d["n_positions_used"],
            }
            for d in done
        ],
    )
    log("[E12b] wrote results/E12b_rare_synonym.json e results/E12b_best.jsonl")
    log(f"[E12b] casi con almeno un flip: {flipped_any}/{len(done)}")
    engine.close()
    return payload
