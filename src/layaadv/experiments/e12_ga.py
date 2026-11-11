"""La macchina di ricerca per gli operatori lessicali (black-box L1).

Cosa c'era qui e cosa c'è ora
------------------------------
Questo modulo ospitava l'esperimento sui **bigrammi**, cioè una ricerca di sostituzioni lessicali
guidata da algoritmo genetico. Il risultato è stato **zero ribaltamenti in 30 casi su tutte e
tre le varianti di fitness**, e il ramo è stato rimosso: non era una scoperta, era un operatore
inerte, e l'esperimento che lo ha sostituito ([[E12b-sinonimi-rari]]) non ne ha bisogno.

Quello che resta è l'infrastruttura, ed è la parte che E12b riusa: la rappresentazione di un
individuo, il vincolo NER che protegge il gold, il filtro di drift, la divergenza KL e la fitness
a distribuzione. Il nome del modulo è "ga" e resta: è un algoritmo genetico, non un esperimento
sui bigrammi.

Perchè la fitness è a distribuzione e non a etichetta
--------------------------------------------------------
Il prior art su questo approccio (ora rimosso con il ramo) gira un GA nero su un classificatore
dove la fitness è "la label è cambiata". Qui la fitness puo' essere molto più ricca, perché
ogni candidato restituisce una **distribuzione** su tutte le opzioni e non solo l'etichetta:
permette di selezionare non solo gli stati che ribaltano la decisione, ma quelli che il modello
sbaglia *con fiducia* (la soglia OER di [[criteri-vittoria]]).

L1 black-box: si usano solo le risposte del modello. Nessun gradiente, nessun peso toccato.

Cosa è study-valid
----------------------
La mutazione tocca **solo lo stato**: `questions`, `criteria` e gold restano byte-identici alla
baseline pulita. Il vincolo NER di spaCy serve proprio a questo (le entità determinano il gold),
e il filtro di drift impedisce che il testo diventi semanticamente un altro. Se un candidato
sposta il gold, viene scartato: la fitness si calcola solo sulle decisioni che il modello azzeccava.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from layaadv.evaluate import evaluate
from layaadv.schema import Case

SEED = 0
# DEFAULT_POPULATION e DEFAULT_GENERATIONS stanno in e12_rare_syn, che è il suo unico
# consumatore: due costanti omonime con valori diversi (40/15 e 3/10) sono una trappola.
CHUNK = 32

# NOTE storico: DOMAIN_BIGRAMS e SYNONYRS (italiani) erano i seed della prima
# versione. Sono stati rimossi: non producevano mutazioni (le frasi non compaiono
# negli stati, che sono in inglese) e non erano usati da nessun operatore.
# La superficie lessica è stata poi esplorata sul serio in E12b
# (e12_rare_syn.py), con WordNet e POS corretti.

# Sostituzioni "parassite": parole da ufficio che il modello ha visto e che spostano
# l'attenzione. È il termine più economico di perturbazione: nessun contenuto nuovo.
# Sono in inglese perché typed-decisions è in inglese: un vocabolario italiano
# non avrebbe toccato una sola parola degli stati reali (misurato: 0 occorrenze).

# Sostituzioni lessicali derivate dagli **stati reali** di typed-decisions, non a caso:
# il vocabolario del dataset è inglese, quindi gli operatori devono esserlo.
# Vanno scambiate solo parole che compaiono nei 400 stati del test (verificato a runtime
# da build_vocabulary), altrimenti il GA non avrebbe nulla su cui lavorare.

VARIANTS = {
    # nome -> pesi (flip, confidenza sul torto, KL sullo shift)
    "V0": (1.0, 0.0, 0.0),
    "V1": (1.0, 1.0, 0.0),
    "V2": (1.0, 1.0, 1.0),
}

# I tre termini non sono fra loro commensurabili: `n_flip` è un conteggio (0..5 su
# typed-decisions), `conf` una probabilita' in [0,1], `kl` una quantità **illimitata**
# in nats. Un candidato con 1 flip e KL=100 batte uno con 5 flip e KL=0.1, quindi
# V1 e V2 non "spingono l'OER" o "lo shift": spingono il termine che domina. Le tre
# varianti isolano il contributo dei termini solo se pesi dello stesso ordine di grandezza.
# Non corretto qui perche' E12 ed E12b hanno gia' prodotto i loro numeri con questa
# formula e cambiarla li renderebbe non riproducibili dalle note: se si decide di
# normalizzare il KL (per esempio su log(k)), i due esperimenti vanno rilanciati e le
# note riscritte. Dichiarato qui perche' un lettore che vede "V2 = flip+conf+kl" deve
# sapere che il confronto fra le varianti non è quello che il nome suggerisce.
_FITNESS_TERMS_ARE_NOT_COMMENSURABLE = True


@dataclass
class Individual:
    """One candidate state and everything needed to score it."""

    state: str
    lineage: list[str] = field(default_factory=list)

    def clone(self) -> "Individual":
        return Individual(state=self.state, lineage=list(self.lineage))


# ------------------------------------------------------------------ operators
# ---------------------------------------------------------------------- filters
def build_nlp():
    """spaCy IT. Entità e numeri non si toccano: determinano il gold."""
    import spacy

    nlp = spacy.load("it_core_news_md")
    # Il NER base su italiano non copre date e importi (solo PER/LOC/ORG/MISC):
    # senza questi vincoli il GA puo' cambiare il gold senza che se ne accorga.
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    # L'EntityRuler richiede che un importo e una data siano ancorati a una parola
    # chiave: senza l'ancoraggio ogni numero diventa MONEY, e la guardia che
    # protegge il gold diventa indiscriminata (misurato: '4471' e la data venivano
    # entrambe etichettate MONEY, quindi ogni candidata veniva scartata).
    ruler.add_patterns([
        {"label": "MONEY", "pattern": [
            {"LOWER": {"IN": ["euro", "eur", "€", " dollari", "dollaro", "$"]}},
            {"IS_DIGIT": True},
        ]},
        {"label": "MONEY", "pattern": [
            {"IS_DIGIT": True},
            {"LOWER": {"IN": ["euro", "eur", "€", " dollari", "dollaro", "$"]}},
        ]},
        {"label": "MONEY", "pattern": [{"TEXT": {"REGEX": r"^\d{1,3}(\.\d{3})+(,\d{2})?$"}}]},
        {"label": "PERCENT", "pattern": [{"TEXT": {"REGEX": r"^\d{1,3}(\.\d+)?\s*%$"}}]},
        {"label": "DATE", "pattern": [{"TEXT": {"REGEX": r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$"}}]},
    ])
    return nlp


# Etichettaggi di cui fidarsi: sono quelli espliciti dall'EntityRuler (valori,
# date, percentuali) piu' le entita' nominate. MISC e i tag generici restano
# fuori: su testo tecnico il NER li mette dappertutto e il filtro diventerebbe
# indiscriminato (misurato in protected_tokens).
ENTITY_LABELS = frozenset(
    {"MONEY", "DATE", "PERCENT", "PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT"}
)


def protected_tokens(doc) -> set[int]:
    """Token indices the GA must not touch.

    The gold on typed-decisions is carried by numbers, identifiers and amounts
    (invoice codes, `$50`, `duration_s`, dates), so those are what must be frozen.
    Plain english words are NOT protected: the NER model on this technical text
    marks most natural-language spans as MISC ("Do not exceed", "Rotate the
    expired TLS certificate"), which would freeze exactly the words the GA has to
    work on — measured, 1 of 20 mutations survived that filter.
    """
    out: set[int] = set()
    for tok in doc:
        text = tok.text
        # numeri, importi, percentuali, date, identificatori, path e URL
        if re.search(r"\d", text):
            out.add(tok.i)
        elif any(ch in text for ch in ("$", "€", "%", "@", "/", "_", "-")) and len(text) > 1:
            out.add(tok.i)
    return out


def passes_constraints(nlp, original: str, candidate: str) -> tuple[bool, str]:
    """NER preservato e drift under threshold. Returns (ok, reason)."""
    # I numeri per primi: sono il gold su typed-decisions, e il controllo è esplicito.
    if re.findall(r"\d+(?:[.,]\d+)*", original) != re.findall(r"\d+(?:[.,]\d+)*", candidate):
        return False, "numeri cambiati"
    # Token protetti (numeri, importi, identificatori): devono essere identici.
    doc_o, doc_c = nlp(original), nlp(candidate)
    prot_o = [doc_o[i].text for i in sorted(protected_tokens(doc_o))]
    prot_c = [doc_c[i].text for i in sorted(protected_tokens(doc_c))]
    if prot_o != prot_c:
        return False, "token protetti cambiati"
    # Entita' nominate (MONEY, DATE, PERCENT, PERSON, ORG, ...): il gold può
    # dipenderne anche quando non c'è un numero. `doc.ents` era costruito ma
    # non confrontato: senza questo un candidato poteva spostare un'entita'
    # senza essere scartato. Solo gli etichettaggi espliciti, perche' il NER
    # base sbaglia sul testo tecnico (vedi protected_tokens).
    ents_o = sorted((e.text, e.label_) for e in doc_o.ents if e.label_ in ENTITY_LABELS)
    ents_c = sorted((e.text, e.label_) for e in doc_c.ents if e.label_ in ENTITY_LABELS)
    if ents_o != ents_c:
        return False, "entita' cambiate"
    wo, wc = original.lower().split(), candidate.lower().split()
    if not wo or not wc:
        return False, "stato vuoto"
    # Jaccard come drift proxy: nessun embedding, deterministico e gratis
    so, sc = set(wo), set(wc)
    jac = len(so & sc) / len(so | sc)
    if jac < 0.5:
        return False, f"drift {jac:.2f}"
    return True, "ok"


# --------------------------------------------------------------------- fitness
def kl(clean_probs: dict, adv_probs: dict) -> float:
    """KL(clean || adv) over the gold label, the common support only."""
    p, q = clean_probs, adv_probs
    keys = set(p) & set(q)
    if not keys:
        return 0.0
    total = 0.0
    for k in keys:
        pk, qk = p[k], q[k]
        if pk > 0 and qk > 0:
            total += pk * math.log(pk / qk)
    return total


def score_population(
    engine,
    cases: list[Case],
    population: list[Individual],
    clean_by_id: dict,
    weights: tuple[float, float, float],
) -> list[dict]:
    """Fitness of every individual, in one batched pass (one forward pass per generation).

    The records are the only input: this is what L1 black-box means. The clean
    predictions are the *measured* ones, never the gold, so a mutation that moves
    the gold is not rewarded.
    """
    w_flip, w_conf, w_kl = weights
    # Tutti i candidati in una sola passata: `evaluate` raggruppa per firma delle
    # domande, e qui la firma è identica, quindi una generazione costa una
    # forward pass e non una per candidato.
    adversarial = [
        Case(id=f"{c.id}#{i}", state=ind.state, questions=c.questions, gold=c.gold)
        for i, (c, ind) in enumerate(zip(cases, population))
    ]
    result = evaluate(engine, adversarial, batch_size=engine.batch_size, chunk=len(adversarial))
    scored = []
    for i, (case, ind) in enumerate(zip(cases, population)):
        rec = result.records[i]
        clean = clean_by_id[case.id]
        flips, confs, kls = 0, [], []
        for qid, gold in case.gold.items():
            if clean["labels"][qid] != gold:
                continue  # gia' sbagliato al pulito: non è ASR
            if rec["labels"][qid] != gold:
                flips += 1
                confs.append(rec["confidence"][qid])
                kls.append(kl(clean["probs"][qid], rec["probs"][qid]))
        n_flip = flips
        conf = sum(confs) / len(confs) if confs else 0.0
        shift = sum(kls) / len(kls) if kls else 0.0
        fitness = w_flip * n_flip + w_conf * conf + w_kl * shift
        scored.append(
            {
                "fitness": round(fitness, 4),
                "flips": n_flip,
                "mean_conf_on_wrong": round(conf, 4),
                "kl_shift": round(shift, 4),
                "individual": ind,
            }
        )
    return scored


