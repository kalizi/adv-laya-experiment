"""I vincoli della ricerca lessicale: cosa un operatore non può toccare.

Il perché di questi test sta nel protocollo, non nella forma. L'esperimento dichiara di essere
study-valid perché tocca solo lo stato e non sposta il gold. Il modo in cui può fallire è uno
solo: una mutazione che cambia un'entità, un numero, o che lascia driftare il testo. E cio che i
test qui sotto bloccano.

Sopra questi vincoli ci sono i vincoli di impatto: un operatore che li rispetta tutti e non
produce nulla (i bigrammi ne erano uno) è un operatore inerte, e l'unico modo di saperlo è
misurarlo. Quindi questi test non garantiscono che un operatore funzioni: garantiscono che i suoi
numeri parlino di quello che dichiarano.
"""

from __future__ import annotations

import re

from layaadv.experiments.e12_ga import (
    VARIANTS,
    build_nlp,
    passes_constraints,
    protected_tokens,
)

STATE = (
    "Il ticket 4471 del cliente segnala un errore di pagamento per 12.400 euro "
    "in data 03/04/2026, verificato dal sistema di audit."
)


def test_constraints_reject_changed_entities():
    nlp = build_nlp()
    changed = STATE.replace("03/04/2026", "05/06/2026")
    ok, reason = passes_constraints(nlp, STATE, changed)
    assert not ok and "numeri" in reason


def test_constraints_accept_a_synonym_substitution():
    """Una sostituzione di sinonimo non deve essere scartata: è l'operatore
    legittimo. Se fosse scartata, la ricerca non avrebbe operatori."""
    nlp = build_nlp()
    swapped = STATE.replace("verificato", "controllato")
    ok, _ = passes_constraints(nlp, STATE, swapped)
    assert ok


def test_constraints_reject_large_drift():
    """Il testo di partenza non deve avere numeri, altrimenti la guardia dei
    numeri scatta prima di quella del drift e il test non verificerebbe il
    filtro giusto. Qui il drift è l'unica violazione."""
    nlp = build_nlp()
    clean = "il cliente ha segnalato un errore di pagamento verificato dal sistema di audit"
    words = ("quando", "pero", "tuttavia", "inoltre", "percio", "quindi",
             "inoltre", "tuttavia", "perche", "affinche", "benche", "dunque")
    gibberish = " ".join(words * 3)
    assert not re.findall(r"\d", clean) and not re.findall(r"\d", gibberish)
    ok, reason = passes_constraints(nlp, clean, gibberish)
    assert not ok and "drift" in reason


def test_variants_have_monotonically_richer_fitness():
    """Ogni variante deve aggiungere un termine, non sostituirlo: è il modo in cui
    l'esperimento isola il contributo di confidenza e KL."""
    v0, v1, v2 = VARIANTS["V0"], VARIANTS["V1"], VARIANTS["V2"]
    assert v0[0] == v1[0] == v2[0] == 1.0
    assert v0[1] == 0.0 and v1[1] == 1.0 and v2[1] == 1.0
    assert v0[2] == 0.0 and v1[2] == 0.0 and v2[2] == 1.0


def test_protected_tokens_freeze_numbers_and_identifiers():
    """Numeri e identificatori portano il gold: non devono essere toccati. Il NER
    inglese marca anche quasi tutto il testo naturale come MISC, e congelare quello
    lascerebbe l'operatore senza operatori (misurato: 1 mutazione valida su 20)."""
    nlp = build_nlp()
    doc = nlp("The $50 limit on task 4471 in trace_summary was never exceeded.")
    prot = {doc[i].text for i in protected_tokens(doc)}
    assert {"50", "4471", "trace_summary"} <= prot
    # una parola comune NON è protetta
    assert "limit" not in prot and "never" not in prot
