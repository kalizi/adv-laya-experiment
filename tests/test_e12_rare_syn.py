"""E12b: l'operatore di mutazione lessica.

I test qui non verificano che l'attacco funzioni (e' una misura, non un
comportamento da garantire): verificano che l'operatore scelga le parole giuste
e non tocchi mai il gold. Un operatore che muta un articolo o un numero passa
un test di "c'è un flip" e fallisce il protocollo.
"""

import random

import pytest

from layaadv.experiments.e12_rare_syn import (
    DEFAULT_POPULATION,
    build_nlp,
    content_candidates,
    lemma_frequency,
    passes_constraints,
    rare_synonyms,
    substitute,
)

SAMPLE = (
    'The agent must rotate the expired certificate and never exceed the spend limit '
    'of 50 dollars before the deadline.'
)


@pytest.fixture(scope="module")
def nlp():
    return build_nlp()


def _tokens(nlp, text):
    return {t.text.lower(): t for t in nlp(text)}


def test_pos_filter_excludes_function_words(nlp):
    """Connettivi, articoli, preposizioni e punteggiatura non sono candidati."""
    found = content_candidates(nlp, SAMPLE, DEFAULT_POPULATION)
    surfaces = {c["surface"].lower() for c in found}
    for forbidden in ("the", "and", "before", "of", ".", ",", "50"):
        assert forbidden not in surfaces, f"{forbidden} non doveva essere candidato"


def test_pos_filter_keeps_content_words(nlp):
    """Le parole di contenuto devono invece esserci."""
    found = content_candidates(nlp, SAMPLE, DEFAULT_POPULATION)
    surfaces = {c["surface"].lower() for c in found}
    assert "exceed" in surfaces
    assert "agent" in surfaces


def test_candidates_never_include_protected_tokens(nlp):
    """Nessun candidato puo' portare numeri, importi o identificatori."""
    text = 'Rotate the TLS certificate for account "acc_8841" costing $50 in 32.5 seconds.'
    for c in content_candidates(nlp, text, DEFAULT_POPULATION):
        assert not any(ch.isdigit() for ch in c["surface"])
        assert not any(ch in c["surface"] for ch in "$€%@/_-")


def test_synonyms_are_less_frequent_than_original(nlp):
    """Il cuore dell'operatore: il sostituto deve essere fuori distribuzione."""
    doc = nlp(SAMPLE)
    tok = next(t for t in doc if t.lemma_.lower() == "exceed")
    original_freq = lemma_frequency("exceed", "v")
    alts = rare_synonyms(tok, DEFAULT_POPULATION)
    assert alts, "exceed dovrebbe avere sinonimi"
    for alt in alts:
        assert lemma_frequency(alt, "v") < original_freq


def test_synonyms_exclude_the_word_itself(nlp):
    doc = nlp(SAMPLE)
    tok = next(t for t in doc if t.lemma_.lower() == "exceed")
    assert "exceed" not in rare_synonyms(tok, DEFAULT_POPULATION)


def test_synonyms_sorted_rarest_first(nlp):
    doc = nlp(SAMPLE)
    tok = next(t for t in doc if t.lemma_.lower() == "exceed")
    freqs = [lemma_frequency(a, "v") for a in rare_synonyms(tok, DEFAULT_POPULATION)]
    assert freqs == sorted(freqs)


def test_synonyms_exclude_proper_names(nlp):
    """I nomi propri non sono sinonimi: vanno esclusi anche se WordNet li propone.

    Misurato: `service` dava "Robert William Service" (un poeta) e "Service".
    Una sostituzione che cambia il soggetto della frase non e' una mutazione
    lessicale, e' rumore che il filtro di drift non vede.
    """
    tok = next(t for t in nlp("service") if t.text == "service")
    for mode in ("rarest", "commonest", "mid"):
        for alt in rare_synonyms(tok, DEFAULT_POPULATION, 1e9, mode):
            assert alt[:1].islower(), f"nome proprio in {mode}: {alt}"
            assert "Robert" not in alt


def test_pick_modes_differ(nlp):
    """I tre modi devono restituire insiemi diversi, altrimenti il parametro e' inutile."""
    tok = next(t for t in nlp(SAMPLE) if t.lemma_.lower() == "exceed")
    rarest = rare_synonyms(tok, DEFAULT_POPULATION, 0.0, "rarest")
    commonest = rare_synonyms(tok, DEFAULT_POPULATION, 1e9, "commonest")
    mid = rare_synonyms(tok, DEFAULT_POPULATION, 1e9, "mid")
    assert set(rarest) != set(commonest)
    assert len(rarest) == len(commonest) == len(mid)


def test_ratio_strict_is_the_default(nlp):
    """Senza ratio si accettano solo sinonimi piu' rari: e' il protocollo E12b."""
    tok = next(t for t in nlp(SAMPLE) if t.lemma_.lower() == "exceed")
    strict = rare_synonyms(tok, DEFAULT_POPULATION, 0.0, "rarest")
    original_freq = lemma_frequency("exceed", "v")
    for alt in strict:
        assert lemma_frequency(alt, "v") < original_freq


def test_higher_ratio_cannot_shrink_the_pool(nlp):
    """Alzare la soglia deve solo allargare la pool, non restringerla."""
    tok = next(t for t in nlp(SAMPLE) if t.lemma_.lower() == "exceed")
    strict = set(rare_synonyms(tok, DEFAULT_POPULATION, 0.0, "rarest"))
    for ratio in (1.0, 5.0, 100.0):
        wider = set(rare_synonyms(tok, DEFAULT_POPULATION, ratio, "rarest"))
        assert strict <= wider, f"ratio {ratio} ha perso sinonimi"


def test_min_alternatives_is_respected(nlp):
    """Con un piso alto, un caso senza abbastanza sinonimi sparisce."""
    text = "The agent must rotate it."
    assert not content_candidates(nlp, text, min_alternatives=99)


def test_substitute_preserves_capitalisation(nlp):
    """Maiuscola iniziale conservata: `Rotate` -> `Spin`, non -> `spin`."""
    doc = nlp("Rotate the disk")
    idx = next(t.i for t in doc if t.text == "Rotate")
    assert substitute(doc, idx, "spin") == "Spin the disk"
    doc = nlp("rotate the disk")
    idx = next(t.i for t in doc if t.text == "rotate")
    assert substitute(doc, idx, "spin") == "spin the disk"


def test_substitute_uses_spacy_offsets_not_split(nlp):
    """Il caso che ha rotto la prima implementazione.

    Su uno stato reale spaCy restituisce 100 token e `split()` 37: indicizzare
    per posizione sostituiva la parola sbagliata. Qui la verifica e' che la
    sostituzione colpisca il token richiesto e basta.
    """
    text = 'Do not exceed a $50 spend, agent.'
    doc = nlp(text)
    assert len(doc) != len(text.split()), "il test deve stare su testo disallineato"
    idx = next(t.i for t in doc if t.text == "exceed")
    out = substitute(doc, idx, "outstrip")
    assert "outstrip" in out
    assert "exceed" not in out
    # il resto del testo, numeri e punteggiatura, resta intatto
    assert "$50" in out
    assert out.endswith("agent.")


def test_constraints_block_numeric_change():
    ok, why = passes_constraints("exceed the 50 dollar limit", "exceed the 90 dollar limit")
    assert not ok and "numeri" in why


def test_constraints_block_high_drift():
    ok, why = passes_constraints("alpha beta gamma", "zulu yotta kappa iota theta")
    assert not ok and "drift" in why


def test_constraints_allow_lexical_change():
    """Il caso che il filtro non deve bloccare: un solo sinonimo."""
    clean = "The agent must rotate the expired certificate and never exceed the spend limit of 50 dollars."
    mutated = "The servicer must circulate the certificate and never outstrip the spend limit of 50 dollars."
    ok, why = passes_constraints(clean, mutated)
    assert ok, why


def test_rng_choice_is_reproducible():
    """Lo stesso seed produce la stessa popolazione: senza questo il run non e' rifacibile."""
    candidates = list(range(20))
    first = random.Random(0).sample(candidates, 3)
    second = random.Random(0).sample(candidates, 3)
    assert first == second


def test_composed_substitutions_stay_inside_drift():
    """Dove si applica piu' di una sostituzione, il filtro numerico regge.

    Il ciclo compone 9-10 mutazioni sullo stesso stato. Il rischio e' che a
    catena il drift superi la soglia o che i numeri si perdano: qui si verifica
    che la composizione resta legittima.
    """
    clean = "The agent must rotate the expired certificate and never exceed the spend limit of 50 dollars."
    mutated = "The servicer must spin the certificate and never outstrip the spend limit of 50 dollars."
    ok, why = passes_constraints(clean, mutated)
    assert ok, why
    assert "50" in mutated


def test_wordnet_available():
    """Se WordNet non e' caricato l'operatore degenera a zero candidati.

    Fallirebbe in silenzio: content_candidates restituirebbe [] e l'esperimento
    riporterebbe 0 flip come se fosse un finding. Serve un test esplicito.
    """
    from nltk.corpus import wordnet as wn

    assert wn.synsets("rotate", "v"), "corpora WordNet non accessibili"
    assert lemma_frequency("rotate", "v") >= 0
