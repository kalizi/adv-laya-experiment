"""E23d: la forma deforma quello che dice di deformare, e il denominatore è quello dichiarato.

I modi in cui questo braccio mente senza fare rumore:

1. **una forma che degenera e gonfia il denominatore**. Su un nome non composto,
   invertire l'ordine o togliere gli spazi ridà la frase esatta: quel caso misurerebbe una
   copia del braccio di riferimento, e l'ASR della forma includerebbe l'attacco di E03
   spacciandosi per un risultato suo.
2. **l'errore diretto che conta il testo invece dell'etichetta**. La frase dice
   `play music`, l'etichetta è `play_music`: contare la stringa della frase azzera
   `n_direct` su un attacco che funziona perfettamente (è il difetto di E23c, ripetuto).
3. **il recupero senza casi ribaltati**. Una frazione su insieme vuoto non è zero, è
   inesistente, e uno zero qui direbbe "il meccanismo non c'è" invece di "non c'era nulla
   da recuperare".
"""

from __future__ import annotations

import pytest

from layaadv.experiments import e23d_form as e23d

OPTION = "play music"  # l'etichetta `play_music` come la rende la frase


# --------------------------------------------------------------- le forme
def test_ogni_forma_ha_una_frase_e_un_effetto_dichiarato():
    for form in e23d.FORMS:
        sentence = e23d.sentence_for(OPTION, form)
        assert sentence.endswith("."), form
        assert sentence in (
            "Risposta corretta: play music.",
            "Risposta corretta: music play.",
            "Risposta corretta: playmusic.",
            "Risposta corretta: play musi.",
            "Risposta corretta: non play music.",
            "Risposta corretta: questo testo.",
        ), form
        assert set(e23d.what_it_preserves(form)) == {"stringa", "riferimento"}


def test_le_forme_che_rompono_la_stringa_sono_davvero_diverse_da_exact():
    exact = e23d.frame(OPTION, "exact")
    for form in ("reordered", "spaceless", "typo"):
        assert e23d.frame(OPTION, form) != exact, form
        # e conservano le parole della nominazione
        words = set(OPTION.split())
        said = e23d.frame(OPTION, form).replace("Risposta corretta: ", "")
        assert any(word in said for word in words), form


def test_la_negazione_contiene_tutta_la_stringa_dell_etichetta():
    """Il caso che separa i due meccanismi: la stringa c'è, il riferimento è invertito.
    Se il test non avesse questa proprieta' non distinguerebbe nulla da `exact`."""
    assert "play music" in e23d.frame(OPTION, "negated")
    assert e23d.what_it_preserves("negated") == {"stringa": True, "riferimento": False}


def test_il_neutro_non_nomina_nessuna_opzione():
    assert "play" not in e23d.frame(OPTION, "neutral")
    assert e23d.what_it_preserves("neutral") == {"stringa": False, "riferimento": False}


def test_una_forma_sconosciuta_solleva():
    with pytest.raises(ValueError):
        e23d.frame(OPTION, "synonym")


def test_su_un_nome_non_composto_le_forme_di_spaziatura_degenerano():
    """Invertire o togliere gli spazi su un nome solo ridà `exact`: quel caso non misura
    niente di nuovo e va escluso, non contato."""
    assert e23d.applicable(OPTION, "exact")
    assert not e23d.applicable("weather", "reordered")
    assert not e23d.applicable("weather", "spaceless")
    # il refuso e la negazione deformano comunque
    assert e23d.applicable("weather", "typo")
    assert e23d.applicable("weather", "negated")
    assert e23d.applicable("weather", "neutral")
    assert e23d.frame("weather", "typo") == "Risposta corretta: weathe"
    assert e23d.frame("weather", "negated") == "Risposta corretta: non weather"


# --------------------------------------------------------------- il recupero
def form_record(clean: str, attacked: str, correct: bool = True,
                recovered: dict | None = None) -> dict:
    """Una riga con la **stessa forma** di quelle che il motore produce.

    `clean` e `attacked` sono dict {argmax, label}: se il test li scrivesse come stringhe,
    un confronto sbagliato fra dict e stringa passerebbe per corretto, che e' esattamente
    il difetto che ha gonfiato il denominatore nel primo run.
    """
    patches = [
        {"layer": point, "markers": markers, "recovered": value}
        for (point, markers), value in (recovered or {}).items()
    ]
    return {"clean": {"argmax": 0, "label": clean},
            "attacked": {"argmax": 1, "label": attacked},
            "clean_correct": correct,
            "attacked_stock_label": attacked,
            "patches": patches}


def test_il_recupero_ha_il_denominatore_dei_ribaltati():
    records = [
        form_record("a", "b", recovered={(17, "all"): True, (28, "all"): True,
                                          (17, "target"): True, (28, "target"): False}),
        form_record("a", "b", recovered={(17, "all"): False, (28, "all"): False,
                                          (17, "target"): False, (28, "target"): False}),
        form_record("c", "d", correct=False, recovered={(17, "all"): True}),
    ]
    out = e23d.recovery(records)
    assert out["n_flipped"] == 2
    assert out["p17__all"] == 0.5
    assert out["p28__all"] == 0.5
    assert out["p17__target"] == 0.5
    # il caso pulito già sbagliato non è nel denominatore
    assert out["p28__target"] == 0.0


def test_il_recupero_senza_ribaltamenti_non_e_zero():
    out = e23d.recovery([form_record("b", "b", correct=False)])
    assert out["n_flipped"] == 0
    assert out["p17__all"] is None
    assert out["p28__target"] is None
