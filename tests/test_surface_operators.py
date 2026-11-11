"""Gli operatori di superficie devono cambiare davvero lo stato.

Il motivo di questi test è un episodio preciso. I caratteri a larghezza zero sono stati scritti
come **letterali** nel sorgente, e U+200B è sparito in silenzio durante il trasferimento del
file: la costante è diventata la stringa vuota. L'operatore "attacco invisibile" non toccava
niente e riportava un ASR di 0.027, cioè il numero di un attacco inerte, senza che nulla
dicesse che il codice sotto era un no-op.

La lezione è già nella nota di E12l ("la entry conteneva uno spazio ASCII, e il risultato
sembrava dire che il tokenizer elimina lo ZWSP: non era vero, non era mai stato testato"), quindi
il test che mancava era proprio questo.
"""

from __future__ import annotations

import pytest

from layaadv.experiments import e12_surface as e12

# Lo stato deve avere una virgola **dentro** un valore stringa: è il presupposto di E12i, che
# inserisce l'underscore dove lo spazio segue una virgola di contenuto e non quella che separa
# le proprietà. Su uno stato senza virgole interne quell'operatore non deve fare niente, ed è
# il comportamento giusto.
STATE = '{"agent": "checkpointed", "task": "Rotate, then verify, the certificate"}'


def test_zero_width_constants_are_real_characters():
    """Il codice point atteso è dichiarato: se sparisce, il test dice quale."""
    for name, char, code in (("ZWSP", e12.ZWSP, 0x200B), ("ZWNJ", e12.ZWNJ, 0x200C),
                             ("ZWJ", e12.ZWJ, 0x200D), ("FEFF", e12.FEFF, 0xFEFF)):
        assert len(char) == 1, f"{name} è lunga {len(char)}: {char!r}"
        assert ord(char) == code, f"{name} è U+{ord(char):04X}, atteso U+{code:04X}"


def test_uts39_table_is_oriented_for_the_attack():
    """`data/uts39/confusables.txt` mappa sorgente -> prototipo, quindi va invertito.

    La riga `03B1 ; 0061` dice che alpha e a si confondono. Per sostituire una lettera latina
    serve `latina -> confusabile`, e leggere il file nel verso sbagliato dà una tabella con una
    coppia sola, che produce un operatore silenziosamente inerte.
    """
    table = e12.uts39_table()
    assert len(table) > 20, f"tabella UTS #39 quasi vuota: {len(table)} coppie"
    assert table.get("a") == "α", f"'a' non mappa su alpha: {table.get('a')!r}"
    assert table.get("o") == "ο", f"'o' non mappa su omicron: {table.get('o')!r}"


@pytest.mark.parametrize("name", sorted(e12.REGISTRY))
def test_every_operator_changes_the_state(name):
    """Un operatore che non cambia lo stato riporta ASR ~ 0 e sembra un attacco debole."""
    out = e12.transform_of(name)(STATE)
    assert out != STATE, f"{name} non modifica lo stato: è un no-op"


@pytest.mark.parametrize("name", sorted(e12.REGISTRY))
def test_every_operator_preserves_the_json(name):
    """Lo stato è JSON serializzato: un operatore che lo invalida rompe l'esperimento.

    Non è un dettaglio: `inject_option` e le metriche indicizzano il testo, e uno stato non
    parsabile fa fallire la valutazione in un modo che sembra un bug del modello.
    """
    import json

    out = e12.transform_of(name)(STATE)
    json.loads(out)  # solleva se l'operatore ha rotto le stringhe


@pytest.mark.parametrize("name", sorted(e12.REGISTRY))
def test_every_registered_operator_has_a_transform(name):
    assert name in e12.TRANSFORMS


def test_every_operator_has_a_declared_reason():
    """Una riga senza `why` non spiega a cosa serve: è una riga di numeri senza domanda."""
    for name, spec in e12.REGISTRY.items():
        assert spec.get("why"), f"{name} non dichiara perché esiste"
        assert spec.get("label"), f"{name} non ha etichetta"


def test_the_control_arms_are_themselves_registered():
    """Un `control` che punta a un operatore inesistente rende la coppia inutile."""
    for name, spec in e12.REGISTRY.items():
        control = spec.get("control")
        if control is not None:
            assert control in e12.REGISTRY, f"{name} punta a un controllo inesistente: {control}"


def test_keyboard_neighbours_are_adjacent_keys():
    """La vicinanza è fisica: `a` tocca `q`, `w`, `s`, `z`, non `p` né `l`."""
    nb = e12.build_neighbours()
    assert nb["a"] == ["q", "s", "w", "x", "z"]
    assert "p" not in nb["a"] and "l" not in nb["a"]


def test_trade_off_separates_cost_from_benefit():
    """`trade_off` tiene i due rami separati perche' sotto attacco il tasso di errore e'
    alto: un numero solo fa sembrare piatta una curva che non lo e'.

    Il costo e' la risposta PULITA bloccata, il beneficio e' l'errore ATTACCATO segnalato.
    """
    from layaadv.experiments.e16_calibration_defense import trade_off
    from layaadv.metrics.core import iter_decisions

    def rec(correct: bool, conf: float) -> dict:
        # forma che `iter_decisions` legge davvero: un record per stato, una entry per qid
        gold, pred = ("a", "a") if correct else ("a", "b")
        return {"id": "s1", "labels": {"q": pred}, "gold": {"q": gold},
                "types": {"q": "choice"}, "confidence": {"q": conf}}

    adv = [rec(False, 0.95)] * 8 + [rec(True, 0.10)] * 2
    rows = trade_off(adv, 1.0, 1.0, (0.5, 0.9))
    at_05, at_09 = rows

    # la risposta corretta ha conf 0.10, quindi resta sotto QUALSIASI soglia >= 0.5:
    # costo zero ovunque, e il beneficio dipende solo da quanti errori sono convinti
    for row in rows:
        assert row["false_block_rate"] == 0.0
    assert at_05["confident_error_rate"] == 1.0 and at_05["flagged_wrong"] == 8
    assert at_09["confident_error_rate"] == 1.0 and at_09["flagged_wrong"] == 8

    # il denominatore del beneficio e' il numero di errori, non di decisioni
    assert sum(r["flagged_wrong"] for r in rows if r["tau"] == 0.9) == 8
    assert len(list(iter_decisions(adv))) == 10
