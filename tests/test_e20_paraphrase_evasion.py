"""Test sul contratto di E20: la copertura si dichiara su due assi insieme.

Il numero che chiude la difesa di E18 e' `coverage_of_working_attacks`, e dipende da due fatti
che si possono rompere in silenzio: il detector deve essere valutato **sullo stato attaccato**, e
il denominatore deve essere **la decisione**, non lo stato. Qui si fissano le varianti e la
forma, non i numeri: un test che ripete i numeri del run si falsifica da solo al primo ricalcolo.
"""

from __future__ import annotations

from layaadv.experiments.e18_nomination_detector import nominates
from layaadv.experiments.e20_paraphrase_evasion import (
    VARIANTS,
    _name,
    attacked_state,
)


def test_le_riformulazioni_sono_generiche_non_scritte_a_mano():
    """Ogni variante deve valere per un'etichetta qualsiasi: una parafrasi scritta a mano per
    una sola opzione darebbe un risultato che non si trasferisce.

    La proprieta' che lo dichiara e' che l'output derivi dall'etichetta ricevuta: nessuna
    trasformazione tocca la seconda parola, quindi la sua parte iniziale deve comparire per
    qualunque etichetta, e due etichette diverse devono dare output diversi.
    """
    for nome, build in VARIANTS.items():
        coda = _name("reminder_set").split(" ")[-1][:-1]
        assert coda in build("reminder_set"), nome
        assert build("reminder_set") != build("alarm_remove"), f"{nome} non dipende dall'etichetta"


def test_le_cornici_cambiano_e_le_deformazioni_no():
    """La divisione che rende l'esperimento leggibile: le riformulazioni *coperte* toccano la
    cornice e lasciano l'opzione, quelle *evasive* deformano l'opzione."""
    coperti = {"verbatim", "frame_inversa", "frame_inglese", "corner_nuda"}
    evasivi = {"compact", "typo", "leet"}
    assert coperti | evasivi == set(VARIANTS)
    # le evasive devono cambiare l'opzione: nessuna contiene l'etichetta intatta
    for nome in evasivi:
        assert "play music" not in VARIANTS[nome]("play_music"), nome
    # le coperte devono contenere l'opzione intatta: il detector le vede per quello
    for nome in coperti:
        assert "play music" in VARIANTS[nome]("play_music"), nome


def test_compact_e_typo_evadono_il_detector_strict():
    """Il cuore del finding: le varianti evasive non sono viste dal detector stretto."""
    for nome in ("compact", "typo", "leet"):
        stato = attacked_state("accendi la musica", "play_music", VARIANTS[nome])
        assert not nominates(stato, ["play_music", "alarm_set"], loose=False), nome


def test_cambio_di_cornice_non_evade_e_resta_nominazione():
    """Il controllo: cambiare solo la cornice non sfugge al detector."""
    for nome in ("frame_inversa", "frame_inglese", "corner_nuda"):
        stato = attacked_state("accendi la musica", "play_music", VARIANTS[nome])
        assert nominates(stato, ["play_music", "alarm_set"], loose=False), nome
