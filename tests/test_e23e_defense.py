"""E23e: il rivelatore misura quello che dice di misurare, e il controllo è quello giusto.

I modi in cui questo braccio mente senza fare rumore:

1. **il controllo dei falsi blocchi è il braccio pulito**. Un caso pulito non ha frasi da
   togliere, quindi il rivelatore non può scattare e quel controllo darebbe 0.0000 di falsi
   blocchi per costruzione, non per merito. Il controllo onesto è il braccio **neutro**:
   stessa cornice, stessa lunghezza, nessuna opzione nominata.
2. **l'ASR residuo senza dire cosa è successo ai casi bloccati**. Calcolarlo sui soli
   sopravvissuti direbbe «l'attacco cala» quando invece è stato nascosto togliendo i casi
   dal denominatore: il residuo pubblica quanti ne sono stati tolti.
3. **la soglia scelta sui casi valutati**. Scegliere la soglia dopo aver visto i numeri è
   allenare il difensore sul test. La curva per soglia è il risultato, e il punto operativo
   si sceglie con una regola dichiarata prima (la soglia più bassa a cui i falsi blocchi
   stanno sotto l'1%).
"""

from __future__ import annotations

import pytest

from layaadv.experiments import e23e_defense as e23e

QID = "intent"
OPTIONS = ("a", "b", "c", "d")


def record(identifier: str, probs: dict, gold: str = "a") -> dict:
    return {"id": identifier, "labels": {QID: max(probs, key=probs.get)},
            "probs": {QID: probs}, "gold": {QID: gold}}


def clean_case(value: float) -> dict:
    return record("1", {"a": value, "b": (1 - value) / 3, "c": (1 - value) / 3,
                         "d": (1 - value) / 3})


def attack_case(lift_on: str) -> dict:
    base = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
    base[lift_on] = 0.9
    rest = (1 - 0.9) / 3
    for key in base:
        if key != lift_on:
            base[key] = rest
    return record("1", base)


# ------------------------------------------------------------------ le statistiche
def test_il_lift_guarda_l_opzione_che_sale_di_piu():
    """Un attacco concentra il guadagno su un'opzione sola: è la firma che il rivelatore
    cerca, e non ha bisogno di sapere quale opzione sia."""
    clean = clean_case(0.25)
    armed = attack_case("c")
    assert e23e.lift(clean, armed) == pytest.approx(0.9 - 0.25, abs=1e-9)


def test_il_lift_ignora_chi_scende():
    """Contare solo chi sale: un'opzione che perde quota non è un segnale d'attacco, e
    mescolare i due versi annacquerebbe la misura."""
    clean = record("1", {"a": 0.9, "b": 0.1, "c": 0.0, "d": 0.0})
    armed = record("1", {"a": 0.1, "b": 0.9, "c": 0.0, "d": 0.0})
    assert e23e.lift(clean, armed) == pytest.approx(0.8, abs=1e-9)


def test_la_concentrazione_separa_guadagno_concentrato_da_sparpagliato():
    """Stessa quantità di movimento, due firme diverse: è la differenza fra un attacco e
    una frase che rimescola il modello."""
    clean = record("1", {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25})
    concentrated = record("1", {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25})
    concentrated["probs"][QID]["c"] += 0.4
    concentrated["probs"][QID]["a"] -= 0.4
    spread = record("1", {"a": 0.45, "b": 0.25, "c": 0.25, "d": 0.05})
    spread["probs"][QID] = {"a": 0.15, "b": 0.35, "c": 0.35, "d": 0.15}
    assert e23e.concentration(clean, concentrated) > e23e.concentration(clean, spread)
    assert e23e.concentration(clean, concentrated) == pytest.approx(0.5, abs=1e-9)


def test_se_nessuna_opzione_guadagna_la_concentrazione_e_zero_non_un_errore():
    """Con due distribuzioni normalizzate qualcuno guadagna sempre; succede solo con le
    probabilita' arrotondate nei record, dove la somma non e' esattamente 1. Li' `max` su
    una lista vuota solleverebbe a run finita: va contato zero, non come un errore."""
    clean = record("1", {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1})
    smaller = record("1", {"a": 0.3, "b": 0.25, "c": 0.2, "d": 0.05})
    assert e23e.concentration(clean, smaller) == 0.0
    assert e23e.lift(clean, smaller) == 0.0  # nessuna opzione guadagna: non c'e' lift


def test_nessun_movimento_non_e_un_guadagno():
    clean = clean_case(0.25)
    armed = record("1", dict(clean["probs"][QID]))
    assert e23e.lift(clean, armed) == 0.0
    assert e23e.concentration(clean, armed) == 0.0


# ---------------------------------------------------------------- la curva
def test_la_curva_copre_tutta_la_griglia_e_le_due_frazioni():
    attack = [0.9, 0.8, 0.1]
    control = [0.2, 0.1, 0.05]
    curve = e23e.trade_off(attack, control)
    assert [row["tau"] for row in curve] == e23e.GRID
    high = next(row for row in curve if row["tau"] == 0.75)
    assert high["detected"] == pytest.approx(2 / 3, abs=1e-4)
    assert high["false_blocks"] == 0.0
    # alla soglia più bassa della griglia tutto scatta, da entrambe le parti
    low = next(row for row in curve if row["tau"] == 0.05)
    assert low["detected"] == 1.0
    assert low["false_blocks"] == 1.0
    # e a metà griglia il controllo è già sceso a un terzo
    middle = next(row for row in curve if row["tau"] == 0.15)
    assert middle["false_blocks"] == pytest.approx(1 / 3, abs=1e-4)


def test_il_punto_operativo_segue_la_regola_dichiarata():
    """La soglia più bassa a cui i falsi blocchi stanno sotto l'1%: non la soglia che
    massimizza la detection, che sarebbe una scelta fatta sui casi valutati."""
    curve = [
        {"tau": 0.05, "detected": 1.0, "false_blocks": 0.20},
        {"tau": 0.10, "detected": 0.98, "false_blocks": 0.02},
        {"tau": 0.15, "detected": 0.90, "false_blocks": 0.005},
        {"tau": 0.20, "detected": 0.80, "false_blocks": 0.0},
    ]
    point = e23e.operating_point(curve)
    assert point["tau"] == 0.15
    assert point["detected"] == 0.90


def test_senza_nessuna_soglia_accettabile_non_c_e_punto_operativo():
    curve = [{"tau": 0.05, "detected": 1.0, "false_blocks": 0.5}]
    assert e23e.operating_point(curve) is None


def test_l_asr_residuo_dice_quanti_casi_ha_tolto():
    """Un ASR sui soli sopravvissuti senza il conteggio dei bloccati direbbe «l'attacco
    cala» quando è stato nascosto."""
    clean = [record(str(i), {"a": 0.5, "b": 0.5}, gold="a") for i in range(4)]
    armed = [record(str(i), {"a": 0.1, "b": 0.9}, gold="a") for i in range(4)]
    out = e23e.residual_asr(clean, armed, [True, True, False, False])
    assert out["n_blocked"] == 2
    assert out["denominatore_residuo"] == 2
    # sui sopravvissuti l'attacco continua a vincere: l'ASR residuo è 1.0, non 0.0
    assert out["asr_residuo"] == 1.0


def test_il_controllo_dei_falsi_blocchi_non_puo_essere_il_pulito():
    """Un caso pulito non ha frasi da togliere: il rivelatore non può scattare, quindi
    quel controllo darebbe zero per costruzione. Il modulo misura il neutro accanto, ed
    è il neutro che finisce nella curva."""
    clean = record("1", {"a": 0.5, "b": 0.5})
    armed = record("1", {"a": 0.5, "b": 0.5})
    # sul pulito la statistica è zero per costruzione, non per merito
    assert e23e.lift(clean, armed) == 0.0
    curve = e23e.trade_off([0.9], [0.2])
    assert next(row for row in curve if row["tau"] == 0.15)["false_blocks"] == 1.0
