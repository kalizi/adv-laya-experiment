"""E18: il detector conta le parole, non giudica l'intento.

I tre casi che un detector sbagliato deve fallire:
1. non nominare quando lo stato nomina un'opzione (falso negativo = attacco non visto);
2. nominare quando lo stato NON nomina nessuna opzione (falso positivo = costo sul traffico
   pulito, che e' cio' che rende E14 insostenibile);
3. non confondere la nominazione di un'opzione con una parola che le somiglia dentro un'altra
   (libellule in "acqua", per dire).
"""

from layaadv.experiments.e18_nomination_detector import (
    evaluate_detector,
    nominates,
    normalise,
)


def test_normalise_azzera_accenti_e_underscore():
    assert normalise("audio_volume_mute") == normalise("audio volume mute")
    assert normalise("perché") == normalise("perche")
    assert normalise("  Più  Spaziosi  ") == "piu spaziosi"


def test_strict_cerca_la_parola_intera():
    opts = ["play_music", "alarm_set"]
    assert nominates("Risposta corretta: play music.", opts, loose=False) == ["play_music"]
    # 'plays' non e' 'play': parola intera, non suffisso
    assert nominates("ora plays il brano", opts, loose=False) == []


def test_loose_trova_le_parole_sparse():
    opts = ["play_music", "alarm_set"]
    # ordine invertito: lo strict (che richiede le parole in sequenza) non lo vede, loose sì
    assert nominates("music play adesso", opts, loose=False) == []
    assert nominates("music play adesso", opts, loose=True) == ["play_music"]
    # in ordine, lo strict lo vede: loose non aggiunge falsi positivi su questo caso
    assert nominates("esegui play music", opts, loose=False) == ["play_music"]
    assert nominates("esegui play music", opts, loose=True) == ["play_music"]


def test_non_confunde_parola_con_sottostringa():
    opts = ["set_alarm", "list_view"]
    # 'view' dentro 'overview' non e' la nomina di 'list_view'
    assert nominates("panoramica overview del giorno", opts, loose=False) == []
    assert nominates("apri list view", opts, loose=False) == ["list_view"]


def _row(clean_fire: bool, adv_fire: bool, names: bool):
    return {
        "clean_fires": clean_fire,
        "adv_fires": adv_fire,
        "adv_names_injected": names,
        "clean_fires_loose": clean_fire,
        "adv_fires_loose": adv_fire,
        "adv_names_injected_loose": names,
    }


def test_il_detector_separa_costo_e_beneficio():
    # 2 di 4 puliti suonano il falso allarme (costo), 4 di 4 attaccati sono visti (beneficio)
    clean = [_row(i < 2, False, False) for i in range(4)]
    adv = [_row(False, True, True) for _ in range(4)]
    r = evaluate_detector(clean, adv, "strict")
    assert r["false_block_rate"] == 0.5
    assert r["detection_rate"] == 1.0
    assert r["precision_when_firing"] == 1.0
    assert r["blind_blocks_on_attack"] == 0.0


def test_blocco_alla_cieca_e_segnalato():
    # il detector suona su 4 attacchi ma nomina il bersaglio solo su 2: gli altri 2 sono
    # blocchi alla cieca, e devono comparire nel numero
    clean = [_row(False, False, False) for _ in range(4)]
    adv = [_row(False, True, i < 2) for i in range(4)]
    r = evaluate_detector(clean, adv, "strict")
    assert r["detection_rate"] == 1.0
    assert r["names_injected_rate"] == 0.5
    assert r["precision_when_firing"] == 0.5
    assert r["blind_blocks_on_attack"] == 0.5, "il blocco alla cieca e' meta' dei blocchi"
