"""Le metriche codificano il PERCHÉ: errori qui rendono false le conclusioni.

Ogni test costruisce i casi a mano: nessun modello, nessun dataset.
"""

from __future__ import annotations

import pytest

from layaadv.metrics.core import (
    PRIMITIVES,
    accuracy,
    aggregate,
    asr,
    brier,
    ece,
    latency_stats,
    margin_at_cost,
    margin_tradeoff,
    mean_conf_wrong,
    oer,
    oer_margin,
    top1_margin,
)


def rec(rid, pred, gold, conf, *, qid="q", qtype="choice", probs=None):
    """Una decisione sintetica (un case con una sola domanda)."""
    if probs is None:
        probs = {pred: conf, "altro": 1.0 - conf}
    return {
        "id": rid,
        "types": {qid: qtype},
        "labels": {qid: pred},
        "confidence": {qid: conf},
        "probs": {qid: probs},
        "gold": {qid: gold},
    }


def test_accuracy_overall_and_per_primitive():
    records = [
        rec("a", "si", "si", 0.9),
        rec("b", "no", "si", 0.8),
        rec("c", "1", "1", 0.7, qid="risk", qtype="score"),
    ]
    assert accuracy(records) == pytest.approx(2 / 3)
    assert accuracy(records, "choice") == 0.5
    assert accuracy(records, "score") == 1.0
    assert accuracy(records, "noul") is None  # nessuna decisione di quel tipo


def test_asr_counts_only_decisions_correct_at_clean():
    """Un attacco non deve 'vincere' su errori preesistenti (metriche-attacco §2)."""
    clean = [rec("a", "si", "si", 0.9), rec("b", "no", "si", 0.6)]  # b già sbagliato
    adv = [rec("a", "no", "si", 0.9), rec("b", "no", "si", 0.6)]  # a ribaltato, b invariato
    rate, denominator = asr(clean, adv)
    assert denominator == 1  # solo a era corretto al pulito
    assert rate == 1.0


def test_asr_undefined_when_nothing_correct_clean():
    clean = [rec("a", "no", "si", 0.9)]
    adv = [rec("a", "sì", "si", 0.9)]
    rate, denominator = asr(clean, adv)
    assert rate is None and denominator == 0


def test_asr_denominator_is_the_attacked_subset_not_the_whole_clean_run():
    """Un attacco parziale non si diluisce: il denominatore e' l'intersezione.

    Prima, con 2 casi attaccati su 10 e tutti e 2 ribaltati, `asr` restituiva 0.2
    invece di 1.0: i 8 non attaccati finivano nel denominatore e il numero
    riprodotto non coincideva con quello dichiarato.
    """
    clean = [rec(str(i), "a", "a", 0.9) for i in range(10)]
    adv = [rec(str(i), "b", "a", 0.9) for i in range(2)]
    rate, denominator = asr(clean, adv, strict=False)
    assert (rate, denominator) == (1.0, 2)


def test_asr_refuses_to_report_a_partial_run_as_a_full_one():
    """Fallo forte invece di restituire un denominatore piu' piccolo in silenzio."""
    clean = [rec(str(i), "a", "a", 0.9) for i in range(10)]
    adv = [rec("99", "b", "a", 0.9)]
    with pytest.raises(ValueError, match="non sono nella run avversaria"):
        asr(clean, adv)
    # dichiarato: nessuna decisione attaccata => nessun ASR rivendicabile
    assert asr(clean, adv, strict=False) == (None, 0)


def test_asr_ignores_adv_only_decisions():
    """Decisioni solo nella run avversaria non entrano nel denominatore."""
    clean = [rec("a", "a", "a", 0.9)]
    adv = [rec("a", "a", "a", 0.9), rec("b", "b", "b", 0.9)]
    assert asr(clean, adv) == (0.0, 1)


def test_oer_needs_wrong_and_confident():
    records = [
        rec("a", "no", "si", 0.80),  # torto e confidente -> conta
        rec("b", "no", "si", 0.50),  # torto ma insicuro -> no
        rec("c", "si", "si", 0.99),  # giusto e confidente -> no
    ]
    assert oer(records, tau=0.75) == pytest.approx(1 / 3)
    assert oer(records, tau=0.85) == 0.0
    assert mean_conf_wrong(records) == pytest.approx(0.65)


def test_ece_zero_when_confidence_matches_accuracy():
    # 10 decisioni tutte con conf 0.5, metà corrette -> perfettamente calibrata
    records = [rec(str(i), "x" if i % 2 else "y", "x", 0.5) for i in range(10)]
    assert ece(records) == pytest.approx(0.0)


def test_the_same_threshold_means_different_things_per_primitive():
    """PERCHÉ esiste `oer_margin`: la soglia 0.75 su `confidence` non è una soglia.

    `record["confidence"]` cambia definizione con `k`: su `k=2` i backend ci mettono
    `max(p, 1-p)`, su `k>2` l'entropia normalizzata `1 - H(p)/log(k)`. Le due numeriche
    non coincidono, quindi un unico `tau` non è un unico giudizio.

    Qui la stessa forma di distribuzione (0.9 sull'opzione vinta) viene valutata come
    `noul` e come `choice`: la `noul` passa la soglia, il `choice` no, perché la sua
    entropia normalizzata è 0.6485 e non 0.9. Sul margin le due passano entrambe: il
    margin è 0.80 e 0.83 ed è la grandezza che descrive davvero la decisione.
    """
    probs2 = {"si": 0.9, "no": 0.1}
    probs3 = {"si": 0.9, "no": 0.07, "forse": 0.03}
    noul = [rec("n", "si", "no", 0.9, qtype="noul", probs=probs2)]
    choice = [rec("c", "si", "no", 0.6485, qtype="choice", probs=probs3)]

    assert oer(noul, tau=0.75) == pytest.approx(1.0)
    assert oer(choice, tau=0.75) == pytest.approx(0.0)
    # stessa soglia, due risposte opposte, sullo stesso campo
    assert oer_margin(noul, tau=0.75) == pytest.approx(1.0)
    assert oer_margin(choice, tau=0.75) == pytest.approx(1.0)


def test_margin_threshold_is_a_different_operating_point_not_a_rescaling():
    """Sostituire la metrica senza rifare la soglia è la stessa classe di bug.

    `confidence = 0.8` e `margin = 0.8` non descrivono la stessa decisione: il primo su
    un `choice` a 4 opzioni è una distribuzione quasi piatta, il secondo è un vincitore
    che stacca di 0.8. Perciò `oer_margin@0.75` è **più severo** di `oer@0.75` e non è
    confrontabile con lui numero per numero. Le due colonne viaggiano insieme e ognuna
    dichiara cosa misura; il punto operativo va rifatto sul dataset dove si spende.
    """
    # entropia normalizzata 0.8 su 4 opzioni: distribuzione piatta, nessun margine
    flat = [rec("a", "x", "y", 0.8, probs={"x": 0.4, "y": 0.35, "z": 0.15, "w": 0.1})]
    assert oer(flat, tau=0.75) == pytest.approx(1.0)
    assert oer_margin(flat, tau=0.75) == pytest.approx(0.0)
    assert top1_margin(flat[0]["probs"]["q"]) == pytest.approx(0.05)


def test_top1_margin_is_top1_minus_top2_and_zero_without_a_runner_up():
    assert top1_margin({"a": 0.5, "b": 0.3, "c": 0.2}) == pytest.approx(0.2)
    assert top1_margin({"si": 0.8, "no": 0.2}) == pytest.approx(0.6)
    # k=1: non esiste un secondo candidato, la decisione non è distinguibile.
    # Va detto 0.0 e non "sicura": dichiararlo è il punto.
    assert top1_margin({"solo": 1.0}) == 0.0
    assert top1_margin({}) == 0.0


def test_aggregate_splits_oer_by_primitive_and_carries_the_margin():
    """Senza lo spezzamento la media nasconde il difetto: su E01 pulito `conf>=0.75`
    scatta su 0.2917 dei `noul` e 0.0000 dei `choice`, e un numero aggregato solo non
    dice che esistono due popoli diversi sotto la stessa soglia."""
    records = [
        rec("n", "si", "no", 0.9, qid="needs", qtype="noul",
            probs={"false": 0.1, "true": 0.9}),
        rec("c", "si", "no", 0.6485, qid="action", qtype="choice",
            probs={"si": 0.9, "no": 0.07, "forse": 0.03}),
    ]
    out = aggregate(records)
    assert out["oer"]["0.75"] == pytest.approx(0.5)  # solo la noul
    assert out["oer_by_primitive"]["noul"]["0.75"] == pytest.approx(1.0)
    assert out["oer_by_primitive"]["choice"]["0.75"] == pytest.approx(0.0)
    # sul margin le due primitive dicono la stessa cosa
    assert out["oer_margin"]["0.75"] == pytest.approx(1.0)
    assert out["oer_margin_by_primitive"]["noul"]["0.75"] == pytest.approx(1.0)
    assert out["oer_margin_by_primitive"]["choice"]["0.75"] == pytest.approx(1.0)


def test_ece_detects_overconfidence():
    records = [rec(str(i), "x", "y", 0.95) for i in range(10)]  # tutte torte, conf altissima
    assert ece(records) == pytest.approx(0.95)


def test_brier_zero_for_certain_and_correct():
    records = [rec("a", "si", "si", 1.0, probs={"si": 1.0, "no": 0.0})]
    assert brier(records) == pytest.approx(0.0)


def test_brier_noul_uses_two_class_vector():
    records = [rec("a", "true", "true", 0.8, qtype="noul", probs={"false": 0.2, "true": 0.8})]
    assert brier(records) == pytest.approx(0.08)  # (0.2)^2 + (0.8-1)^2


def test_latency_stats_percentiles():
    stats = latency_stats([10.0, 20.0, 30.0, 40.0, 100.0])
    assert stats["n"] == 5
    assert stats["p50_ms"] == 30.0
    assert stats["p95_ms"] == 100.0
    assert stats["mean_ms"] == 40.0


def test_latency_stats_reports_std_because_run_to_run_varies():
    """Senza std il confronto fra run e' indistinguibile dal rumore (~3%): la
    dispersione serve a dire se una differenza e' reale."""
    tight = latency_stats([10.0, 10.1, 9.9, 10.0])
    spread = latency_stats([10.0, 20.0, 30.0, 40.0])
    assert tight["std_ms"] < 0.2
    assert spread["std_ms"] > 10.0


def test_latency_stats_single_sample_has_zero_std():
    assert latency_stats([7.0])["std_ms"] == 0.0


def test_aggregate_reports_the_standard_block():
    out = aggregate([rec("a", "si", "si", 1.0, probs={"si": 1.0, "no": 0.0})])
    assert out["n_cases"] == 1 and out["n_decisions"] == 1
    assert out["accuracy"] == 1.0
    assert set(out["accuracy_by_primitive"]) == set(PRIMITIVES)
    assert set(out["oer"]) == {"0.75", "0.85"}
    assert out["ece"] == 0.0


def test_ece_clamps_an_out_of_range_confidence_into_a_real_bucket():
    """Una confidence fuori [0,1] deve finire nel bucket 0, non nell'ultimo.

    `int(-0.5 * 10) = -5` indicizza la lista **dall'ultimo bucket**, quindi i 10 record
    a conf -0.5 finivano nel bucket 5 invece che nel bucket 0 e l'ECE valeva 0.275
    invece di 0.225. Impossibile con laya oggi (`confidence_from_probs` e' clipata in
    [0,1]): e' una guardia per un backend che emette confidence diverse, non per
    aggiustare un numero.
    """
    good = [rec(str(i), "x" if i % 2 else "y", "x", 0.5) for i in range(10)]
    assert ece(good) == pytest.approx(0.0)

    mixed = [rec(f"a{i}", "x", "y", 0.05) for i in range(10)]
    mixed += [rec(f"b{i}", "x", "y", -0.5) for i in range(10)]
    # bucket 0 unico: conf media -0.225, accuracy 0 => 0.225
    assert ece(mixed) == pytest.approx(0.225)
    # oltre 1: stessa guardia dall'altra parte
    assert ece([rec("a", "x", "y", 1.4), rec("b", "x", "y", 1.4)]) == pytest.approx(1.4)


def test_margin_at_cost_derives_the_threshold_from_the_clean_run():
    """PERCHÉ esiste: con CLM la temperatura è 0.01 e le probabilità saturano.

    Una soglia fissa sul margin sarebbe un numero fittato sulla distribuzione di
    laya e applicato a quella di CLM. Qui la soglia viene derivata dal costo sul
    traffico pulito, che è la moneta che E16 ha già dichiarato spendibile.
    """
    # 10 decisioni corrette con margin 0.0..0.9. Budget 0.2 = ne puo' bloccare 2.
    records = [
        rec(str(i), "x", "x", 0.5, probs={"x": 0.5 + i / 20, "y": 0.5 - i / 20})
        for i in range(10)
    ]
    out = margin_at_cost(records, budget=0.2)
    assert out["n_clean_correct"] == 10
    assert out["n_blocked"] <= 2, "il budget e' un tetto, non un obiettivo"
    assert out["false_block_rate"] <= 0.2
    # la soglia deve stare dove il margin supera il blocco consentito
    assert 0.0 < out["tau"] <= 1.0


def test_margin_at_cost_refuses_an_empty_correct_population():
    """Un denominatore vuoto non è uno zero da interpretare: è un modello che non
    risolve il compito, e tarare una soglia su niente produrrebbe un numero con
    aria addosso."""
    records = [rec("a", "x", "y", 0.5, probs={"x": 0.6, "y": 0.4})]
    with pytest.raises(ValueError, match="nessuna decisione corretta"):
        margin_at_cost(records, budget=0.01)


def test_margin_at_cost_is_monotone_in_the_budget():
    records = [
        rec(str(i), "x", "x", 0.5, probs={"x": 0.5 + i / 100, "y": 0.5 - i / 100})
        for i in range(20)
    ]
    tight = margin_at_cost(records, budget=0.05)["tau"]
    loose = margin_at_cost(records, budget=0.50)["tau"]
    assert tight >= loose, "piu' budget permette di bloccare di piu', quindi soglia piu' bassa"


def test_margin_at_cost_flags_a_budget_finer_than_the_data():
    """Con 4 decisioni corrette e budget 1% nessuna soglia blocca zero risposte buone
    senza essere più stretta dell'intera distribuzione. Va detto: è un punto
    operativo estremo, non una misura di precisione, e un lettore che lo prendesse
    per tarato leggerebbe il numero successivo come se fosse tarato."""
    records = [
        rec(str(i), "x", "x", 0.5, probs={"x": 0.6, "y": 0.4 + i / 100}) for i in range(4)
    ]
    out = margin_at_cost(records, budget=0.01)
    assert out["budget_binds"] is False
    assert out["n_blocked"] == 0
    assert out["tau"] > out["max_margin"], "la soglia deve stare sopra il max pulito"


def test_margin_tradeoff_reports_cost_and_benefit_together():
    """Una soglia da sola e' una tesi. Il costo sul traffico pulito e il beneficio
    sull'attaccato vanno letti insieme, e qui il caso interessante e' quello in cui
    la soglia stretta **costa di piu' e non compra nulla in piu'**.

    Il pulito e' metà a margin 0.60 e metà a 0.99; gli errori indotti sono tutti a
    0.99. Quindi a tau=0.5 si bloccano tutte le risposte buone e tutti e 5 gli errori
    indotti; a tau=0.99 si blocca metà del pulito e gli stessi 5 errori. Il
    `benefit_over_clean` è 0.25 in entrambi i punti: stringere la soglia qui è
    costo puro.
    """
    mid = [rec(f"m{i}", "x", "x", 0.5, probs={"x": 0.80, "y": 0.20}) for i in range(5)]
    top = [rec(f"t{i}", "x", "x", 0.5, probs={"x": 0.995, "y": 0.005}) for i in range(5)]
    clean = mid + top
    induced = [
        rec(f"a{i}", "y", "x", 0.5, probs={"x": 0.005, "y": 0.995}) for i in range(5)
    ]
    out = margin_tradeoff(clean, clean + induced, taus=(0.5, 0.99))
    loose, tight = out["rows"]

    assert loose["false_block_rate_clean"] == pytest.approx(1.0)
    assert tight["false_block_rate_clean"] == pytest.approx(0.5), "meta' del pulito ha margin 0.99"
    assert loose["oer_adv"] == pytest.approx(5 / 15)
    assert tight["oer_adv"] == pytest.approx(5 / 15), "gli stessi errori, gia' sopra 0.5"
    assert tight["benefit_over_clean"] == pytest.approx(5 / 15)
    assert loose["benefit_over_clean"] == pytest.approx(5 / 15), "stringere non ha comprato nulla"
