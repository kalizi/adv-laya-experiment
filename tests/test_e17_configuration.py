"""E17: il verdetto e' calcolato sui numeri, e i casi da dichiarare sono tre."""

from layaadv.experiments.e17_configuration import GRID, _verdict, load_cfgs


def row(config: str, oer_adv: float, oer_clean: float = 0.02, asr: float = 0.9, **kw):
    base = {
        "config": config,
        "oer_adv": {"0.75": oer_adv},
        "oer_clean": {"0.75": oer_clean},
        "asr": asr,
        "batching_is_inert": False,
        "requested_fast": False,
        "fast_engaged": False,
    }
    base.update(kw)
    return base


def test_verdetto_dipende_dai_numeri_e_non_dalle_parole():
    """Nessuna configurazione chiude il buco solo se tutte le righe restano sopra soglia."""
    v = _verdict([row("stock", 0.88), row("fast", 0.87)], [16], "massive-it", "prepend", 300)
    assert v["configurazioni_che_chiudono_il_buco"] == []
    assert "nessuna configurazione chiude" in v["verdetto"]

    # una sola riga sotto soglia e' sufficiente a dichiarare che il buco si chiude
    v2 = _verdict([row("stock", 0.88), row("fast", 0.40)], [16], "massive-it", "prepend", 300)
    assert v2["configurazioni_che_chiudono_il_buco"] == ["fast"]
    assert "si chiude" in v2["verdetto"]


def test_verdetto_separa_errore_convinto_e_attacco():
    """Chiudiere il buco e chiudere l'attacco sono due soglie diverse.

    Una configurazione che abbassa l'OER sotto 0.50 mentre l'ASR resta sopra 0.50 ha chiuso il
    guardrail senza aver indebolito l'attacco: va detta come due fatti distinti.
    """
    v = _verdict([row("x", 0.30, asr=0.90)], [16], "massive-it", "prepend", 300)
    assert v["configurazioni_che_chiudono_il_buco"] == ["x"]
    assert v["configurazioni_che_chiudono_l_attacco"] == []


def test_fast_richiesto_e_non_attivato_e_dichiarato():
    """`accelerate()` e' strict=False: un fast che ricade sullo stock va detto, non taciuto."""
    v = _verdict(
        [row("fast", 0.87, requested_fast=True, fast_engaged=False)],
        [16], "massive-it", "prepend", 300,
    )
    assert v["fast_richiesto_e_non_attivato"] == ["fast"]


def test_la_griglia_copre_i_knob_e_un_solo_by_only():
    names = [c["config"] for c in load_cfgs(_Args())]
    assert names == [g[0] for g in GRID]
    # la combinazione fast+compile non e' ridondante: laya dichiara che il fast prevale, e se
    # tornasse stock sarebbe un bug, quindi la cella deve esistere per accorgersene
    assert "fast+compile" in names
    # il batch e' una leva dichiarata: anche dove non agisce, la riga lo dichiara
    assert any(c["batch_size"] == 1 for c in load_cfgs(_Args()))
    solo = load_cfgs(_Args(), ["stock"])
    assert [c["config"] for c in solo] == ["stock"]


class _Args:
    model = None
    device = None
    quantize = None
    variant = None


def _rec(i, pred, gold, conf=0.9):
    return {
        "id": str(i),
        "labels": {"q": pred},
        "gold": {"q": gold},
        "types": {"q": "choice"},
        "confidence": {"q": conf},
    }


def test_paired_analysis_appaia_su_intersezione_e_legge_una_volta():
    """`iter_decisions` e' un generatore: leggerlo due volte dà zero la seconda.

    Il test fallisce (KeyError) se l'implementazione riusa il generatore, cioè esattamente il
    modo in cui l'ho scritto la prima volta.
    """
    from layaadv.experiments.e17_configuration import paired_analysis

    # config A: il caso 1 ribalta, il caso 2 no
    a = ([_rec(1, "a", "a"), _rec(2, "a", "a")], [_rec(1, "b", "a"), _rec(2, "a", "a")])
    # config B: il caso 1 e' gia' sbagliato al pulito, quindi non entra nell'intersezione
    b = ([_rec(1, "b", "a"), _rec(2, "a", "a")], [_rec(1, "b", "a"), _rec(2, "a", "a")])

    pa = paired_analysis({"A": a})
    assert pa["n_common_correct"] == 2, "entrambe le config sono corrette al pulito su 2 casi"
    # il caso 1 ribalta (a->b), il caso 2 resta: 1 ribaltamento su 2
    assert pa["rows"][0]["asr_paired"] == 0.5
    assert pa["rows"][0]["oer_adv_paired"] == 0.5, "conf 0.9 sopra 0.75: l'errore e' convinto"

    # B ha gli stessi id di A ma il caso 1 non e' corretto al pulito: esce dall'intersezione.
    pb = paired_analysis({"A": a, "B": b})
    assert pb["n_common_correct"] == 1, "l'intersezione esclude il caso non corretto in B"
    assert pb["rows"][1]["asr_paired"] == 0.0, "B non ribalta nulla su quell'unico caso comune"

    # due configurazioni sullo stesso dataset, lo stesso caso ribaltato solo in una:
    # e' il caso in cui il confronto appaiato vede la differenza che i denominatori separati
    # avrebbero mescolato
    # gli stessi id nelle due configurazioni: e' un confronto appaiato, quindi lo stesso dataset
    ids = (1, 2, 3, 4)
    a4 = ([_rec(i, "a", "a") for i in ids], [_rec(i, "a", "a") for i in ids])
    a4[1][1] = _rec(1, "b", "a")  # A ribalta il caso 1
    b4 = ([_rec(i, "b" if i == 1 else "a", "a") for i in ids],
          [_rec(i, "b" if i == 1 else "a", "a") for i in ids])
    pb2 = paired_analysis({"A": a4, "B": b4})
    # intersezione = casi 2,3,4 (il 1 non e' corretto in B). A non ribalta nulla li', B nemmeno.
    assert pb2["n_common_correct"] == 3, "il caso 1 e' sbagliato al pulito in B: esce"
    assert pb2["rows"][0]["asr_paired"] == 0.0, "A ha ribaltato proprio il caso escluso"
    assert pb2["escursione_appaiata"] == 0.0


def test_paired_richiede_ci_per_dire_di_piu_piccolo_del_rumore():
    """Il rapporto escursione/CI e' la dichiarazione di incertezza, e deve esserci."""
    from layaadv.experiments.e17_configuration import paired_analysis

    arms = {
        "A": ([_rec(i, "a", "a") for i in range(100)],
              [_rec(i, "a", "a") for i in range(99)] + [_rec(99, "b", "a")]),
        "B": ([_rec(i, "a", "a") for i in range(100)],
              [_rec(i, "a", "a") for i in range(98)] + [_rec(98, "b", "a"), _rec(99, "b", "a")]),
    }
    p = paired_analysis(arms)
    assert p["ci95_ampiezza"] and p["escursione_rispetto_a_ci"] is not None
    assert p["escursione_rispetto_a_ci"] > 1, "su 100 decisioni il rumore domina, e va detto"
