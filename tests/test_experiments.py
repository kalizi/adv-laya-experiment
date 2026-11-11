import pytest


class TestE15ConfigSelection:
    """`--variant` e `--only` selezionano davvero, e una selezione impossibile solleva.

    `--variant` accettava il valore e non lo usava: la curva delle opzioni richiesta per E15b
    sarebbe costata 4x perche' ogni ampiezza veniva girata su tutte e quattro le precisioni, e
    l'utente che chiedeva una riga ne avrebbe ricevute quattro senza un errore.
    """

    @staticmethod
    def _labels(**kwargs):
        from types import SimpleNamespace

        from layaadv.experiments import e15_quantization as e15

        kwargs.setdefault("model", "laya")
        args = SimpleNamespace(**kwargs)
        return [e15.label_of(c) for c in e15.configs(args)]

    def test_variant_selects_one_precision_of_the_pair_family(self):
        assert self._labels(model="deberta-onnx", variant="fp32") == ["deberta-onnx/fp32"]

    def test_only_selects_several(self):
        got = self._labels(model="deberta-onnx", only="fp32,q4")
        assert got == ["deberta-onnx/fp32", "deberta-onnx/q4"]

    def test_precisions_do_not_cross_between_families(self):
        """int8 e' di laya, fp32 e' di deberta-onnx: nessuno dei due e' silenziosamente ignorato."""
        assert self._labels(variant="int8") == ["laya/int8"]
        assert self._labels(only="nf4") == ["laya/nf4"]

    def test_impossible_selection_raises_instead_of_running_nothing(self):
        """Senza questo, `--only fp32` sulla famiglia laya produceva zero righe e un json vuoto."""
        with pytest.raises(SystemExit, match="selezione vuota"):
            self._labels(only="fp32")
        with pytest.raises(SystemExit, match="selezione vuota"):
            self._labels(model="deberta-onnx", variant="nf4")

    def test_no_selection_keeps_every_configuration(self):
        assert len(self._labels()) == 3
        assert len(self._labels(model="deberta-onnx")) == 4

    def test_variant_is_the_generic_precision_selector_of_both_families(self):
        """`--variant nf4` sulla famiglia laya dev'essere valido, e su deberta-onnx no.

        argparse accettava solo i quattro nomi ONNX, quindi l'nf4 di laya non era
        raggiungibile da quel flag e il fallimento diceva "invalid choice" invece di spiegare
        che le precisioni appartengono a famiglie diverse.
        """
        from layaadv.cli import build_parser

        assert self._labels(variant="nf4") == ["laya/nf4"]
        argv = build_parser().parse_args(["e15", "--model", "deberta-onnx", "--variant", "nf4"])
        with pytest.raises(SystemExit, match="selezione vuota"):
            from layaadv.experiments import e15_quantization as e15

            e15.configs(argv)

    def test_parser_default_selects_every_precision(self):
        """Il default di `--variant` e' None (= tutte), non 'fp32'.

        Con default 'fp32' ogni `adv e15` senza flag sarebbe girato su una sola precisione e
        l'indice avrebbe perso tre righe senza errori.
        """
        from layaadv.cli import build_parser

        argv = build_parser().parse_args(["e15"])
        assert argv.variant is None
        assert len(self._labels(model=argv.model, variant=argv.variant)) == 3

class TestE15OptionCurve:
    """`--k` accetta una curva, e ogni K scrive su un file proprio.

    Senza, la curva si riduceva all'ultimo K misurato: ogni invocazione scriveva lo stesso
    `E15_quantization__<famiglia>.json` e cancellava la precedente, quindi un risultato spariva
    senza errore. E' il buco dell'indice per-scope, ricomparso sull'asse delle opzioni.
    """

    @staticmethod
    def _ks(raw):
        from types import SimpleNamespace

        from layaadv.experiments import e15_quantization as e15

        return e15.ks_of(SimpleNamespace(k=raw))

    def test_single_k_and_curve(self):
        assert self._ks(16) == [16]
        assert self._ks("4,8,16,32") == [4, 8, 16, 32]
        assert self._ks("32,4,8") == [4, 8, 32], "la curva va ordinata e deduplicata"

    def test_a_curve_lands_in_its_own_file(self):
        from layaadv.experiments import e15_quantization as e15

        assert e15.stem_of([16]) == "E15_quantization"
        assert e15.stem_of([4, 8]) == "E15_kcurve"

    def test_the_dataset_suffix_appears_only_on_an_override(self):
        """Il dataset entra nel nome solo quando non e' il default della famiglia.

        Il confronto era lista contro set, quindi sempre vero: il suffisso finiva in ogni
        nome di file. Non rompeva i numeri, ma rendeva indistinguibile una curva sul dataset
        di default da una curva su un altro.
        """
        from layaadv.experiments import e15_quantization as e15

        assert e15.result_stem([16], "laya", {"massive-it"}, {"massive-it"}) == (
            "E15_quantization__laya"
        )
        assert e15.result_stem([16], "laya", {"banking77"}, {"massive-it"}) == (
            "E15_quantization__laya__banking77"
        )
        assert e15.result_stem([4, 8], "laya", {"massive-it"}, {"massive-it"}) == (
            "E15_kcurve__laya"
        )

    def test_nonsense_k_fails_loudly(self):
        for bad in ("4,abc", "", "1", "0"):
            with pytest.raises(SystemExit):
                self._ks(bad)

    def test_index_keeps_the_curve_and_the_single_k_table_apart(self):
        """Le chiavi dell'indice non devono collidere fra tabella singola e curva.

        Prendendo solo il nome della famiglia, `E15_quantization__deberta-onnx` e
        `E15_kcurve__deberta-onnx` finivano sulla stessa chiave e la curva spariva: 16 righe
        misurate, 4 nell'indice, nessun errore.
        """
        from layaadv.experiments import e15_quantization as e15

        index = e15._write_index()
        keys = set(index["scopes"])
        assert "E15_kcurve__deberta-onnx" in keys
        # il dataset non di default entra nel nome, altrimenti la stessa famiglia su due
        # dataset si sovascrive a vicenda (le righe vivevano solo in runs/)
        assert "E15_quantization__deberta-onnx__banking77" in keys
        assert "E15_quantization__laya__massive-it" in keys
        assert "E15_quantization__laya__banking77" in keys
        curve = index["scopes"]["E15_kcurve__deberta-onnx"]
        assert curve["n_rows"] == 16
        assert curve["k"] == [4, 8, 16, 32]
        assert index["n_rows"] == sum(v["n_rows"] for v in index["scopes"].values())

class TestE15DatasetOverride:
    """`--dataset` sposta tutte le famiglie su un dataset, e il default non ne sposta nessuna.

    Il flag era dichiarato con `default="banking77"` e non letto da nessuna parte: innocuo solo
    per caso. Collegarlo senza cambiarne il default avrebbe mandato laya su banking77, dove va a
    0.03 di accuracy, e la tabella di confronto si sarebbe riempita di righe degenerate senza
    che nessuno l'avesse chiesto.
    """

    @staticmethod
    def _datasets(**kwargs):
        from types import SimpleNamespace

        from layaadv.experiments import e15_quantization as e15

        kwargs.setdefault("model", "laya")
        return sorted({c["dataset"] for c in e15.configs(SimpleNamespace(**kwargs))})

    def test_parser_default_moves_nothing(self):
        from layaadv.cli import build_parser

        assert build_parser().parse_args(["e15"]).dataset is None
        assert self._datasets(dataset=None) == ["massive-it"]
        assert self._datasets(model="deberta-onnx", dataset=None) == ["banking77"]

    def test_override_moves_every_family(self):
        assert self._datasets(dataset="banking77") == ["banking77"]
        assert self._datasets(model="deberta-onnx", dataset="massive-it") == ["massive-it"]

    def test_min_agreement_is_refused_instead_of_ignored(self):
        """Accettare un filtro che non esiste prometterebbe un ASR su un sottoinsieme non detto."""
        from types import SimpleNamespace

        from layaadv.experiments import e15_quantization as e15

        with pytest.raises(SystemExit, match="min-agreement"):
            e15.run(SimpleNamespace(model="laya", k=16, limit=2, min_agreement=0.8))

class TestE15Replicates:
    """`--replicates` ridisegna il benchmark e dichiara se le repliche attraversano la soglia.

    Senza, la curva era single seed e un ASR di 0.496 si sarebbe pubblicato come 0.50 con una
    cifra che sembrava una misura. Il numero che conta non e' la media: e' se la soglia sta
    dentro l'escursione, e quello va detto per nome.
    """

    @staticmethod
    def _seeds(**kw):
        from types import SimpleNamespace

        from layaadv.experiments import e15_quantization as e15

        return e15.seeds_of(SimpleNamespace(**kw))

    def test_seeds_are_consecutive_from_the_base(self):
        assert self._seeds(replicates=1) == [0]
        assert self._seeds(replicates=3) == [0, 1, 2]
        assert self._seeds(replicates=3, seed=100) == [100, 101, 102]

    def test_nonsense_refused(self):
        for bad in (0, -1):
            with pytest.raises(SystemExit, match="replicates"):
                self._seeds(replicates=bad)

    @staticmethod
    def _row(asr, k=16, config="laya/none", seed=0):
        return {
            "config": config, "k": k, "seed": seed, "asr": asr, "accuracy_clean": 0.3,
            "oer_adv": {"0.75": 0.1},
        }

    def test_straddling_the_threshold_is_declared(self):
        from layaadv.experiments import e15_quantization as e15

        rows = [self._row(a, seed=s) for a, s in zip((0.496, 0.500, 0.504), (0, 1, 2))]
        spread = e15.aggregate(rows)
        assert len(spread) == 1
        got = spread[0]
        assert got["asr_mean"] == 0.5
        assert got["asr_spread_pp"] == pytest.approx(0.8)
        assert got["asr_straddles_threshold"] is True
        assert "attraversa" in got["note"]

    def test_a_tight_group_does_not_straddle(self):
        from layaadv.experiments import e15_quantization as e15

        rows = [self._row(a, seed=s) for a, s in zip((0.930, 0.935, 0.940), (0, 1, 2))]
        spread = e15.aggregate(rows)
        assert spread[0]["asr_straddles_threshold"] is False
        assert spread[0]["note"] is None

    def test_a_single_replicate_is_not_aggregated(self):
        """Con una sola replica non c'e' dispersione: riportarne una sarebbe inventarla."""
        from layaadv.experiments import e15_quantization as e15

        assert e15.aggregate([self._row(0.9)]) == []
