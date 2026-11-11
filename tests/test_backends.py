"""Il backend e' un parametro, non una convenzione.

`open_backend` deve restare l'unico punto in cui un esperimento sceglie il
modello: se domani un backend qui, l'import di torch/onnx resta confinato a
`model/engine.py` come impone la regola di isolamento del progetto.

E il backend ONNX deve produrre **le stesse risposte** del backend nativo, non
un numero plausibile: il test centrale qui sotto e' differenziale, chiama il
codice di decodifica di laya e pretendeuguaglianza campo per campo. Non servono
pesi e non si accetta "quasi uguale".
"""

from types import SimpleNamespace

import numpy as np
import pytest

from layaadv.model.backends import OnnxLayaBackend, resolve_backend
from layaadv.model.engine import DEFAULT_CHECKPOINT, open_from_args
from layaadv.schema import Case


def test_resolve_known_backends():
    assert resolve_backend("laya") == "LayaEngine"
    assert resolve_backend("laya-onnx") == "OnnxLayaBackend"
    assert resolve_backend("clm-llama") == "ClmLlamaBackend"
    assert resolve_backend("d1-liquid") == "D1Backend"


def test_resolve_rejects_unknown():
    with pytest.raises(KeyError):
        resolve_backend("non-esiste")


def test_resolve_defaults_to_native():
    assert resolve_backend(None) == "LayaEngine"
    assert resolve_backend("  LAYA ") == "LayaEngine"


def test_backends_module_stays_lazy():
    """Importare il registry non deve tirare dentro torch o onnxruntime.

    Se lo facesse, `import layaadv` diventerebbe pesante e i test che non
    toccano i modelli pagherebbero un costo che oggi non pagano. La verifica
    gira in una subprocess perché toccare `sys.modules` in questa sessione
    inquinerebbe gli altri test (l'ho fatto e rompeva cinque test di loaders).
    """
    import subprocess
    import sys

    code = (
        "import sys; import layaadv.model.backends as b; "
        "assert 'torch' not in sys.modules, 'torch importato'; "
        "assert 'onnxruntime' not in sys.modules, 'onnxruntime importato'; "
        "print('ok')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_default_checkpoint_unchanged():
    """Il backend nativo resta il default: gli esperimenti esistenti non cambiano."""
    assert DEFAULT_CHECKPOINT == "convaiinnovations/laya-typed-decisions"


def test_clm_rejects_laya_checkpoint_before_contacting_servers():
    from layaadv.model.clm_backend import ClmLlamaBackend

    with pytest.raises(ValueError, match="checkpoint Laya"):
        ClmLlamaBackend("convaiinnovations/laya-typed-decisions")


# ------------------------------------------------------------------ --model wiring
def test_open_from_args_honours_the_model_flag(monkeypatch):
    """`--model` non puo' essere accettato e ignorato.

    E' il modo piu' economico di misurare una cosa e riportarne un'altra: il flag
    c'era nell'help, il resolver c'era, e nessun esperimento li collegava.
    """
    import layaadv.model.engine as engine

    seen: dict = {}

    def fake(spec, checkpoint, **kwargs):
        seen.update(spec=spec, checkpoint=checkpoint, **kwargs)
        return "engine"

    monkeypatch.setattr(engine, "open_backend", fake)
    args = SimpleNamespace(
        model="laya-onnx", device="cpu", batch_size=8, compile=True, variant="q4", fast=True
    )
    assert open_from_args(args, "some/dir") == "engine"
    # `fast` deve arrivare all'engine: la capacita' esiste in laya da prima ma senza il flag
    # era irraggiungibile dalla CLI, cioe' una accelerazione documentata che nessuno poteva
    # accendere.
    assert seen == {
        "spec": "laya-onnx",
        "checkpoint": "some/dir",
        "device": "cpu",
        "batch_size": 8,
        "compile": True,
        "fast": True,
        "quantize": "none",
        "variant": "q4",
    }


def test_open_from_args_defaults_to_the_native_backend(monkeypatch):
    import layaadv.model.engine as engine

    seen: dict = {}

    def fake(spec, checkpoint, **kwargs):
        seen.update(spec=spec, **kwargs)
        return "engine"

    monkeypatch.setattr(engine, "open_backend", fake)
    open_from_args(SimpleNamespace(device=None, batch_size=16), "x")
    assert seen == {
        "spec": "laya",
        "device": None,
        "batch_size": 16,
        "compile": False,
        "fast": False,
        "quantize": "none",
        "variant": "fp32",
    }


# ------------------------------------------------------------------ backend ONNX
# Una label che CONTIENE un ':' e' il caso che il vecchio backend rompeva:
# `render_options` rende "label: descrizione", e splittare sul primo ':' trasformava
# "I7: calendar_query" in "I7", fuori dal set di opzioni, con accuracy 0 e un numero
# che sembrava legittimo.
ALL_THREE = {
    "intent": {
        "type": "choice",
        "instructions": "?",
        "criteria": {"I7: calendar_query": "", "weather_query": ""},
    },
    "risk": {"type": "noul", "instructions": "?"},
    "urgency": {"type": "score", "instructions": "?", "criteria": ["basso", "medio", "alto"]},
}

LOGITS = np.array([[2.0, 0.5, -1.0], [1.0, -0.5, 0.25], [0.2, 0.9, 0.4]], dtype=np.float32)
ACT = np.array([[0.7, 0.3], [0.6, 0.4], [0.55, 0.45]], dtype=np.float32)


def _decode_both(questions: dict, temperature, by_options) -> tuple[dict, dict]:
    """(risposte di laya, risposte del backend ONNX) sugli stessi logit."""
    from laya.agent import Agent
    from laya.common import QTYPES, clamp_temperature, confidence_from_probs, render_options, temp_bucket

    temp = [clamp_temperature(t) for t in temperature]
    by_opt = {k: clamp_temperature(v) for k, v in by_options.items()}

    stub = SimpleNamespace(temperature=temp, temperature_by_options=by_opt)
    internal = {qid: Agent._to_internal(q) for qid, q in questions.items()}
    ids = list(questions)
    items = [
        {
            "ids": [0],
            "markers": list(range(len(render_options(internal[qid])))),
            "qtype": QTYPES[internal[qid]["t"]],
        }
        for qid in ids
    ]
    laya_out = Agent._decode_answers(stub, LOGITS, ACT, items, ids, internal, 0)

    b = OnnxLayaBackend.__new__(OnnxLayaBackend)
    b._qtypes = QTYPES
    b._temperature = temp
    b._temperature_by_options = by_opt
    b._confidence_from_probs = confidence_from_probs
    b._temp_bucket = temp_bucket
    onnx_out = {
        qid: b._decode(qid, internal[qid], LOGITS[r], float(ACT[r][0]), len(items[r]["markers"]))
        for r, qid in enumerate(ids)
    }
    return laya_out, onnx_out


@pytest.mark.parametrize(
    "temperature,by_options",
    [
        ([1.0, 1.0, 1.0], {}),  # laya-multilingual: nessuna temperatura
        ([1.0, 1.0, 1.0], {"choice:2": 0.10058280825614929}),  # sotto 1 -> clamp 0.5: il caso che divergeva
        ([1.6369, 1.2514, 1.9834], {"choice:11+": 0.5, "noul:2": 1.98}),  # profilo di `laya`
    ],
    ids=["no-temperature", "choice:2-clamped", "laya-like"],
)
def test_onnx_decode_is_identical_to_laya(temperature, by_options):
    """Ogni risposta deve coincidere campo per campo, `confidence` compresa.

    Senza la confidence `record_for` metteva 0.0 su ogni decisione, quindi
    OER, ECE e confidenza sul torto erano zero per costruzione: il backend
    alternativo riportava "il guardrail non si rompe" senza aver misurato nulla.
    """
    laya_out, onnx_out = _decode_both(ALL_THREE, temperature, by_options)
    assert set(onnx_out) == set(laya_out) == set(ALL_THREE)
    for qid, expected in laya_out.items():
        assert onnx_out[qid] == expected, f"{qid}: {onnx_out[qid]} != {expected}"


def test_onnx_decode_keeps_a_label_containing_a_colon():
    _, onnx_out = _decode_both(ALL_THREE, [1.0] * 3, {})
    assert onnx_out["intent"]["choice"] in {"I7: calendar_query", "weather_query"}
    assert set(onnx_out["intent"]["probabilities"]) == {"I7: calendar_query", "weather_query"}


def test_onnx_decode_emits_confidence_for_every_primitive():
    _, onnx_out = _decode_both(ALL_THREE, [1.0] * 3, {})
    for qid, ans in onnx_out.items():
        assert 0.0 <= ans["confidence"] <= 1.0, qid
    # softmax([1.0, -0.5])[1]: se la temperature fosse ignorata il valore sarebbe
    # diverso, ed e' il controllo che distingue i due backend.
    assert onnx_out["risk"]["noul"] == pytest.approx(round(1 / (1 + np.exp(1.5)), 4), abs=1e-4)
    assert onnx_out["risk"]["confidence"] == pytest.approx(round(1 / (1 + np.exp(-1.5)), 4), abs=1e-4)


def test_onnx_lens_comes_from_the_declared_config_not_a_constant():
    """`laya` dichiara 512/192: un 1024/256 fisso sbaglia entrambi i budget.

    Con `max_len` sbagliato E04 non troncherebbe mai, con `head_max_len` sbagliato
    il muro del budget di E03 cadrebbe dove il checkpoint non tronca.
    """
    b = OnnxLayaBackend.__new__(OnnxLayaBackend)
    b._max_len, b._head_max_len = 512, 192
    assert b._lens() == (512, 192)


def test_missing_laya_config_is_an_error_not_a_guess(tmp_path):
    from layaadv.model.backends import _load_laya_config

    with pytest.raises(FileNotFoundError, match="laya_config.json"):
        _load_laya_config(tmp_path)
    (tmp_path / "laya_config.json").write_text('{"max_len": 512, "head_max_len": 192}')
    assert _load_laya_config(tmp_path)["head_max_len"] == 192


class _FakeSession:
    """Sessione ONNX minima: un solo forward, logit decrescenti, nessun peso."""

    def __init__(self):
        self.seen: list[dict] = []

    def run(self, names, feeds):
        self.seen.append(feeds)
        n, kmax = feeds["input_ids"].shape[0], feeds["marker_mask"].shape[1]
        return (
            np.tile(np.arange(kmax, dtype=np.float32)[::-1], (n, 1)),
            np.zeros((n, 2), dtype=np.float32),
        )

    def get_providers(self):
        return ["CPUExecutionProvider"]


def _bare_onnx_backend() -> OnnxLayaBackend:
    """Backend con solo il decoding e il collate reali: tokenizzatore e grafo finiti."""
    from laya.common import QTYPES, confidence_from_probs, render_options, temp_bucket

    b = OnnxLayaBackend.__new__(OnnxLayaBackend)
    b.batch_size = 16
    b._tok = None
    b._max_len, b._head_max_len = 1024, 256
    b._session = _FakeSession()
    b._internal = lambda qs: {qid: _internal(q) for qid, q in qs.items()}
    b._build_sequence = lambda tok, state, q, ml, hml: (
        np.array([1, 2, 3, 4], dtype=np.int64),
        list(range(len(render_options(q)))),
    )
    b._qtypes = QTYPES
    b._temperature = [1.0, 1.0, 1.0]
    b._temperature_by_options = {}
    b._confidence_from_probs = confidence_from_probs
    b._temp_bucket = temp_bucket
    return b


def _internal(qdef: dict) -> dict:
    from laya.agent import Agent

    return Agent._to_internal(qdef)


def test_onnx_predict_batch_answers_every_question_of_the_case():
    """Una riga del grafo per (stato, domanda).

    Rispondere solo alla prima domanda dava un `accuracy` calcolato su un quinto
    dei dati, con `n_decisions` 400 invece di 2000 e nessun segnale che qualcosa
    fosse andato storto.
    """
    b = _bare_onnx_backend()
    out = b.predict_batch([Case(id="c1", state="stato", questions=ALL_THREE, gold={})])

    assert len(out) == 1
    assert set(out[0].answers) == set(ALL_THREE), "una domanda del caso non e' stata risposta"
    assert set(out[0].labels) == set(ALL_THREE)
    assert b._session.seen[0]["input_ids"].shape[0] == 3, "una riga per (stato, domanda), non una per stato"


def test_onnx_predict_batch_masks_the_markers_it_actually_uses():
    b = _bare_onnx_backend()
    b.predict_batch([Case(id="c", state="s", questions=ALL_THREE, gold={})])
    feeds = b._session.seen[0]
    assert feeds["marker_mask"].sum(axis=1).tolist() == [2, 2, 3]
    assert feeds["qtype"].tolist() == [0, 2, 1]  # choice, noul, score


def test_onnx_predict_batch_empty_input():
    assert _bare_onnx_backend().predict_batch([]) == []


# ------------------------------------------------------- backend a coppie (DeBERTa)
def test_assemble_pairs_is_plain_index_bookkeeping():
    """La testa dell'export open-jev legge `seg`: ogni token porta il suo slot.

    -1 su stato e marker, `pairs + indice domanda` sui token della domanda,
    l'indice della coppia su quelli dell'opzione. E' tutta l'informazione che serve
    per far uscire un logit per (domanda, opzione), quindi si testa come aritmetica:
    il grafo da 3.4 GB non serve qui.
    """
    from layaadv.model.backends import assemble_pairs

    plan = assemble_pairs(
        [10, 11, 12],
        [("q0", [20, 21], [[30], [31, 32], [33]]), ("q1", [40], [[50]])],
    )
    assert plan["n_pairs"] == 4
    # stato: tre -1 in testa
    assert plan["seg"][:3] == [-1, -1, -1]
    # q0: due token di domanda, poi 1+2+1 token di opzione
    assert plan["seg"][3:5] == [4, 4]
    assert plan["seg"][5:6] == [0]
    assert plan["seg"][6:8] == [1, 1]
    assert plan["seg"][8:9] == [2]
    # q1: una domanda (pairs + 1 = 5) e la sua opzione (3)
    assert plan["seg"][9:10] == [5]
    assert plan["seg"][10:11] == [3]
    assert plan["groups"] == [[0, 1, 2], [3]]
    assert plan["pair_q"] == [4, 4, 4, 5]
    assert plan["pair_opt"] == [0, 1, 2, 3]
    assert len(plan["input_ids"]) == len(plan["seg"])


def test_assemble_pairs_handles_a_question_with_no_options_without_shifting_slots():
    """Una domanda senza opzioni non deve spostare gli slot delle successive.

    `seg` assegna gli slot delle domande in base al numero totale di coppie calcolato
    in anticipo: se quel conteggio dipendesse da quante domande sono state già viste,
    un caso degenere sposterebbe ogni coppia successiva e le risposte cambierebbero.
    """
    from layaadv.model.backends import assemble_pairs

    plan = assemble_pairs([1], [("q0", [2], []), ("q1", [3], [[4]])])
    assert plan["n_pairs"] == 1
    assert plan["groups"] == [[], [0]]
    assert plan["pair_q"] == [2]  # pairs + indice domanda = 1 + 1
    # q0 ha zero opzioni ma il suo slot di domanda e' comunque assegnato (pairs+0 = 1),
    # altrimenti q1 riceverebbe lo slot sbagliato e la sua risposta cambierebbe
    assert plan["seg"] == [-1, 1, 2, 0]


def test_pair_backend_rejects_torch_quantization_and_names_the_right_knob():
    """La precisione di questo backend si sceglie con `--variant`, non `--quantize`.

    I grafi quantizzati sono gia' nell'export: accettare `--quantize` qui produrrebbe un
    "int4" che in realta e' un grafo fp32 con un nome falso.
    """
    from layaadv.model.backends import OnnxPairBackend

    with pytest.raises(ValueError, match="--variant"):
        OnnxPairBackend("/nonexistent", quantize="nf4")
    assert set(OnnxPairBackend.VARIANTS) == {"fp32", "fp16", "q4", "q4f16"}


def test_pair_backend_labels_come_from_a_map_not_from_splitting_the_text():
    """La label confrontabile viene da una mappa costruita all'emissione.

    E' il bug che aveva il primo backend ONNX di laya: `render_options` rende
    "label: descrizione", splittare sul primo ':' tagliava "I7: calendar_query" in "I7",
    fuori dal set di opzioni, con accuracy 0 e un numero che sembrava legittimo.
    """
    from layaadv.model.backends import _labels_of

    q = {"t": "choice", "crit": {"I7: calendar_query": "", "weather_query": ""}, "ins": "?"}
    lab = _labels_of(q, ["I7: calendar_query", "weather_query"])
    assert lab["order"] == ["I7: calendar_query", "weather_query"]

    score = {"t": "score", "crit": ["basso", "medio", "alto"], "ins": "?"}
    assert _labels_of(score, ["level 0: basso"])["order"] == ["0", "1", "2"]
    noul = {"t": "noul", "crit": None, "ins": "?"}
    assert _labels_of(noul, ["false: ...", "true: ..."])["order"] == ["false", "true"]


def test_pair_backend_confidence_means_the_same_thing_as_layas():
    """Confrontare un OER fra due backend ha senso solo se la confidence significa la
    stessa cosa: laya usa 1 - H(p)/log(k) e l'export dichiara la propria calibrazione."""
    import numpy as np
    from laya.common import confidence_from_probs

    from layaadv.model.backends import _entropy_conf

    for p in ([0.5, 0.5], [0.9, 0.1], [0.6, 0.3, 0.1], [1.0, 0.0, 0.0]):
        assert _entropy_conf(p) == pytest.approx(confidence_from_probs(np.array(p), len(p)), abs=1e-4)


def test_open_from_args_forwards_the_graph_variant():
    """`variant` e' la precisione dei grafi ONNX e non puo' restare sulla soglia.

    Il primo giro di E15 ha caricato `model.onnx` quattro volte: quattro configurazioni
    diverse, quattro righe identiche, e una tabella che diceva "la quantizzazione non
    cambia niente". Il plumbing che perde un flag non e' un bug di comodo: e' un bug che
    produce la conclusione sbagliata con la forma giusta.
    """
    from layaadv.model.engine import open_from_args

    seen = {}

    def fake(spec, checkpoint, **kwargs):
        seen.update(kwargs)
        return "engine"

    import layaadv.model.engine as engine_mod

    original = engine_mod.open_backend
    engine_mod.open_backend = fake
    try:
        open_from_args(SimpleNamespace(model="deberta-onnx", variant="q4f16"), "dir")
    finally:
        engine_mod.open_backend = original
    assert seen["variant"] == "q4f16"


def test_e15_refuses_a_backend_that_is_not_the_one_requested():
    """La guardia di E15: quattro righe identiche sembrano un null result, non un errore."""
    from layaadv.experiments.e15_quantization import _assert_identity

    _assert_identity(
        {"quantize": "none", "variant": "q4"}, {"quantization": "none", "variant": "q4"}
    )
    with pytest.raises(RuntimeError, match="le due non si confrontano"):
        _assert_identity(
            {"quantize": "none", "variant": "fp16"}, {"quantization": "none", "variant": "fp32"}
        )
    with pytest.raises(RuntimeError, match="le due non si confrontano"):
        _assert_identity(
            {"quantize": "nf4", "variant": None}, {"quantization": "none", "variant": None}
        )


def test_each_backend_rejects_the_other_ones_precision_knob():
    """Ogni backend conosce una sola maniera di scegliere la precisione.

    Se `--quantize` arrivasse a deberta-onnx o `--variant` alla famiglia laya, senza rifiuto
    esplicito, entrambe le configurazioni girerebbero sullo stesso grafo e E15 produrrebbe
    una tabella di "la quantizzazione non cambia niente" costruita a mano da un errore di
    plumbing. E' successo: il primo giro ha caricato `model.onnx` quattro volte e il
    risultato sembrava un null result.

    Le guardie stanno prima di qualsiasi I/O, quindi il checkpoint inesistente non serve
    e il test non tocca la rete.
    """
    from layaadv.model.backends import OnnxPairBackend
    from layaadv.model.engine import LayaEngine

    with pytest.raises(ValueError, match="deberta-onnx"):
        LayaEngine("does-not-exist", variant="q4")
    with pytest.raises(ValueError, match="deberta-onnx"):
        OnnxLayaBackend("/does-not-exist", variant="q4")
    with pytest.raises(ValueError, match="--variant"):
        OnnxPairBackend("/does-not-exist", quantize="nf4")
    # d1 e' la quarta famiglia e ha una sola precisione (il dtype del checkpoint):
    # rifiuta tutti e quattro i selettori, non solo il proprio.
    from layaadv.model.d1_backend import D1Backend

    for knob in ("quantize", "variant", "compile", "fast"):
        with pytest.raises(ValueError):
            D1Backend("/does-not-exist", **{knob: "nf4" if knob == "quantize" else True})


def test_run_identity_carries_the_device_for_every_backend():
    """Il device sta nel `run_id`, o un resume mescola device diversi sotto un numero solo.

    Senza questo i sette record CPU di E15 sarebbero stati riletti da un run GPU con lo stesso
    `run_id`: le previsioni avrebbero avuto la latenza di una 3080 misurata su CPU. E su CPU la
    fp16 costa 1.8x la fp32 (nessun kernel nativo), su GPU e' nativa: la stessa tabella dice
    l'opposto, quindi il device e' una condizione sperimentale, non un dettaglio.
    """
    import inspect

    from layaadv.model.backends import OnnxLayaBackend, OnnxPairBackend
    from layaadv.model.engine import LayaEngine

    for cls in (LayaEngine, OnnxLayaBackend, OnnxPairBackend):
        assert hasattr(cls, "run_identity"), f"{cls.__name__} non espone run_identity()"
        assert "device" in inspect.getsource(cls.run_identity), f"{cls.__name__}: device assente"

    # d1 espone `run_identity` ma il device non e' nel sorgente del metodo con quel
    # nome: lo prende da `self._device_seen`, quindi il controllo sul nome non
    # basta. Si verifica sull'istanza nuda, come fa il test dedicato.
    from layaadv.model.d1_backend import D1Backend

    class _Param:
        dtype = "torch.bfloat16"

    class _Model:
        def parameters(self):
            return iter([_Param()])

    b = D1Backend.__new__(D1Backend)
    b._device_seen = "cuda"
    b._model = _Model()
    assert D1Backend.run_identity(b)["device"] == "cuda"


def test_onnx_laya_backend_has_a_run_identity_at_all():
    """Il buco vero: `--model laya-onnx` accettava il valore e il run moriva subito dopo."""
    from layaadv.model.backends import OnnxLayaBackend

    class Fake(OnnxLayaBackend):
        def __init__(self):  # noqa: D107 - si evita di caricare un grafo
            self.name = "laya-onnx"
            self._device_seen = "onnx:CUDAExecutionProvider"

    got = Fake().run_identity()
    assert got["backend"] == "laya-onnx"
    assert got["device"] == "onnx:CUDAExecutionProvider"
