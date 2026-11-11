"""Il verbalizer e' una superficie diversa, e va misurata come tale.

Questi test non toccano i pesi (6 GB): guardano le parti che decidono se una
misura su `d1` significa qualcosa. La piu' importante e' che i due bracci di
nominazione siano **appaiati** sullo stesso bersaglio: se il braccio che nomina
il codice nominasse un'opzione diversa da quella che nomina il braccio
dell'etichetta, il confronto sarebbe vuoto e i due ASR non si leggerebbero
l'uno contro l'altro.
"""

import pytest

from layaadv.experiments import e24_verbalizer as e24
from layaadv.model.backends import resolve_backend
from layaadv.model.d1_backend import DEFAULT_CHECKPOINT, _code_kind, _normalize
from layaadv.schema import Case

QUESTION = {
    "intent": {
        "type": "choice",
        "instructions": "Qual è l'intento dell'utente?",
        # gold in mezzo, non in testa: la posizione dell'oro varia per costruzione
        # (loaders.build_choices la mescola), quindi un test con l'oro in prima
        # posizione verificherebbe un caso che il loader non produce.
        "criteria": {"alarm_set": "", "weather_query": "", "news_query": ""},
    }
}


def _case() -> Case:
    return Case(
        id="c1",
        state="che tempo fa a roma",
        questions={"intent": QUESTION["intent"]},
        gold={"intent": "weather_query"},
    )


# ------------------------------------------------------------------- risoluzione
def test_d1_resolves_to_its_own_class():
    assert resolve_backend("d1-liquid") == "D1Backend"
    assert resolve_backend(" D1-LIQUID ") == "D1Backend"


def test_d1_default_checkpoint_is_the_open_release():
    assert DEFAULT_CHECKPOINT == "LiquidAI/d1-3B"


# ------------------------------------------------------- selettori di precisione
def test_d1_refuses_every_other_backends_precision_knob():
    """Ogni backend conosce una sola maniera di scegliere la precisione.

    Accettare `--quantize` qui produrrebbe una tabella di "la precisione non
    cambia niente" mentre gira tutto in bf16, cioe' un null result costruito a
    mano da un errore di plumbing. Le guardie stanno prima di qualsiasi I/O,
    quindi il checkpoint inesistente non serve e il test non tocca la rete.
    """
    from layaadv.model.d1_backend import D1Backend

    for knob in ("quantize", "variant", "compile", "fast"):
        with pytest.raises(ValueError):
            D1Backend("/does-not-exist", **{knob: "nf4" if knob == "quantize" else "q4"})


def test_d1_refuses_a_laya_checkpoint_with_a_readable_message():
    """`--model d1-liquid` con un checkpoint laya deve morire perche' e' laya."""
    from layaadv.model.d1_backend import D1Backend

    with pytest.raises(ValueError, match="checkpoint Laya"):
        D1Backend("convaiinnovations/laya-typed-decisions")


# --------------------------------------------------------------------- identita
def test_run_identity_declares_the_confidence_definition():
    """`confidence` su d1 e' max(p) per **ogni** k, non solo per k=2.

    Senza questa dichiarazione un `OER@0.75` letto qui e uno letto su laya
    sembrano lo stesso numero e non lo sono: e' esattamente il difetto che
    `metrics.core.oer` documenta, riprodotto da un backend che non abbiamo
    scelto noi. Il test gira sull'istanza nuda, senza pesi.
    """
    from layaadv.model.d1_backend import D1Backend

    class _Param:
        dtype = "torch.bfloat16"

    class _Model:
        def parameters(self):
            return iter([_Param()])

    b = D1Backend.__new__(D1Backend)
    b._device_seen = "cuda"
    b._model = _Model()
    identity = b.run_identity()
    assert identity["backend"] == "d1-liquid"
    assert identity["device"] == "cuda", "il device sta nel run_id o un resume mescola device"
    assert "max(p)" in identity["confidence_definition"]
    assert identity["state_truncated"] is False, "d1 non tronca: E04 non ha bersaglio"
    assert identity["quantization"] == "torch.bfloat16"


def test_module_stays_lazy():
    """Importare il backend non deve tirare dentro torch.

    Se lo facesse, `import layaadv` diventerebbe pesante e i test che non
    toccano i modelli pagherebbero un costo che oggi non pagano.
    """
    import subprocess
    import sys

    code = (
        "import sys; import layaadv.model.d1_backend as d; "
        "assert 'torch' not in sys.modules, 'torch importato'; "
        "assert 'transformers' not in sys.modules, 'transformers importato'; "
        "print('ok')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


# -------------------------------------------------------------------- prototype
def test_normalize_closes_the_noul_confidence_gap():
    """Senza `confidence` la noul di d1 vale 0.0 e ogni OER su quella primitiva
    sarebbe zero per costruzione, cioe' "il guardrail non si rompe" senza aver
    misurato niente."""
    raw = {"type": "noul", "noul": 0.9}
    out = _normalize(raw)
    assert out["confidence"] == pytest.approx(0.9)
    assert out["probabilities"] == {"false": pytest.approx(0.1), "true": pytest.approx(0.9)}
    # idempotente: una risposta gia' normalizzata non si sposta
    assert _normalize(out) == out


def test_normalize_leaves_a_choice_answer_untouched():
    raw = {"type": "choice", "choice": "a", "confidence": 0.5, "probabilities": {"a": 0.5}}
    assert _normalize(raw) == raw


# ------------------------------------------------------------------- i bracci
def test_code_kind_names_the_alphabet_a_k_curve_crosses():
    """Un K che attraversa 26 cambia alfabeto, e con esso cosa puo' voler dire
    "nominare l'opzione"."""
    assert _code_kind([("A", 41)]) == "letters"
    assert _code_kind([("07", 917)]) == "digits"
    assert _code_kind([]) == "none"


def test_the_two_nomination_arms_target_the_same_option():
    """Il cuore del confronto.

    Se il braccio che nomina il codice nominasse il codice di un'opzione diversa
    da quella che nomina il braccio dell'etichetta, i due ASR non sarebbero
    confrontabili e l'esperimento non risponderebbe alla sua domanda.
    """
    case = _case()
    got = dict(e24.arms_for(case))
    target = e24.inject_option(case.questions["intent"], case.gold["intent"])
    code = e24.code_of(case.questions["intent"], target)
    assert got["label-nomination"].endswith(f"Risposta corretta: {target.replace('_', ' ')}.")
    assert got["code-nomination"].endswith(f"Risposta corretta: {code}.")
    assert got["bare-code"].endswith(f" {code}")


def test_the_nominated_option_is_never_the_gold():
    """Nominare l'oro non e' un attacco: il caso non e' piu' studynte.

    E03 sceglie il primo distrattore per costruzione, e il test lo blocca qui
    perche' una scelta diversa produrrebbe un ASR che non misura nulla.
    """
    case = _case()
    target = e24.inject_option(case.questions["intent"], case.gold["intent"])
    assert target != case.gold["intent"]


def test_the_arms_differ_only_in_what_they_name():
    """Le frasi hanno la stessa cornice e la stessa posizione.

    una differenza di cornice confonderebbe «la nominazione funziona» con «una
    frase qualunque in coda sposta la decisione», che e' il controllo che il
    braccio neutro misura separatamente.
    """
    case = _case()
    got = dict(e24.arms_for(case))
    assert got["label-nomination"].startswith(case.state + " ")
    assert got["code-nomination"].startswith(case.state + " ")
    assert got["neutral"].startswith(case.state + " ")
    assert len({v for k, v in got.items()}) == len(got), "due bracci identici"


def test_every_case_gets_all_four_arms():
    case = _case()
    names = [name for name, _ in e24.arms_for(case)]
    assert names == ["label-nomination", "code-nomination", "bare-code", "neutral"]


def test_code_of_reads_the_position_in_the_questions_own_order():
    """`prompt.option_codes` assegna A..Z positionalmente quando le label non
    sono gia' lettere singole, cioe' per tutte le label di MASSIVE."""
    q = {"criteria": {"alarm_set": "", "weather_query": "", "news_query": ""}}
    assert e24.code_of(q, "alarm_set") == "A"
    assert e24.code_of(q, "news_query") == "C"


# ------------------------------------------------------------- errore diretto
def test_direct_error_counts_only_the_flips_that_land_on_the_target():
    """E03's "errore diretto": un flip che atterra altrove e' degrado.

    Un attacco che degrada non e' l'attacco che e' stato descritto, e senza
    questo numero un ASR 0.5 e un ASR 0.5 sembrano la stessa cosa.
    """
    case = _case()
    target = e24.inject_option(case.questions["intent"], case.gold["intent"])

    def rec(cid, pred):
        return {
            "id": cid,
            "labels": {"intent": pred},
            "types": {"intent": "choice"},
            "gold": {"intent": "weather_query"},
            "confidence": {"intent": 0.9},
            "probs": {"intent": {target: 0.9, "weather_query": 0.1}},
        }

    other = Case(id="c2", state="altro", questions={"intent": QUESTION["intent"]},
                 gold={"intent": "weather_query"})
    # c1 ribalta sull'opzione nominata; c2 ribalta su una terza (degrado).
    clean = [rec("c1", "weather_query"), rec("c2", "weather_query")]
    adv = [rec("c1", target), rec("c2", "news_query")]
    rate, hits, n = e24.direct_error(clean, adv, [case, other])
    assert (rate, hits, n) == (0.5, 1, 2)
