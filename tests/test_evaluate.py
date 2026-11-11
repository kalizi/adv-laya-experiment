"""evaluate: il blocco latency e il contratto di resume.

Nessun torch/GPU qui: lo scopo e' verificare che le percentili registrate
siano attributed al run giusto e sopravvivano al resume, cosa che a runtime
e' invisibile ma renderebbe i confronti fra esperimenti falsi.
"""

from __future__ import annotations

import pytest

from layaadv import paths
from layaadv.evaluate import evaluate
from layaadv.io import RunStore
from layaadv.schema import Case, Prediction

Q = {"risk": {"type": "noul", "instructions": "Risky?"}}


@pytest.fixture(autouse=True)
def results_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS", tmp_path)
    return tmp_path


class FakeEngine:
    """Stand-in for LayaEngine: same predict_batch contract, no model."""

    batch_size = 16

    def __init__(self, label: str = "true", delay: float = 0.0, device: str = "cuda:0"):
        self.label = label
        self.delay = delay
        self.device = device
        self.batches: list[int] = []

    def runtime_state(self) -> dict:
        return {"device": self.device, "dtype": "bf16"}

    def predict_batch(self, cases, batch_size=None):
        import time

        self.batches.append(len(cases))
        if self.delay:
            time.sleep(self.delay)
        return [
            Prediction(
                id=c.id,
                answers={"risk": {"type": "noul", "noul": 1.0, "confidence": 1.0}},
                labels={"risk": self.label},
                usage={"input_tokens": 10, "output_tokens": 0},
            )
            for c in cases
        ]


def cases(n: int) -> list[Case]:
    return [Case(id=str(i), state="s", questions=Q, gold={"risk": "true"}) for i in range(n)]


def test_metrics_block_includes_latency():
    res = evaluate(FakeEngine(), cases(4), chunk=2)
    lat = res.metrics["latency"]
    assert lat["n"] == 2  # 4 casi, chunk 2 => 2 batch
    assert lat["p50_ms"] > 0
    assert "mean_ms" in lat and "p95_ms" in lat


def test_one_latency_sample_per_batch():
    engine = FakeEngine()
    res = evaluate(engine, cases(6), chunk=2)
    assert engine.batches == [2, 2, 2]
    assert res.metrics["latency"]["n"] == 3


def test_latency_persisted_and_replayed_on_resume():
    cfg = {"experiment": "T", "checkpoint": "ck"}
    store = RunStore.create("T", dict(cfg))
    first = evaluate(FakeEngine(), cases(4), store=store, chunk=2)
    assert first.skipped == 0

    resumed = RunStore.create("T", dict(cfg))
    second = evaluate(FakeEngine(), cases(4), store=resumed, chunk=2)

    assert second.skipped == 4  # nulla ricalcolato
    # i percentili coprono i 2 batch originali, non zero del segmento finale
    assert second.metrics["latency"]["n"] == first.metrics["latency"]["n"] == 2


def test_resume_accumulates_onto_historical_latency():
    """Dopo un resume il blocco latency deve sommare i campioni storici a quelli
    nuovi: 1 campione dal primo segmento + 1 dal batch ricalcolato. Se tenesse
    solo i nuovi, il p50 di un run interrotto e ripreso sarebbe diverso da
    quello di un run mai interrotto, e i confronti fra run falsi."""
    cfg = {"experiment": "T", "checkpoint": "ck"}
    store = RunStore.create("T", dict(cfg))
    first = evaluate(FakeEngine(), cases(2), store=store, chunk=2)
    assert first.metrics["latency"]["n"] == 1

    resumed = RunStore.create("T", dict(cfg))
    res = evaluate(FakeEngine(), cases(4), store=resumed, chunk=2)

    assert res.skipped == 2
    assert res.metrics["latency"]["n"] == 2
    assert res.metrics["runtime_s"]["batches_this_process"] == 1


def test_accuracy_unaffected_by_latency_instrumentation():
    """La strumentazione non deve perturbare lo scoring: stesso input, stessa
    accuracy, indipendentemente dal chunking."""
    a = evaluate(FakeEngine(), cases(10), chunk=3).metrics["accuracy"]
    b = evaluate(FakeEngine(), cases(10), chunk=10).metrics["accuracy"]
    assert a == b == 1.0


def test_runtime_block_reports_batches():
    res = evaluate(FakeEngine(), cases(5), chunk=2)
    assert res.metrics["runtime_s"]["batches_this_process"] == 3
    assert res.metrics["runtime_s"]["this_process"] >= 0.0


def test_chunk_none_scores_everything_in_one_batch():
    """chunk=None = un solo chunk: nessun caso viene perso (il checkpoint non
    deve MAI troncare i dati, solo la frequenza di flush)."""
    engine = FakeEngine()
    res = evaluate(engine, cases(10))
    assert engine.batches == [10]
    assert len(res.records) == 10
    assert res.skipped == 0


def test_one_chunk_reports_a_single_latency_sample():
    """Con un solo chunk c'e' un solo campione: p95 e std non hanno significato
    e il blocco deve dirlo con n=1, non fingere una distribuzione."""
    lat = evaluate(FakeEngine(), cases(8), chunk=8).metrics["latency"]
    assert lat["n"] == 1
    assert lat["std_ms"] == 0.0


def test_chunk_does_not_change_which_cases_are_scored():
    """Il chunk non filtra nulla: stesso set di id e stesso numero di record,
    qualunque sia la granularita'."""
    baseline = {r["id"] for r in evaluate(FakeEngine(), cases(12), chunk=12).records}
    for chunk in (1, 5, 12, 100):
        ids = {r["id"] for r in evaluate(FakeEngine(), cases(12), chunk=chunk).records}
        assert ids == baseline, f"chunk={chunk} ha perso casi"


def test_duplicate_case_ids_are_refused():
    """I record sono raccolti per `id`: un id ripetuto fa contare due volte la stessa
    predizione, e il risultato è un numero che non si può riprodurre dal dump."""
    dupes = [Case(id="7", state="s", questions=Q, gold={"risk": "true"}) for _ in range(2)]
    with pytest.raises(ValueError, match="id duplicati"):
        evaluate(FakeEngine(), dupes)


@pytest.mark.parametrize("mutation, message", [
    (lambda predictions: predictions[:-1], "predizioni per"),
    (lambda predictions: list(reversed(predictions)), "predizioni fuori ordine"),
])
def test_backend_must_return_one_prediction_per_case_in_order(mutation, message):
    engine = FakeEngine()
    original = engine.predict_batch
    engine.predict_batch = lambda batch, batch_size=None: mutation(original(batch, batch_size))
    with pytest.raises(ValueError, match=message):
        evaluate(engine, cases(2))


# ------------------------------------------------------------------ velocita'
def test_throughput_is_reported_per_decision_not_per_batch():
    """Un p50 per batch non dice niente: si muove con `batch_size` e con quante
    domande porta un caso. L'unita' confrontabile fra run e' il millisecondo per
    *decisione*, ed e' quella che serve a un Pareto contro ASR/OER."""
    res = evaluate(FakeEngine(delay=0.01), cases(10), chunk=5)
    tp = res.metrics["throughput"]
    assert tp["ms_per_decision"] > 0 and tp["decisions_per_s"] > 0
    # 10 decisioni in 2 batch da 5: i millisecondi cronometrati sono tutti, e il
    # denominatore sono le decisioni dentro i batch, non il totale della run
    assert tp["decisions_per_s"] == pytest.approx(1000.0 * 10 / res.metrics["latency"]["total_ms"], rel=1e-3)


def test_throughput_survives_a_resume_without_counting_reused_records():
    """Un resume riusa i record gia' scritti ma riusa anche i loro campioni.

    Se il denominatore fosse il totale della run, throughput sottostimerebbe della
    frazione riusata: qui 2 record su 4 arrivano dal primo segmento, e il conteggio
    deve sommare solo i 2 cronometrati ora.
    """
    cfg = {"experiment": "T", "checkpoint": "ck"}
    RunStore.create("T", dict(cfg))
    evaluate(FakeEngine(delay=0.01), cases(2), store=RunStore.create("T", dict(cfg)), chunk=2)
    res = evaluate(FakeEngine(delay=0.01), cases(4), store=RunStore.create("T", dict(cfg)), chunk=2)
    # 2 campioni storici + 2 nuovi = 4 decisioni cronometrate su 4 record
    assert res.metrics["latency"]["n"] == 2
    assert res.metrics["throughput"] is not None
    assert res.metrics["throughput_basis"].startswith("decisioni dentro")


def test_latency_records_where_the_model_actually_was():
    """Il device del campione, non quello del blocco env.

    `env_fingerprint` riporta la GPU se CUDA e' disponibile, quindi un run che laya
    ha demoto a CPU a meta' dice ancora "RTX 3080": e un fallback costa ~10x. Il
    numero di device distinti dentro un run rende il fatto visibile invece che
    nascosto, che e' l'unico motivo per cui il confronto fra checkpoint e' lecito.
    """
    cfg = {"experiment": "T", "checkpoint": "ck"}
    evaluate(FakeEngine(delay=0.01), cases(2), store=RunStore.create("T", dict(cfg)), chunk=1)
    res = evaluate(
        FakeEngine(delay=0.01, device="cpu"),
        cases(4),
        store=RunStore.create("T", dict(cfg)),
        chunk=2,
    )
    assert set(res.metrics["latency"]["devices"]) == {"cuda:0", "cpu"}


def test_throughput_is_none_without_samples():
    assert evaluate(FakeEngine(), cases(1)).metrics["latency"]["n"] == 1
    from layaadv.metrics.core import throughput

    assert throughput({"n": 0}, 10) is None
    assert throughput({"total_ms": 100.0}, 0) is None


def test_batching_report_says_when_batch_size_cannot_act():
    """`batch_size` agisce solo se piu' stati condividono la firma delle domande.

    Il caso normale sui dataset degli attacchi e' l'opposto: `loaders.load_cases` dà a ogni
    caso il proprio set di opzioni, quindi ogni gruppo ha un membro e `batch_size` non ha
    effetto. Un `run_id` che registra `batch_size: 16` senza dire questo descrive una leva
    che non ha mosso niente, quindi il fatto va registrato.
    """
    from layaadv.model.engine import LayaEngine

    def case(i, q):
        return Case(id=str(i), state=f"s{i}", questions={"q": q}, gold={"q": "a"})

    shared = case(0, {"type": "choice", "criteria": {"a": "", "b": ""}})
    distinct = [
        case(i, {"type": "choice", "criteria": {"a": "", f"o{i}": ""}}) for i in range(4)
    ]

    grouped = LayaEngine.batching_report([shared] * 8)
    assert grouped["question_groups"] == 1
    assert grouped["max_states_per_group"] == 8
    assert grouped["batch_size_effective"] == 8
    assert grouped["batching_is_inert"] is False

    inert = LayaEngine.batching_report(distinct)
    assert inert["question_groups"] == 4
    assert inert["batch_size_effective"] == 1
    assert inert["batching_is_inert"] is True, "4 set di opzioni diversi => 4 forward, batch_size irrilevante"
