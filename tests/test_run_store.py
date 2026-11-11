"""RunStore: il resume è la differenza fra 'riga di nuovo' e 'riga persa'."""

from __future__ import annotations

import json

import pytest

from layaadv import paths
from layaadv.io import ConfigMismatch, RunStore, read_jsonl


@pytest.fixture(autouse=True)
def results_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS", tmp_path)
    return tmp_path


CONFIG = {"experiment": "T", "checkpoint": "ck", "n": 10}


def test_append_is_idempotent_and_resume_filters():
    store = RunStore.create("T", dict(CONFIG))
    assert store.append([{"id": "a"}, {"id": "b"}]) == 2
    assert store.append([{"id": "a"}, {"id": "c"}]) == 1  # "a" già fatto

    resumed = RunStore.create("T", dict(CONFIG))
    assert resumed.completed_ids() == {"a", "b", "c"}
    assert resumed.append([{"id": "b"}]) == 0


def test_tampered_config_is_refused():
    """run_id deriva dal config: config diversi finiscono in run diversi (nessun
    mix). Il guard scatta solo se il config.json registrato è stato manomesso."""
    store = RunStore.create("T", dict(CONFIG))
    (store.dir / "config.json").write_text('{"checkpoint": "altro"}')
    with pytest.raises(ConfigMismatch):
        RunStore.create("T", dict(CONFIG))


def test_different_config_gets_its_own_run():
    a = RunStore.create("T", dict(CONFIG))
    b = RunStore.create("T", {**CONFIG, "n": 99})
    assert a.run_id != b.run_id
    a.append([{"id": "a"}])
    assert b.completed_ids() == set()  # i progressi di a non contaminano b


def test_no_resume_refuses_existing_run():
    RunStore.create("T", dict(CONFIG))
    with pytest.raises(ConfigMismatch):
        RunStore.create("T", dict(CONFIG), resume=False)


def test_truncated_tail_is_dropped(tmp_path):
    """Un kill a metà riga non deve rovinare il resto del run."""
    path = tmp_path / "preds.jsonl"
    path.write_text('{"id": "a"}\n{"id": "b"}\n{"id": "c"')  # ultima riga tagliata
    assert [r["id"] for r in read_jsonl(path)] == ["a", "b"]


def test_resume_repairs_truncated_tail_before_appending():
    store = RunStore.create("T", dict(CONFIG))
    store._preds_path.write_bytes(b'{"id":"a"}\n{"id":"cut"')
    resumed = RunStore.create("T", dict(CONFIG))
    assert resumed.append([{"id": "b"}]) == 1
    assert [r["id"] for r in resumed.read_records()] == ["a", "b"]
    assert resumed._preds_path.read_bytes() == b'{"id":"a"}\n{"id": "b"}\n'


def test_resume_separates_valid_final_line_without_newline():
    store = RunStore.create("T", dict(CONFIG))
    store._preds_path.write_bytes(b'{"id":"a"}')
    resumed = RunStore.create("T", dict(CONFIG))
    assert resumed.append([{"id": "b"}]) == 1
    assert [r["id"] for r in resumed.read_records()] == ["a", "b"]


def test_append_rejects_duplicate_ids_within_one_batch():
    store = RunStore.create("T", dict(CONFIG))
    with pytest.raises(ValueError, match="id duplicato"):
        store.append([{"id": "a"}, {"id": "a"}])
    assert not store._preds_path.exists()


def test_append_rejects_records_without_id():
    store = RunStore.create("T", dict(CONFIG))
    with pytest.raises(KeyError, match="record senza chiave"):
        store.append([{"id": "a"}, {"label": "missing-id"}])
    assert not store._preds_path.exists()


def test_finish_writes_summary_with_env():
    store = RunStore.create("T", dict(CONFIG))
    store.append([{"id": "a"}])
    out = store.finish({"accuracy": 1.0})
    payload = json.loads(out.read_text())
    assert payload["run_id"] == store.run_id
    assert payload["summary"]["accuracy"] == 1.0
    assert "env" in payload and "laya" in payload["env"]


def test_state_file_tracks_progress():
    store = RunStore.create("T", dict(CONFIG))
    store.append([{"id": "a"}])
    state = json.loads((store.dir / "state.json").read_text())
    assert state["completed"] == 1 and state["last_key"] == "a"


def test_latency_sidecar_survives_resume():
    """Un run ripreso deve riportare i percentili di TUTTO il run, non solo dei
    batch finali: senza sidecar il p50 di E01 cambierebbe a ogni resume e i
    confronti con E00/E02 non significherebbero piu' nulla."""
    store = RunStore.create("T", dict(CONFIG))
    store.append_latency(10.0)
    store.append_latency(20.0)

    resumed = RunStore.create("T", dict(CONFIG))
    assert resumed.read_latencies() == [10.0, 20.0]
    resumed.append_latency(30.0)
    assert resumed.read_latencies() == [10.0, 20.0, 30.0]


def test_latency_is_isolated_per_run():
    a = RunStore.create("T", dict(CONFIG))
    b = RunStore.create("T", {**CONFIG, "n": 99})
    a.append_latency(5.0)
    assert b.read_latencies() == []


# --------------------------------------------------------------------------------------
# Due modelli diversi non possono condividere uno store
# --------------------------------------------------------------------------------------
def _adv_config(identity: dict) -> dict:
    """Il config che E03 costruisce per il braccio avversario.

    Riprodotto qui con e senza `run_identity`, che è esattamente la differenza
    introdotta il 2026-10-04.
    """
    return {
        "experiment": "E03_option_injection",
        "kind": "adversarial",
        "checkpoint": "some-model",
        "dataset": "massive-it",
        "k": 16,
        "seed": 0,
        **identity,
    }


def test_two_backends_cannot_share_an_adversarial_store():
    """PERCHÉ il config avversario porta `run_identity`.

    `RunStore.create` su una directory esista **riprende senza sollevare** se
    `stored == config`. Quindi un config che non registra l'identità del modello
    non è solo una provenienza mancante: è un config che due backend diversi
    producono identico, e il resume legge in silenzio le previsioni dell'altro.
    Questo è il buco che ha retto dal 2026-09-30 al 2026-10-04 nei 149 store
    avversari di E03.
    """
    laya = {"backend": "laya", "device": "cuda", "quantization": "w-only none"}
    clm = {"backend": "clm-llama", "device": "cpu", "quantization": "Q6_K"}

    with_identity = RunStore.create("T", _adv_config(laya))
    assert with_identity.run_id != RunStore.create("T", _adv_config(clm)).run_id


def test_without_identity_two_backends_would_actually_collide():
    """Il contro-prova: senza `run_identity` i due hash coincidono.

    Serve a dimostrare che il test sopra non passa per caso, e che il pericolo è
    reale e non teorico. Se questo test fallisce perché il hashing è cambiato,
    il test precedente sta difendendo una proprietà che non esiste più.
    """
    without = _adv_config({})
    a = RunStore.create("T", dict(without))
    b = RunStore.create("T", dict(without), force_new=True)
    assert a.run_id != b.run_id, "forza nuova produce un id diverso per costruzione"
    # il punto: le due config sono identiche, quindi il resume le confonderebbe
    assert RunStore.create("T", dict(without)).run_id == a.run_id
