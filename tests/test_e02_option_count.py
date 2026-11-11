"""E02 pure-function tests: expand_choice_question."""

from __future__ import annotations

from layaadv.experiments.e02_option_count import expand_choice_question


def test_expand_returns_same_type():
    q = {"type": "choice", "instructions": "Pick", "criteria": {"a": "opt a", "b": "opt b"}}
    out = expand_choice_question(q, "a", 4)
    assert out["type"] == "choice"
    assert out["instructions"] == "Pick"


def test_expand_keeps_gold():
    q = {"type": "choice", "instructions": "Pick", "criteria": {"a": "opt a", "b": "opt b"}}
    out = expand_choice_question(q, "a", 4)
    assert "a" in out["criteria"]
    assert out["criteria"]["a"] == "opt a"


def test_expand_reaches_k_options():
    q = {"type": "choice", "instructions": "Pick", "criteria": {"a": "opt a", "b": "opt b"}}
    out = expand_choice_question(q, "a", 8)
    assert len(out["criteria"]) == 8


def test_expand_preserves_original_non_gold():
    q = {"type": "choice", "instructions": "Pick", "criteria": {"a": "opt a", "b": "opt b"}}
    out = expand_choice_question(q, "a", 4)
    assert "b" in out["criteria"]
    assert out["criteria"]["b"] == "opt b"


def test_expand_noul_unchanged():
    q = {"type": "noul", "instructions": "Is it?"}
    out = expand_choice_question(q, "true", 4)
    assert out == q


def test_expand_score_unchanged():
    q = {"type": "score", "instructions": "Rate", "criteria": ["low", "mid", "high"]}
    out = expand_choice_question(q, "low", 4)
    assert out == q


def test_expand_k_smaller_than_original_trims():
    q = {
        "type": "choice",
        "instructions": "Pick",
        "criteria": {"a": "opt a", "b": "opt b", "c": "opt c", "d": "opt d"},
    }
    out = expand_choice_question(q, "a", 2)
    assert len(out["criteria"]) == 2
    assert "a" in out["criteria"]


def test_expand_gold_not_in_criteria_still_reaches_k():
    q = {"type": "choice", "instructions": "Pick", "criteria": {"x": "opt x"}}
    out = expand_choice_question(q, "missing_gold", 4)
    assert len(out["criteria"]) == 4
    assert "x" in out["criteria"]


def test_expand_large_k():
    q = {"type": "choice", "instructions": "Pick", "criteria": {"a": "opt a"}}
    out = expand_choice_question(q, "a", 32)
    assert len(out["criteria"]) == 32
    assert "a" in out["criteria"]


def test_run_config_pins_batch_size_chunk_and_provenance():
    """Ogni parametro che sposta una previsione o una riproducibilita' va nel config.

    E02 aveva solo `k` e `n_cases`: senza `batch_size`/`chunk` due run con
    parametri diversi condividerebbero il `run_id`, e in bf16 la composizione del
    batch puo' spostare una previsione. Senza `dataset_sha256`/`subset_hash` non si
    puo' dimostrare che baseline e run confrontano lo stesso split.
    """
    import inspect

    from layaadv.experiments import e02_option_count

    src = inspect.getsource(e02_option_count.run)
    for key in ('"batch_size": engine.batch_size', '"chunk": CHUNK', '"dataset_sha256"', '"subset_hash"'):
        assert key in src, f"{key} manca dal config di E02"
    assert "chunk=CHUNK" in src, "evaluate non riceve il chunk dichiarato nel config"
    assert e02_option_count.CHUNK > 1
