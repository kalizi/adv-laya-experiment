"""E04 pure-function tests: push_out_state."""

from __future__ import annotations

from layaadv.experiments.e04_truncation import push_out_state


def test_zero_filler_returns_original():
    state = "hello world"
    assert push_out_state(state, 0) == state


def test_filler_prepends():
    state = "hello world"
    out = push_out_state(state, 10)
    assert out.endswith(state)
    assert len(out) > len(state)


def test_filler_token_count():
    state = "hello world"
    out = push_out_state(state, 20)
    filler_part = out[: -len(state) - 1]
    words = filler_part.split()
    assert len(words) == 20


def test_filler_is_neutral():
    state = "URGENT: payment overdue"
    out = push_out_state(state, 50)
    filler_part = out[: -len(state) - 1]
    assert "URGENT" not in filler_part
    assert "payment" not in filler_part


# ------------------------------------------------------------------ config/run
def test_config_chunk_is_the_chunk_actually_used():
    """Il config non deve raccontare un chunk che la run non usa.

    Il config dichiarava `chunk = engine.batch_size` mentre `evaluate` veniva
    chiamata senza `chunk`, quindi una sola forward pass su 400 casi: il p50 e il
    p95 erano lo stesso numero e `std` era 0 per costruzione, non perche' il run
    fosse silenzioso. Il numero nel config e' la promessa di riproducibilita':
    se non e' quello usato, l'identita' del run_id non descrive la run.
    """
    import inspect

    from layaadv.experiments import e04_truncation

    src = inspect.getsource(e04_truncation.run)
    assert '"chunk": CHUNK' in src, "il chunk nel config non e' quello dichiarato"
    assert "chunk=CHUNK" in src, "evaluate non riceve il chunk dichiarato nel config"
    assert e04_truncation.CHUNK > 1, "un chunk solo non produce campioni di latenza"
