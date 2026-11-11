"""Raggruppamento per batching: stesso schema di questions => stesso gruppo.

La correttezza qui è invisibile a runtime (una riordinatura silenziosa delle
predizioni renderebbe ogni metrica falsa), quindi si testa puramente.
"""
from __future__ import annotations

import pytest

from layaadv.model.engine import group_cases, question_signature
from layaadv.schema import Case

Q_A = {"risk": {"type": "noul", "instructions": "Risky?"}}
Q_B = {"outcome": {"type": "choice", "instructions": "Outcome?", "criteria": {"ok": "ok", "ko": "ko"}}}


def case(cid, questions, state="s"):
    return Case(id=cid, state=state, questions=questions, gold={})


def test_signature_ignores_key_order():
    reordered = {"outcome": Q_B["outcome"], }  # stessi contenuti
    assert question_signature(Q_A) == question_signature(dict(Q_A))
    assert question_signature(Q_B) == question_signature(reordered)


def test_signature_differs_on_different_questions():
    assert question_signature(Q_A) != question_signature(Q_B)


def test_group_cases_partitions_by_signature_and_keeps_indices():
    cases = [case("0", Q_A), case("1", Q_B), case("2", Q_A), case("3", Q_B), case("4", Q_A)]
    groups = group_cases(cases)

    assert len(groups) == 2  # due schema di questions
    sizes = sorted(len(g) for g in groups)
    assert sizes == [2, 3]
    # ogni caso compare una volta sola, con il suo indice originale
    flat = sorted(i for g in groups for i, _ in g)
    assert flat == [0, 1, 2, 3, 4]


def test_reassembly_order_is_input_order():
    """predict_batch ricostruisce out[indice]: l'ordine di input deve tornare."""
    cases = [case("0", Q_B), case("1", Q_A), case("2", Q_B)]
    out = [None] * len(cases)
    for members in group_cases(cases):
        for i, c in members:
            out[i] = c.id
    assert out == ["0", "1", "2"]


def test_decision_temperature_is_logged_and_not_constant():
    """La curva K attraversa i cambi di bucket della temperature.

    `temp_bucket` passa da `3-5` a `6-10` a `11+`: oltre K=10 le temperature
    applicate non sono piu' quelle delle run con K bassi, quindi due righe della
    stessa tabella non sono confrontabili a occhio. E' il motivo per cui il bucket
    finisce in ogni riga.
    """
    from layaadv.model.engine import LayaEngine

    class FakeAgent:
        temperature = [1.0, 1.0, 1.0]
        temperature_raw = [1.0, 1.0, 1.0]
        temperature_by_options = {"choice:3-5": 1.760, "choice:11+": 0.5}
        temperature_by_options_raw = {"choice:3-5": 1.760, "choice:11+": 0.1006}

    engine = LayaEngine.__new__(LayaEngine)
    engine._agent = FakeAgent()
    engine._temperature = engine._agent.temperature
    engine._temperature_raw = engine._agent.temperature_raw
    engine._temperature_by_options = engine._agent.temperature_by_options
    engine._temperature_by_options_raw = engine._agent.temperature_by_options_raw

    small = engine.decision_temperature("choice", 4)
    big = engine.decision_temperature("choice", 16)
    assert small["bucket"] == "choice:3-5" and big["bucket"] == "choice:11+"
    assert small["temperature_applied"] == 1.760 and big["temperature_applied"] == 0.5
    # il valore dichiarato dal checkpoint e' quello sotto il clamp: la ragione del mismatch
    assert big["temperature_raw"] == 0.1006


def test_state_budget_is_the_real_room_not_the_head_size(monkeypatch):
    """Laya tiene un token per il SEP che chiude lo stato: `room = max_len - len(head) - 1`.

    Restituire `max_len - len(head)` faceva passare il gate di E00 per un motivo
    sbagliato: a `state_tokens == budget` l'ultimo token dello stato (che e' la coda,
    dove sta il marker di E00 e di E04) verrebbe tagliato, quindi il marker NON
    sarebbe visibile mentre la condizione del gate diceva il contrario.
    """
    pytest.importorskip("laya.agent")
    import laya.agent

    monkeypatch.setattr(laya.agent, "build_sequence", lambda tok, st, q, max_len, head_max_len: (
        list(range(30)),  # testa di 30 token
        [0],
    ))
    from layaadv.model.engine import LayaTokenBudget

    class B(LayaTokenBudget):
        _tok = None

        def _lens(self):
            return (100, 50)

        def _internal(self, questions):
            return {"q": {"t": "noul", "ins": "?", "crit": None}}

    assert B().state_budget({"q": {}}) == {"q": 100 - 30 - 1}


def test_cached_tokenizer_returns_identical_ids_and_forwards_everything_else():
    """La cache del tokenizer devono essere trasparente, non una reimplementazione.

    `build_sequence` ricalcola l'intera testa (domanda + opzioni) per ogni stato,
    anche se dipende solo dalla domanda: a K=16 quella testa e' 95 dei 111 token e
    17 delle 18 chiamate sono identiche a quelle dello stato prima. Il prezzo della
    cache e' che un errore qui cambierebbe gli id in silenzio, quindi il test
    confronta l'output e verifica che gli attributi non intercettati passino
    attraverso (laya usa mask_token, cls/sep/pad e decode).
    """
    from layaadv.model.engine import CachedTokenizer

    calls = []

    class Tok:
        mask_token = "[MASK]"
        mask_token_id = 1
        cls_token_id = 0
        sep_token_id = 2
        pad_token_id = 3

        def __call__(self, text, **kw):
            calls.append(text)
            return {"input_ids": [len(w) for w in text.split()]}

        def decode(self, ids, skip_special_tokens=False):
            return " ".join(str(i) for i in ids)

    raw, tok = Tok(), CachedTokenizer(Tok())
    texts = ["a b c", "opzione_1", "opzione_1", "a b c", "d e"]
    first = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    second = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    assert first == second
    assert len(calls) == 3, "il testo ripetuto deve colpire la cache"

    # attributi non intercettati
    assert tok.mask_token == "[MASK]" and tok.mask_token_id == 1
    assert (tok.cls_token_id, tok.sep_token_id, tok.pad_token_id) == (0, 2, 3)
    assert tok.decode([1, 2]) == raw.decode([1, 2])
    # la lista restituita e' fresca: una caller che la muta non deve corrompere la cache
    got = tok("a b c", add_special_tokens=False)["input_ids"]
    got.append(999)
    assert tok("a b c", add_special_tokens=False)["input_ids"] == [1, 1, 1]


def test_cached_tokenizer_is_bounded():
    """La cache non puo' crescere senza limite: gli stati sono tutti distinti."""
    from layaadv.model.engine import CachedTokenizer

    class Tok:
        def __call__(self, text, **kw):
            return {"input_ids": [1]}

    tok = CachedTokenizer(Tok(), maxsize=4)
    for i in range(20):
        tok(f"stato {i}")
    assert len(tok._cache) <= 4
