"""Loader intent: le invarianti che rendono gli esperimenti credibili.

Un loader sbagliato non falla: restituisce metriche false. Percio' qui si
testano le garanzie, non la forma dei dati.
"""

from __future__ import annotations

import random

import pytest

from layaadv.data import loaders
from layaadv.schema import Case


def fake_engine_with_head(head_max_len: int = 256):
    """Stand-in che espone il *contratto* del backend, non i suoi attributi privati.

    `check_head_budget` ora chiede `head_text()` invece di scavare in `engine._agent`:
    l'attuale versione funzionava solo sul backend nativo, quindi `--model laya-onnx`
    moriva con un AttributeError a meta' dell'esperimento.
    """
    tok = _FakeTok()

    def build_sequence(active_tok, state, q, max_len, head_max_len, truncate_left=False):
        from laya.common import build_sequence as real

        return real(active_tok, state, q, max_len, head_max_len, truncate_left=truncate_left)

    class FakeEngine:
        def __init__(self):
            self.tok = tok

        def _lens(self):
            return 1024, head_max_len

        def _internal(self, questions):
            return {
                qid: {"t": q["type"], "ins": q["instructions"], "crit": q["criteria"]}
                for qid, q in questions.items()
            }

        def head_text(self, questions):
            out = {}
            for qid, q in self._internal(questions).items():
                ids, markers = build_sequence(self.tok, "", q, *self._lens())
                out[qid] = {
                    "head_tokens": len(ids) - 1,
                    "markers": len(markers),
                    "text": self.tok.decode(ids, skip_special_tokens=False),
                }
            return out

    return FakeEngine()


class _FakeTok:
    """Tokenizer stand-in: 1 token per word, decode joins on spaces."""

    mask_token = "[MASK]"
    mask_token_id = 1
    cls_token_id = 0
    sep_token_id = 2

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(i) for i in ids if isinstance(i, str))


def build_case(labels, gold, state="testo", instructions="Qual e' l'intento?"):
    crit = {lab: "" for lab in labels}
    return Case(
        id="0",
        state=state,
        questions={"intent": {"type": "choice", "instructions": instructions, "criteria": crit}},
        gold={"intent": gold},
    )


def test_choices_have_exactly_k_options_with_gold_inside():
    labels = [f"lab_{i}" for i in range(50)]
    rng = random.Random(0)
    for k in (2, 4, 16, 40):
        q = loaders.build_choices("lab_7", k, labels, rng=rng)
        assert len(q["criteria"]) == k
        assert "lab_7" in q["criteria"]


def test_choices_never_repeat_an_option():
    labels = [f"lab_{i}" for i in range(50)]
    q = loaders.build_choices("lab_0", 20, labels, rng=random.Random(1))
    assert len(set(q["criteria"])) == 20


def test_options_descriptions_are_empty_by_design():
    """Una descrizione uguale al label fa pagare il testo due volte e dimezza
    le opzioni che entrano nel budget: vedi E03 docstring."""
    q = loaders.build_choices("lab_0", 8, [f"lab_{i}" for i in range(20)], rng=random.Random(0))
    assert all(v == "" for v in q["criteria"].values())


def test_gold_position_varies_so_asr_cannot_be_inflated():
    """Se il gold sta sempre nella stessa posizione, un ASR alto potrebbe essere
    position bias invece che injection."""
    labels = [f"lab_{i}" for i in range(40)]
    positions = {
        list(loaders.build_choices(gold, 8, labels, rng=random.Random(gold))["criteria"]).index(gold)
        for gold in labels
    }
    assert len(positions) > 1


def test_option_set_does_not_depend_on_limit():
    """Il seed e' per riga: cambiare --limit non deve cambiare le domande, altrimenti
    un subset non e' confrontabile con la run piena."""
    labels = [f"lab_{i}" for i in range(30)]
    a = loaders.build_choices("lab_3", 8, labels, rng=random.Random("s:3:testo"))
    b = loaders.build_choices("lab_3", 8, labels, rng=random.Random("s:3:testo"))
    assert a == b


def test_k_beyond_label_count_is_refused():
    with pytest.raises(SystemExit):
        loaders.build_choices("lab_0", 50, ["lab_0", "lab_1"], rng=random.Random(0))


def test_check_head_budget_passes_when_everything_fits():
    cases = [build_case(["a", "b", "c"], "a")]
    audit = loaders.check_head_budget(fake_engine_with_head(), cases)
    assert audit["missing_options"] == 0 and audit["missing_gold"] == 0


def test_check_head_budget_aborts_when_the_gold_is_truncated():
    """Il caso per cui esiste la guardia: sopra budget laya tronca in silenzio e
    l'accuracy crollerebbe per un motivo che non e' il modello."""

    class TruncatingTok(_FakeTok):
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": ["blocco"]}  # ogni opzione sparisce

    engine = fake_engine_with_head()
    engine.tok = TruncatingTok()
    with pytest.raises(SystemExit, match="budget della testa"):
        loaders.check_head_budget(engine, [build_case(["a", "b"], "a")])


def test_check_head_budget_non_strict_reports_the_wall_instead_of_aborting():
    """Uno sweep su K deve potersi fermare al muro: il soffitto dipende da
    head_max_len e dal dataset, quindi non e' una costante che l'esperimento puo'
    hardcodare. strict=False deve restituire l'audit, non far perdere la run.

    Le opzioni sono lunghe di proposito: build_string tronca ciascuna a
    ``per = max(4, (head_max_len - 16) // n_option)`` token, quindi con nomi
    brevi la guardia non trippa mai e il test passerebbe senza verificare nulla.
    """
    long_label = lambda i: " ".join(f"lab{i}" for _ in range(8))  # noqa: E731
    engine = fake_engine_with_head(head_max_len=40)
    n_options = 8
    cases = [build_case([long_label(i) for i in range(n_options)], long_label(0))]
    audit = loaders.check_head_budget(engine, cases, strict=False)
    assert audit["over_budget"] is True
    # una per opzione troncata, piu' il gold
    assert audit["missing_options"] == n_options
    assert audit["missing_gold"] == 1


def test_check_head_budget_non_strict_is_falsy_when_everything_fits():
    engine = fake_engine_with_head()
    audit = loaders.check_head_budget(engine, [build_case(["a", "b"], "a")], strict=False)
    assert audit["over_budget"] is False


def test_unknown_dataset_name_is_refused():
    with pytest.raises(SystemExit):
        loaders.get_dataset("non-esiste")


def test_seed_changes_the_option_design_but_not_the_gold_position_rule():
    """`--seed` deve cambiare gli option set, altrimenti è un flag che non fa niente.

    Il seed entra nel `Random` per riga (`f"{seed}:{id}:{testo}"`), quindi un seed diverso
    ridisegna le opzioni. Il gold resta in posizione variabile per costruzione in entrambi i
    casi: è la proprietà che tiene valido l'ASR, e il seed non la tocca.
    """
    from layaadv.data import loaders

    a, _, _ = loaders.load_cases("massive-it", k=8, limit=50, seed=0)
    b, _, _ = loaders.load_cases("massive-it", k=8, limit=50, seed=1)
    assert [c.gold["intent"] for c in a] == [c.gold["intent"] for c in b]
    assert [c.state for c in a] == [c.state for c in b]
    changed = sum(
        1
        for x, y in zip(a, b)
        if list(x.questions["intent"]["criteria"]) != list(y.questions["intent"]["criteria"])
    )
    assert changed > 25, f"solo {changed}/50 option set cambiati: il seed non sta agendo"


def test_survives_is_a_whole_word_test_not_a_substring_test():
    """Il buco che il substring test lasciava aperto.

    `check_head_budget` decideva con `label not in decoded`: se laya taglia
    `email_query` a `email_` e nella testa c'e' anche `email_querycontact`, il
    taglio passava il controllo e l'audit dichiarava "0 opzioni perse" su una
    testa troncata. Con il confine a destra, un troncamento non si nasconde
    dentro un'opzione piu' lunga.
    """
    from layaadv.data.loaders import survives

    head = "opzione[MASK] email_querycontact[MASK] email_[SEP]"
    assert survives(head, "email_querycontact")          # intera, seguita da [MASK]
    assert not survives(head, "email_query")              # troncata, trovata solo dentro la lunga
    assert survives(head, "opzione")
    assert not survives(head, "email_querycontactx")


def test_survives_accepts_the_labels_with_a_colon_and_underscore():
    """Le label reali contengono `_` e, in E13, `:`: il confine non deve romperle."""
    from layaadv.data.loaders import survives

    head = "I7: calendar_query[MASK] weather_query[MASK] qa_maths[SEP]"
    for label in ("I7: calendar_query", "weather_query", "qa_maths"):
        assert survives(head, label), label
    assert not survives(head, "I7: calendar_querycontact")
