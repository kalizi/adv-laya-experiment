"""Loader typed-decisions: parsing dei campi JSON-encoded e filtri deterministici."""

from __future__ import annotations

import pytest

from layaadv import paths
from layaadv.data import typed_decisions as td
from layaadv.data.raw import maybe_json, read_jsonl

HAS_DATA = (paths.RAW / "typed-decisions" / "test.jsonl").exists()
pytestmark = pytest.mark.skipif(not HAS_DATA, reason="data/raw/typed-decisions non scaricato")


def test_load_cases_parses_json_encoded_fields():
    cases = td.load_cases("test", limit=5)
    assert len(cases) == 5
    first = cases[0]
    assert isinstance(first.questions, dict) and isinstance(first.gold, dict)
    assert set(first.gold) == set(first.questions)
    assert all(isinstance(v, str) for v in first.gold.values())


def test_load_cases_is_deterministic_and_capped():
    a = [c.id for c in td.load_cases("test", limit=20)]
    b = [c.id for c in td.load_cases("test", limit=20)]
    assert a == b == list(dict.fromkeys(a))  # ordine del file, id unici


def test_case_agreement_is_a_share_of_questions():
    rows = list(read_jsonl(td.dataset_file("test")))[:20]
    for row in rows:
        assert 0.0 <= td.case_agreement(row) <= 1.0


def test_min_agreement_filters_before_limit():
    """Protocollo: filtro dichiarato PRIMA del limite (no selection bias)."""
    rows = [r for r in read_jsonl(td.dataset_file("test")) if td.case_agreement(r) >= 0.8]
    expected = [str(r["id"]) for r in rows]
    assert expected  # il filtro non deve azzerare il set

    assert [c.id for c in td.load_cases("test", min_agreement=0.8)] == expected
    assert [c.id for c in td.load_cases("test", min_agreement=0.8, limit=5)] == expected[:5]


def test_subset_hash_changes_with_subset():
    small = td.load_cases("test", limit=10)
    other = td.load_cases("test", limit=11)
    assert td.subset_hash(small) != td.subset_hash(other)
    assert td.subset_hash(small) == td.subset_hash(td.load_cases("test", limit=10))


def test_maybe_json_accepts_str_and_dict():
    assert maybe_json('{"a": 1}') == {"a": 1}
    assert maybe_json({"a": 1}) == {"a": 1}
