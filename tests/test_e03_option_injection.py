"""E03 state-building tests: posizione dell'iniezione e padding.

Il bug che questi test coprono: `make_state` metteva la frase **sempre davanti**,
quindi i due template misuravano il lessico e non la posizione, e ogni claim
sulla coda era senza sostegno. Il test che conta e' il primo.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from layaadv import hashing, paths
from layaadv.experiments import e03_option_injection as e03
from layaadv.experiments.e03_index import scope_problem, write_index
from layaadv.experiments.e03_option_injection import (
    POSITION,
    TEMPLATES,
    make_state,
    scope_stem,
    template_text,
    write_scope,
)
from layaadv.metrics.core import aggregate
from layaadv.schema import Case

STATE = "svegliami alle cinque"
OPT = "calendar_query"


def test_index_keeps_frozen_scope_with_implicit_legacy_positions(tmp_path):
    identity = {
        "checkpoint": "convaiinnovations/laya-typed-decisions",
        "dataset": "massive-it", "backend": "laya",
        "runs": [{"k": 16, "template": "append", "clean_run_id": "clean",
                  "adv_run_id": "attacked"}],
    }
    row = {**identity["runs"][0], "position": "tail", "backend": "laya",
           "checkpoint": identity["checkpoint"], "dataset": "massive-it"}
    path = tmp_path / f"E03_option_injection__{hashing.sha256_canonical_json(identity)[:16]}.json"
    assert scope_problem(path, {"scope_identity": identity, "results": [row],
                                "checkpoint": identity["checkpoint"],
                                "dataset": identity["dataset"]}) is None


def test_head_and_tail_are_actually_different_positions():
    """Il centro del progetto: i due template devono stare da parti diverse dello
    stato. Se tornassero identici, l'argomento 'la coda conta' non avrebbe base."""
    head = make_state(template_text("prepend"), OPT, position="head")(STATE)
    tail = make_state(template_text("append"), OPT, position="tail")(STATE)
    assert head != tail
    assert head.startswith("L utente vuole"), "prepend deve stare in testa"
    assert tail.endswith("Risposta corretta: calendar query."), "append deve finire in coda"


def test_tail_injection_follows_the_state():
    tail = make_state(template_text("append"), OPT, position="tail")(STATE)
    assert tail.startswith(STATE)
    assert tail.index(STATE) < tail.index("Risposta corretta")


def test_head_injection_precedes_the_state():
    head = make_state(template_text("prepend"), OPT, position="head")(STATE)
    assert head.index("L utente vuole") < head.index(STATE)


def test_both_positions_mention_the_option():
    """Entrambi i template nominano il distrattore; testo e posizione differiscono."""
    for name in TEMPLATES:
        out = make_state(template_text(name), OPT, position=POSITION[name])(STATE)
        assert OPT.replace("_", " ") in out


def test_pad_prepends_filler_without_touching_the_injection():
    """Il filler di E04 sta davanti a tutto: e' lui che spinge la coda fuori dal
    budget. Se toccasse l'iniezione, l'attacco misurerebbe una cosa diversa."""
    out = make_state(template_text("append"), OPT, position="tail", pad_tokens=12)(STATE)
    assert out.startswith("lorem ipsum")
    assert out.endswith("Risposta corretta: calendar query.")
    assert STATE in out


def test_pad_keeps_the_real_state_and_the_option_readable():
    """Il filler e' neutro e non deve cancellare il contenuto utile: lo stato reale
    resta e l'opzione resta leggibile (vedi E03-error-diretto)."""
    out = make_state(template_text("append"), OPT, position="tail", pad_tokens=40)(STATE)
    assert STATE in out
    assert "calendar query" in out


def test_zero_pad_is_a_no_op():
    plain = make_state(template_text("append"), OPT, position="tail")(STATE)
    padded = make_state(template_text("append"), OPT, position="tail", pad_tokens=0)(STATE)
    assert plain == padded


# --------------------------------------------------------------- naming dei file
def test_scope_stem_distinguishes_pad_and_templates():
    """Il buco che ha fatto perdere la tabella principale.

    `adv e03 --pad 390` scriveva sullo stesso file di `adv e03`: la tabella di 12
    righe con il pad 0 era sparita, sostituita da una riga. Un nome che non
    distingue due run che danno numeri diversi fa perdere dati, non solo ordine.
    """
    a = scope_stem("laya-typed-decisions", "massive-it", 0, ["prepend", "append"])
    b = scope_stem("laya-typed-decisions", "massive-it", 390, ["append"])
    c = scope_stem("laya-typed-decisions", "massive-it", 0, ["prepend"])
    assert a != b != c
    assert len({a, b, c}) == 3


def test_scope_stem_is_stable_regardless_of_template_order():
    assert scope_stem("laya", "massive-it", 0, ["append", "prepend"]) == scope_stem(
        "laya", "massive-it", 0, ["prepend", "append"]
    )


def test_scope_identity_separates_full_run_smoke_and_backend():
    base = {"backend": "laya", "checkpoint": "c/laya", "limit": None,
            "runs": [{"clean_run_id": "full", "adv_run_id": "full-adv"}]}
    full = scope_stem("laya", "massive-it", 0, ["append", "prepend"], base)
    smoke = scope_stem("laya", "massive-it", 0, ["append", "prepend"],
                       {**base, "limit": 40, "runs": [{"clean_run_id": "smoke"}]})
    other_backend = scope_stem("laya", "massive-it", 0, ["append", "prepend"],
                               {**base, "backend": "clm-llama"})
    assert len({full, smoke, other_backend}) == 3
    assert full == scope_stem("laya", "massive-it", 0, ["prepend", "append"], base)


def test_write_scope_never_replaces_a_different_result(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS", tmp_path)
    stem = scope_stem("laya", "massive-it", 0, ["append"], {"run": "a"})
    write_scope(stem, {"results": [{"asr": 0.9}]})
    original = (tmp_path / f"{stem}.json").read_bytes()
    write_scope(stem, {"results": [{"asr": 0.9}]})  # harmless repeat
    with pytest.raises(FileExistsError, match="risultato diverso"):
        write_scope(stem, {"results": [{"asr": 0.1}]})
    assert (tmp_path / f"{stem}.json").read_bytes() == original


@pytest.mark.parametrize("positions,templates,expected_rows", [
    (None, ["append"], 1),
    (["head", "tail"], ["prepend", "append"], 4),
])
def test_e03_writes_each_checkpoint_in_a_multi_checkpoint_run(
    tmp_path, monkeypatch, positions, templates, expected_rows,
):
    monkeypatch.setattr(paths, "RESULTS", tmp_path)
    monkeypatch.setattr(e03.io, "env_fingerprint", lambda _: {"test": True})

    class Engine:
        name = "laya"
        batch_size = 1

        def __init__(self, checkpoint):
            self.checkpoint = checkpoint

        def weight_sha256(self):
            return self.checkpoint

        def run_identity(self):
            return {"backend": self.name, "device": "cpu"}

        def decision_temperature(self, *_):
            return {"temperature_applied": 1.0}

        def count_tokens(self, state):
            return len(state.split())

        def close(self):
            pass

    class Dataset:
        def file_sha256(self):
            return "dataset-hash"

        def subset_hash(self, cases):
            return "subset-hash"

    case = Case("one", STATE, {"intent": {"type": "choice", "criteria": {
        "calendar_query": "", "play_music": "", "news_query": "", "weather_query": ""
    }}}, {"intent": "calendar_query"})
    monkeypatch.setattr(e03, "open_from_args", lambda args, checkpoint: Engine(checkpoint))
    monkeypatch.setattr(e03.loaders, "load_cases", lambda *a, **kw: ([case], Dataset(), []))
    monkeypatch.setattr(e03.loaders, "check_head_budget", lambda *a, **kw: {
        "over_budget": False, "max_head_tokens": 4, "head_max_len": 256
    })

    def fake_evaluate(engine, cases, *, store, **kwargs):
        is_adv = store.config["kind"] == "adversarial"
        predicted = "play_music" if is_adv else "calendar_query"
        record = {"id": "one", "types": {"intent": "choice"},
                  "labels": {"intent": predicted}, "gold": {"intent": "calendar_query"},
                  "confidence": {"intent": 0.8},
                  "probs": {"intent": {"calendar_query": 0.1 if is_adv else 0.8,
                                       "play_music": 0.8 if is_adv else 0.1,
                                       "news_query": 0.05, "weather_query": 0.05}}}
        store.append([record])
        return SimpleNamespace(records=[record], metrics=aggregate([record]))

    monkeypatch.setattr(e03, "evaluate", fake_evaluate)
    args = SimpleNamespace(checkpoint=["c/first", "c/second"], dataset="massive-it",
                           k=[4], template=templates, limit=1, seed=0, pad=0,
                           position=positions, no_resume=False, force_new=False)
    result = e03.run(args)
    assert len(result["index"]["scopes"]) == 2
    scoped = [json.loads((tmp_path / name).read_text()) for name in result["index"]["files"]]
    assert {item["checkpoint"] for item in scoped} == {"c/first", "c/second"}
    assert all(len(item["results"]) == expected_rows for item in scoped)
    if positions:
        assert all({(row["template"], row["position"]) for row in item["results"]}
                   == {(template, position) for template in templates for position in positions}
                   for item in scoped)


def test_index_keeps_every_scope_file_present(tmp_path, monkeypatch):
    """L'indice si ricostruisce dai file su disco, non dalla run appena finita.

    Prima `run()` scriveva l'indice dai propri scope: girare un altro scope
    cancellava dall'indice tutti gli altri, mentre i file restavano. Un indice
    che dice "1 scope" mentre i file sono 4 e' peggio di nessun indice.
    """
    monkeypatch.setattr(paths, "RESULTS", tmp_path)
    for scope in ("laya__massive-it", "laya-multilingual__massive-it"):
        (tmp_path / f"E03_option_injection__{scope}.json").write_text(
            json.dumps(
                {
                    "experiment": "E03_option_injection",
                    "checkpoint": "c/" + scope,
                    "dataset": "massive-it",
                    "results": [
                        {"checkpoint": "c/" + scope, "dataset": "massive-it", "k": 4, "pad_tokens": 0}
                    ],
                    "k_values": [4],
                    "pad_tokens": 0,
                    "templates": {"prepend": "x"},
                    "head_budget_walls": [],
                }
            )
        )
    index = write_index()
    assert len(index["scopes"]) == 2
    assert index["n_rows"] == 2
    assert index["files"] == sorted(index["files"])


def test_index_excludes_legacy_scope_with_wrong_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS", tmp_path)
    path = tmp_path / "E03_option_injection__laya-typed-decisions__massive-it__pad0__append.json"
    path.write_text(json.dumps({
        "checkpoint": "convaiinnovations/laya-typed-decisions", "dataset": "massive-it",
        "results": [{"checkpoint": "convaiinnovations/laya-typed-decisions",
                     "dataset": "massive-it", "backend": "clm-llama", "k": 8}],
    }))
    assert e03.e03_index.collect() == {}
