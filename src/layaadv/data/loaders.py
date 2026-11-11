"""Loader for intent datasets (MASSIVE it-IT, Banking77, ...) as ``Case``.

Why this exists
---------------
Every intent dataset is a (text, label) table, while laya wants
``(state, questions, gold)``. Converting once, here, keeps E02/E03 from
growing their own copy of the same mapping.

Two invariants, checked rather than assumed
-------------------------------------------
1. **Head budget.** ``head_max_len`` (256) covers question + criteria + all
   options. Past it laya keeps the option *markers* but silently truncates the
   option *text*, so accuracy falls for a reason that has nothing to do with
   the model. :func:`check_head_budget` refuses such a K instead of reporting a
   fake attack result. Measured on it-IT: K<=42 is safe with the worst-case
   intent names, K=44 already loses options.
2. **Gold present.** A truncated head can also drop the gold label itself
   (K>=40 on it-IT with duplicated descriptions). Same guard, same reason.

Option descriptions are left **empty** on purpose: laya renders an option as
``"label: description"``, so a description equal to the label pays for the same
text twice and halves the number of options that fit.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Sequence

from layaadv import hashing
from layaadv.data.raw import dataset_path, read_jsonl
from layaadv.schema import Case

QID = "intent"


@dataclass(frozen=True)
class IntentDataset:
    """An intent dataset addressed by slug, with its text and label columns."""

    slug: str
    file: str
    text_field: str
    label_field: str
    partition: str | None = None  # MASSIVE only: train/dev/test

    def rows(self) -> list[dict]:
        out = []
        for row in read_jsonl(dataset_path(self.slug, self.file)):
            if self.partition is not None and row.get("partition") != self.partition:
                continue
            out.append(row)
        return out

    def file_sha256(self) -> str:
        return hashing.sha256_file(dataset_path(self.slug, self.file))

    def subset_hash(self, cases: Sequence[Case]) -> str:
        return hashing.sha256_canonical_json([c.id for c in cases])


DATASETS: dict[str, IntentDataset] = {
    "massive-it": IntentDataset("massive-1.0", "it-IT.jsonl", "utt", "intent", partition="test"),
    "massive-en": IntentDataset("massive-1.0", "en-US.jsonl", "utt", "intent", partition="test"),
    # `label_text` = nome dell'intent. Con `label` (id intero) le opzioni sarebbero
    # "63", "13": il modello non ha niente su cui decidere e la baseline va a caso.
    "banking77": IntentDataset("banking77", "test.jsonl", "text", "label_text"),
    "ag-news": IntentDataset("ag-news", "test.jsonl", "text", "label"),
}


def get_dataset(name: str) -> IntentDataset:
    if name not in DATASETS:
        raise SystemExit(f"dataset sconosciuto: {name!r} (noti: {', '.join(sorted(DATASETS))})")
    return DATASETS[name]


def labels_of(ds: IntentDataset) -> list[str]:
    return sorted({str(r[ds.label_field]) for r in ds.rows()})


def build_choices(
    gold: str,
    k: int,
    labels: Sequence[str],
    *,
    rng: random.Random,
    instructions: str = "Qual è l'intento dell'utente?",
) -> dict:
    """A ``choice`` question with exactly K options, gold always among them.

    Distractors are sampled from the dataset's own label set (seeded, so the
    question is reproducible) rather than invented, and the description is left
    empty: see the module docstring.
    """
    others = [lab for lab in labels if lab != gold]
    if k - 1 > len(others):
        raise SystemExit(f"K={k} impossibile: il dataset ha solo {len(others) + 1} label")
    chosen = rng.sample(others, k - 1)
    options = chosen + [gold]
    rng.shuffle(options)  # gold position varies, so position bias cannot inflate ASR
    return {"type": "choice", "instructions": instructions, "criteria": {o: "" for o in options}}


def load_cases(
    name: str,
    *,
    k: int,
    limit: int | None = None,
    seed: int = 0,
    state_transform: Callable[[str], str] | None = None,
    question_transform: Callable[[dict, str], dict] | None = None,
) -> tuple[list[Case], IntentDataset, list[str]]:
    """Build ``Case`` objects for one K.

    ``state_transform``/``question_transform`` are the two hooks an attack uses;
    they receive the untouched record so the baseline and the adversarial run
    are provably the same subset. The gold is never modified by either.
    """
    ds = get_dataset(name)
    labels = labels_of(ds)
    cases: list[Case] = []
    for row in ds.rows():
        gold = str(row[ds.label_field])
        # Per-case seed: the option set must not depend on limit or on the
        # position of the row in the file, or a rerun with a different limit
        # would silently change the questions.
        rng = random.Random(f"{seed}:{row.get('id', gold)}:{text_of(row, ds)}")
        question = build_choices(gold, k, labels, rng=rng)
        if question_transform is not None:
            question = question_transform(question, gold)
        state = text_of(row, ds)
        if state_transform is not None:
            state = state_transform(state)
        cases.append(
            Case(
                id=str(row.get("id", len(cases))),
                state=state,
                questions={QID: question},
                gold={QID: gold},
            )
        )
        if limit is not None and len(cases) >= limit:
            break
    return cases, ds, labels


def text_of(row: dict, ds: IntentDataset) -> str:
    return str(row[ds.text_field])


def survives(head: str, label: str) -> bool:
    """Does ``label`` appear **whole** in the decoded head?

    A plain `label in head` is a substring test, and a substring test passes when the
    label was cut and the prefix happens to be inside a *longer* option: `email_query`
    truncated to `email_` would be found inside `email_querycontact` and the audit would
    report the option as intact. Requiring a non-word character after the match is what
    makes it a whole-word test: laya renders `[MASK] label [MASK] label ...`, so an
    uncut option is always followed by a special token.

    On MASSIVE-it the two labels are `email_query` and `email_querycontact` and the
    substring is the *shorter* one, which the uniform per-option token cap never cuts
    first, so this cannot fire there today. It fires on any dataset with a hierarchical
    label set, which is exactly the case a head-budget guard exists for.
    """
    import re

    return re.search(re.escape(label) + r"(?![0-9A-Za-z_])", head) is not None


def check_head_budget(engine, cases: Sequence[Case], sample: int = 200, strict: bool = True) -> dict:
    """Fail loudly when the option head does not fit ``head_max_len``.

    A truncated head makes accuracy collapse for a reason unrelated to the
    model, so an experiment built on it would measure nothing. Returns the
    audit numbers; raises SystemExit when options or the gold do not survive.

    With ``strict=False`` the audit is returned instead of raised, so a sweep
    over K can stop at the wall instead of losing the whole experiment: the
    ceiling depends on the checkpoint (``head_max_len`` 192 vs 256) and on the
    dataset, so it is not a constant an experiment can hardcode.

    It asks the engine for the head it *built* (``head_text``), not for the one the
    caller described: the check is about what the model reads, and it has to work
    the same on every backend.
    """
    checked = 0
    missing_options = 0
    missing_gold = 0
    max_head = 0
    for case in list(cases)[:sample]:
        qid = next(iter(case.questions))
        question = case.questions[qid]
        head = engine.head_text({qid: question})[qid]
        decoded = head["text"]
        max_head = max(max_head, head["head_tokens"])
        for label in question["criteria"]:
            if not survives(decoded, str(label)):
                missing_options += 1
        if not survives(decoded, str(case.gold[qid])):
            missing_gold += 1
        checked += 1

    head_max_len = engine._lens()[1]
    audit = {
        "checked_cases": checked,
        "head_max_len": head_max_len,
        "max_head_tokens": max_head,
        "missing_options": missing_options,
        "missing_gold": missing_gold,
    }
    if (missing_options or missing_gold) and strict:
        raise SystemExit(
            "K oltre il budget della testa: opzioni o gold verrebbero troncati da laya, "
            f"l'accuracy crollerebbe per un motivo che non e' il modello. {audit}. "
            "Riduci K."
        )
    audit["over_budget"] = bool(missing_options or missing_gold)
    return audit
