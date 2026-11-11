"""typed-decisions: the official benchmark (gold standard of the study).

Loader for the frozen local copy in ``data/raw/typed-decisions/``. Field notes
and protocol: 30-dataset/typed-decisions-dataset.md.
"""

from __future__ import annotations

from pathlib import Path

from layaadv import hashing
from layaadv.data.raw import dataset_path, maybe_json, read_jsonl
from layaadv.schema import Case


def dataset_file(split: str) -> Path:
    return dataset_path("typed-decisions", f"{split}.jsonl")


def dataset_sha256(split: str) -> str:
    """Provenance hash logged with every experiment result (protocollo)."""
    return hashing.sha256_file(dataset_file(split))


def case_agreement(row: dict) -> float:
    """Share of questions where the teacher's argmax matches the gold label.

    Used to drop ambiguous examples **before** choosing attack targets
    (30-dataset/protocollo-dataset-valido.md: filtro_pre, dichiarato).
    """
    agreement = maybe_json(row["label_agreement"])
    return sum(1 for v in agreement.values() if v.get("argmax_agree")) / len(agreement)


def load_cases(
    split: str = "test",
    limit: int | None = None,
    min_agreement: float | None = None,
) -> list[Case]:
    """Load cases in file order (deterministic: resume and subsets stay stable).

    ``min_agreement`` filters ambiguous examples first, then ``limit`` caps the
    set — the declared order of the dataset protocol.
    """
    cases: list[Case] = []
    for row in read_jsonl(dataset_file(split)):
        if min_agreement is not None and case_agreement(row) < min_agreement:
            continue
        gold = maybe_json(row["gold"])
        cases.append(
            Case(
                id=str(row["id"]),
                state=row["state"],
                questions=maybe_json(row["questions"]),
                gold={qid: str(v["label"]) for qid, v in gold.items()},
            )
        )
        if limit is not None and len(cases) >= limit:
            break
    return cases


def subset_hash(cases: list[Case]) -> str:
    """Hash of the exact ids evaluated — proves baseline and attack used the
    same S (metriche-attacco: drop is relative to the same subset)."""
    return hashing.sha256_canonical_json([c.id for c in cases])
