"""Data contract shared by the engine, the metrics and the experiments.

Pure types + pure functions: no model, no IO. Everything downstream agrees on
what a "label" and a "probability vector" are, defined exactly once here
(see 40-esperimenti/E01-baseline.md for the per-primitive decision rules).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Case:
    """One evaluable example: a state, its questions and the gold answers."""

    id: str
    state: str
    questions: dict[str, Any]
    gold: dict[str, str]  # {qid: gold label} — no extra fields


@dataclass
class Prediction:
    """Model output for one case, normalized for scoring."""

    id: str
    answers: dict[str, Any]  # raw Laya answer dicts, untouched
    labels: dict[str, str]  # {qid: predicted label}
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def confidence(self) -> dict[str, float]:
        """{qid: confidence as reported by the model (post-temperature)."""
        return {qid: float(a.get("confidence", 0.0)) for qid, a in self.answers.items()}


def answer_label(answer: dict[str, Any]) -> str:
    """Raw answer -> comparable label.

    Decision rules (E01, identical for baseline and adversarial runs):
      - choice: argmax of the option probabilities
      - noul:   P(true) >= 0.5
      - score:  argmax over the level probabilities
    """
    qtype = answer.get("type")
    if qtype == "choice":
        return str(answer["choice"])
    if qtype == "noul":
        return "true" if float(answer["noul"]) >= 0.5 else "false"
    if qtype == "score":
        probs = answer["probabilities"]
        return str(max(probs, key=lambda k: float(probs[k])))
    raise ValueError(f"unknown question type: {qtype!r}")


def answer_probs(answer: dict[str, Any]) -> dict[str, float]:
    """Raw answer -> probability distribution over labels (for Brier/ECE)."""
    qtype = answer.get("type")
    if qtype == "noul":
        p = float(answer["noul"])
        return {"false": 1.0 - p, "true": p}
    if qtype in ("choice", "score"):
        return {str(k): float(v) for k, v in answer["probabilities"].items()}
    raise ValueError(f"unknown question type: {qtype!r}")


def primitive_of(answer: dict[str, Any]) -> str:
    """Which primitive produced this answer: choice | noul | score."""
    return str(answer.get("type"))
