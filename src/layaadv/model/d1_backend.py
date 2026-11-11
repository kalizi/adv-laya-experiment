"""Liquid AI `d1` decision models: a verbalizer readout, the fourth family.

Why this is a separate module
-----------------------------
`OnnxLayaBackend` and `OnnxPairBackend` live in `backends.py` because they
inherit :class:`LayaTokenBudget`, whose every method either imports
`laya.agent` or encodes laya's marker-head semantics. `d1` has neither: it is a
causal LFM backbone that reads the answer off a **softmax over single-token
option codes** (`prompt.py`'s `aliases`, `readout`). Inheriting the budget class
here would leave a laya fallback reachable on a model that has no marker head,
which is the same reason `clm_backend` is separate.

The prompt is **not reimplemented**. The rendering is reached through the
model's own `engine` (a `SystemOne`) and its sibling `prompt` module, looked up
from the dynamically loaded remote code. Two copies of that contract would drift
silently, and the drift would look like a model behaving differently.

What is genuinely different, and belongs in every published number
-----------------------------------------------------------------
1. **The answer channel is one token per option.** `readout` scores the logits
   of `A`, `B`, `C`... only, so an option cannot be silently truncated out of a
   head the way laya's can: there is no head. The failure mode that replaces it
   is an **alias collision**, and :meth:`alias_audit` measures it instead of
   assuming it.
2. **There is no state budget.** `system_one` encodes the whole prompt with no
   truncation, so E04 (the truncation attack) has no target here and
   `state_budget` reports the context length, not a clip. A run of E04 on this
   backend would measure the absence of a guard, not a defence.
3. **`confidence` is `max(p)` for every k** (`api.py`'s `answer`), which is
   laya's `k == 2` definition applied to a 16-way `choice`. An `OER@0.75` read
   on this backend and one read on laya are therefore **not the same quantity**,
   which is the defect `metrics.core.oer` already documents; the comparable
   metric is `oer_margin`, which reads `probs`.
4. **The label the dataset names is not the string the model reads.**
   `prompt._option_line` renders a label as `label.replace("_", " ")`, so
   `weather_query` reaches the model as `weather query`. Anything that audits a
   head by searching for the raw label will report every option as missing.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Sequence

from layaadv import hashing
from layaadv.schema import Case, Prediction, answer_label

DEFAULT_CHECKPOINT = "LiquidAI/d1-3B"


class D1Backend:
    """`system_one`/`system_one_batch` behind the repo's `DecisionBackend`.

    Deliberately **not** a subclass of `LayaTokenBudget`: see the module
    docstring. Every method the protocol asks for is implemented here, so a
    missing piece fails at construction with its own name rather than with an
    `AttributeError` in the middle of a run.
    """

    name = "d1-liquid"

    def __init__(
        self,
        checkpoint: str | None = None,
        device: str | None = None,
        batch_size: int = 16,
        compile: bool = False,
        fast: bool = False,
        quantize: str = "none",
        variant: str = "fp32",
        **_: Any,
    ) -> None:
        # Simmetrico al rifiuto di `quantize` su deberta-onnx e di `variant` sulla
        # famiglia laya: ogni backend conosce una sola maniera di scegliere la
        # precisione, e qui e' il dtype del checkpoint. Accettare un selettore che
        # non fa nulla produrrebbe una tabella di "la precisione non cambia niente"
        # costruita a mano da un errore di plumbing.
        if quantize not in (None, "", "none"):
            raise ValueError(
                "d1-liquid non ha un selettore --quantize: la precisione e' il dtype del "
                "checkpoint (bf16). Esiste un export W8A8 separato, che e' un altro "
                "checkpoint e non un knob. --quantize vale solo per la famiglia laya."
            )
        if variant not in (None, "", "fp32"):
            raise ValueError(
                "d1-liquid non ha una famiglia di grafi: --variant vale solo per "
                "deberta-onnx (l'export open-jev)."
            )
        # `--compile` e `--fast` su laya significanoTileLang / torch.compile
        # sull'encoder, con numeri misurati (1.37x, 2.42x). d1 ha un flag `compile`
        # proprio dentro `SystemOne`, con CUDA graph e un meccanismo diverso:
        # accettare quello della CLI qui sarebbe un flag che lavora su un'altra cosa.
        if compile:
            raise ValueError(
                "--compile e' il percorso di laya (torch.compile sull'encoder, 1.37x "
                "misurato). d1-liquid ha il proprio `compile` dentro SystemOne, che non "
                "e' esposto: un run compilato e uno eager su questo backend non sarebbero "
                "confrontabili e il flag non lo direbbe."
            )
        if fast:
            raise ValueError(
                "--fast e' il percorso TileLang di laya (2.42x col batch scarso). "
                "d1-liquid non ha kernel fusi: il flag non farebbe niente."
            )
        if checkpoint and checkpoint.startswith("convaiinnovations/laya"):
            raise ValueError(
                "d1-liquid ha ricevuto un checkpoint Laya: passa --checkpoint "
                f"{DEFAULT_CHECKPOINT} (o un path locale con i pesi d1)"
            )

        from transformers import AutoModel  # lazy: importare layaadv resta torch-free

        self.checkpoint = checkpoint or DEFAULT_CHECKPOINT
        self.batch_size = batch_size
        self.load_warnings: list[str] = []
        self._weight_sha256: str | None = None
        self._device_seen = str(device or ("cuda" if _cuda() else "cpu"))

        self._model = AutoModel.from_pretrained(
            self.checkpoint, trust_remote_code=True, dtype="auto"
        ).to(self._device_seen)
        self._model.eval()
        if not hasattr(self._model, "system_one"):
            raise ValueError(
                f"{self.checkpoint} non espone `system_one`: non e' un modello decisionale "
                "d1. `--model d1-liquid` accetta solo checkpoint della famiglia d1 "
                "(LiquidAI/d1-3B, LiquidAI/d1-omni-600M)."
            )
        self._engine = self._model.engine
        self._prompt = _prompt_module(self._engine)

    # --------------------------------------------------------------- protocollo
    @property
    def cfg(self) -> dict:
        return {
            "max_len": self._max_len(),
            "head_max_len": self._max_len(),
            "device": self.runtime_state()["device"],
            "dtype": str(next(self._model.parameters()).dtype),
            "model_id_or_path": self.checkpoint,
        }

    def _max_len(self) -> int:
        """The context length, read from the checkpoint and not assumed.

        32768 for d1-3B is declared in `text_config.max_position_embeddings`. A
        hardcoded constant would be wrong on the omni sibling (16384) and would
        silently make `state_budget` a number about a model that is not loaded.
        """
        cfg = getattr(self._model, "config", None)
        text = getattr(cfg, "text_config", None) or cfg
        for key in ("max_position_embeddings", "max_length"):
            value = getattr(text, key, None)
            if value:
                return int(value)
        return 32768

    def _lens(self) -> tuple[int, int]:
        """"(context, head room). d1 has no separate head: see the module docstring.

        Returning the context length twice says the truth a caller needs: the
        question block is never clipped, because nothing is clipped. Reporting a
        small `head_max_len` here would make `check_head_budget` invent a wall
        that does not exist on this backend.
        """
        n = self._max_len()
        return n, n

    def count_tokens(self, text: str) -> int:
        return len(self._engine.tokenizer(text, add_special_tokens=False)["input_ids"])

    def render_options(self, questions: dict[str, Any]) -> dict[str, list[str]]:
        """Option labels as the model reads them, in the question's own order.

        `weather_query` comes back as `weather query`: that is what the prompt
        contains (`prompt._option_line`). Returning the raw label here would
        describe a string the model never sees. The line builder is d1's own, so
        this stays a view and not a second copy of the rendering.
        """
        tok = self._engine.tokenizer
        style = self._engine.option_style
        out: dict[str, list[str]] = {}
        for qid, qdef in questions.items():
            q = self._prompt.as_question(qdef)
            if q.type == "choice":
                codes = self._prompt.aliases(tok, list(q.criteria.keys()))
                out[qid] = [
                    self._prompt._option_line(codes[i][0], lab, desc, style)
                    for i, (lab, desc) in enumerate(q.criteria.items())
                ]
            elif q.type == "score":
                out[qid] = [f"{i} {name}" for i, name in enumerate(q.criteria)]
            else:
                out[qid] = [str(q.criteria.get(k, k)) for k in ("false", "true")] if q.criteria else []
        return out

    def predict(self, case: Case) -> Prediction:
        return self.predict_batch([case])[0]

    def predict_batch(self, cases: list[Case], batch_size: int | None = None) -> list[Prediction]:
        """Score many cases, `batch_size` requests per `system_one_batch` call.

        `system_one_batch` packs the requests it is given into token-budgeted
        trees, so the grouping by identical questions that laya needs is not
        needed here: one state and all of its questions are one request, which is
        what `runner.run` already does for multi-question states.
        """
        if not cases:
            return []
        size = batch_size or self.batch_size
        out: list[Prediction] = []
        for start in range(0, len(cases), size):
            part = cases[start : start + size]
            replies = self._model.system_one_batch(
                [(case.state, case.questions) for case in part]
            )
            if len(replies) != len(part):
                raise RuntimeError(
                    f"d1 ha restituito {len(replies)} risposte per {len(part)} casi"
                )
            for case, reply in zip(part, replies):
                answers = {
                    qid: _normalize(ans) for qid, ans in reply["answers"].items()
                }
                missing = set(case.questions) - set(answers)
                if missing:
                    raise RuntimeError(
                        f"d1 non ha risposto a {sorted(missing)} per il caso {case.id!r}: "
                        "un accuracy calcolata su un sottoinsieme silenzioso e' il buco "
                        "che il backend ONNX di laya aveva gia' avuto"
                    )
                out.append(
                    Prediction(
                        id=case.id,
                        answers=answers,
                        labels={qid: answer_label(a) for qid, a in answers.items()},
                        usage=dict(reply.get("usage") or {}),
                    )
                )
        return out

    def close(self) -> None:
        model = getattr(self, "_model", None)
        if model is not None:
            del model
        self._model = None
        self._engine = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - chiudere non deve mai rompere una run
            pass

    # ------------------------------------------------------------------ budget
    def state_budget(self, questions: dict[str, Any]) -> dict[str, int]:
        """Token room for the state, per question: context minus the prompt.

        This is a **room**, not a clip: d1 encodes the whole prompt and never
        truncates it, so the number says how much fits before the context ends
        and nothing else. A caller that treats it as a truncation threshold (E04)
        is measuring a guard this backend does not have.
        """
        out: dict[str, int] = {}
        for qid, qdef in questions.items():
            out[qid] = max(0, self._max_len() - self._question_tokens(qdef))
        return out

    def _question_tokens(self, qdef: dict) -> int:
        """Tokens of the whole prompt with an empty state: the fixed cost."""
        q = self._prompt.as_question(qdef)
        prompt = self._prompt.render(self._engine.tokenizer, "", q, self._engine.bos,
                                     self._engine.lead, self._engine.state_style,
                                     self._engine.system, self._engine.option_style)
        return self.count_tokens(prompt)

    def probe_sequence(
        self, state: str, questions: dict[str, Any], *, marker: str, truncate_left: bool = False
    ) -> list[dict]:
        """The built prompt per question, and whether the marker survived it.

        d1 never truncates, so `marker_visible` stays True until the context
        length; `truncated` is reported rather than the flag being silently
        ignored, because a probe that drops its own parameter would look like it
        had been honoured.
        """
        rows = []
        for qid, qdef in questions.items():
            q = self._prompt.as_question(qdef)
            prompt = self._prompt.render(self._engine.tokenizer, state, q, self._engine.bos,
                                         self._engine.lead, self._engine.state_style,
                                         self._engine.system, self._engine.option_style)
            ids = self._engine.tokenizer.encode(prompt, add_special_tokens=False)
            rows.append(
                {
                    "qid": qid,
                    "seq_tokens": len(ids),
                    "state_tokens": self.count_tokens(state),
                    "option_markers": len(qdef.get("criteria") or ()),
                    "marker_visible": marker in self._engine.tokenizer.decode(ids),
                    "truncated": False,
                }
            )
        return rows

    def probe_head(self, questions: dict[str, Any]) -> list[dict]:
        """The question block in tokens, and its options' code map.

        `head_fits` is always True for the same reason `state_budget` is a room:
        there is no head to clip. The field is kept because the protocol asks for
        it and E00 reads it; `codes` is the audit that means something here.
        """
        rows = []
        for qid, qdef in questions.items():
            tokens = self._question_tokens(qdef)
            rows.append(
                {
                    "qid": qid,
                    "head_max_len": self._max_len(),
                    "built_head_tokens": tokens,
                    "option_markers": len(qdef.get("criteria") or ()),
                    "head_fits": True,
                    "codes": dict(self._codes(qdef)),
                }
            )
        return rows

    def head_text(self, questions: dict[str, Any]) -> dict[str, dict]:
        """The prompt the model actually reads, with an empty state.

        `loaders.check_head_budget` searches the decoded text for the raw label.
        On this backend the label is rendered with its underscores replaced, so
        that audit reports every option as missing for a reason that is a
        rendering convention and not a clipped head: E24 records
        :meth:`alias_audit` instead, and says why.
        """
        out: dict[str, dict] = {}
        for qid, qdef in questions.items():
            q = self._prompt.as_question(qdef)
            prompt = self._prompt.render(self._engine.tokenizer, "", q, self._engine.bos,
                                         self._engine.lead, self._engine.state_style,
                                         self._engine.system, self._engine.option_style)
            out[qid] = {
                "head_tokens": self.count_tokens(prompt),
                "markers": len(qdef.get("criteria") or ()),
                "text": prompt,
            }
        return out

    # ------------------------------------------------------- audit specifico d1
    def _codes(self, qdef: dict) -> list[tuple[str, int]]:
        """(code, token_id) per option, exactly as `readout_ids` builds them."""
        q = self._prompt.as_question(qdef)
        if q.type != "choice":
            return []
        return self._prompt.aliases(self._engine.tokenizer, list(q.criteria.keys()))

    def alias_audit(self, questions: dict[str, Any]) -> dict[str, dict]:
        """Does every option get a distinct single-token code?

        This is the failure mode that replaces laya's head truncation here.
        `prompt._aliases_cached` guarantees distinctness by construction (it
        refuses to reuse a token id), and raises when the pool runs out, so a
        collision cannot be silent. What *can* be surprising is how many options
        a code set holds: 26 letters, then two-digit, then lowercase, then
        `#i`, then letter pairs. Measuring it per K is what makes a K curve on
        this backend readable, because a K that changes the code alphabet also
        changes what "nominate the option" means.
        """
        out: dict[str, dict] = {}
        for qid, qdef in questions.items():
            codes = self._codes(qdef)
            out[qid] = {
                "n_options": len(codes),
                "n_distinct_codes": len({c for c, _ in codes}),
                "n_distinct_tokens": len({t for _, t in codes}),
                "code_kind": _code_kind(codes),
                "codes": {c: t for c, t in codes},
            }
        return out

    def decision_temperature(self, primitive: str, k: int) -> dict:
        """d1 applies no temperature: the readout is a log-softmax over option codes.

        Declared rather than omitted, because every other backend logs a
        temperature per K (`temp_bucket` switches at K=5/10/11 on laya) and a
        curve that crosses a bucket change compares numbers produced under
        different sharpness. Here there is no bucket and nothing to switch: the
        same row on every K says so explicitly.
        """
        return {
            "primitive": primitive,
            "options": k,
            "bucket": None,
            "temperature_raw": None,
            "temperature_applied": 1.0,
            "note": "d1 non applica temperatura: il readout e' un log-softmax "
            "sui logit dei soli token-codice delle opzioni",
        }

    # ----------------------------------------------------------- riproducibilita
    def runtime_state(self) -> dict:
        """Where the model is right now, and in what precision.

        Same contract as the native backend: the latency samples take the device
        from here, not from `env_fingerprint`, which would report a GPU the model
        is not on.
        """
        return {
            "device": self._device_seen,
            "dtype": str(next(self._model.parameters()).dtype),
        }

    def run_identity(self) -> dict:
        """What changes the numbers, for a run config.

        `confidence_definition` is the load-bearing entry: it is `max(p)` for
        **every** k, which is laya's `k == 2` definition applied to a 16-way
        `choice`. Without it an `OER@0.75` on this backend and one on laya look
        like the same number and are not. `oer_margin` is the comparable metric.
        """
        return {
            "backend": self.name,
            # d1 non ha i percorsi `compile`/`fast` di laya e li rifiuta: dichiararli
            # False qui dice che la run non li ha usati, non che non esistono.
            "compiled": False,
            "fast": False,
            "confidence_definition": "max(p) for every k (d1 reads the answer off one "
            "softmax over the option-code tokens)",
            "temperature_applied": None,
            "quantization": str(next(self._model.parameters()).dtype),
            "readout": "verbalizer: single-token option codes, softmax over those logits only",
            # Il device va nell'identita': su CPU lo stesso checkpoint non darebbe gli
            # stessi numeri, e un resume leggerebbe le previsioni dell'altro.
            "device": self._device_seen,
            "state_truncated": False,
        }

    def fingerprint(self) -> dict:
        return {
            "checkpoint": self.checkpoint,
            "cfg": self.cfg,
            "device": self.runtime_state()["device"],
            "weight_path": str(self._weight_path()),
            "weight_sha256": self.weight_sha256(),
        }

    def _weight_path(self) -> Path | None:
        """The safetensors file of the loaded snapshot."""
        base = Path(getattr(self._model.config, "_name_or_path", "") or self.checkpoint)
        for cand in (base / "model.safetensors",):
            if cand.exists():
                return cand
        hits = sorted(base.glob("*.safetensors")) if base.exists() else []
        return hits[0] if hits else None

    def weight_sha256(self) -> str | None:
        if self._weight_sha256 is None:
            path = self._weight_path()
            if path is None:
                return None
            self._weight_sha256 = hashing.sha256_file(path)
        return self._weight_sha256

    # ------------------------------------------------------------------ batching
    def batching_report(self, cases: Sequence[Case]) -> dict:
        """How the batching really went, outside the `run_id` (see `evaluate`).

        d1's real batching is the token-budgeted tree inside `system_one_batch`,
        so unlike the massive-it case on laya this is not inert: requests are
        packed by tokens, not by shape. `batch_size` is the number of requests
        per call, which is the knob the CLI exposes and the one in the config.
        """
        states = [c for c in cases]
        groups = {}
        for case in states:
            groups.setdefault(_question_key(case.questions), []).append(case.id)
        return {
            "question_groups": len(groups),
            "max_states_per_group": max((len(v) for v in groups.values()), default=0),
            "batching_is_inert": len(groups) <= 1 and len(states) <= 1,
            "token_budget": getattr(self._engine, "token_budget", None),
            "note": "d1 impacchetta le richieste su alberi di token: il batch e' "
            "reale, a differenza di massive-it su laya dove ogni caso e' un "
            "gruppo singleton",
        }


# ----------------------------------------------------------------------- helpers
def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def _prompt_module(engine: Any):
    """The `prompt` module of the remote code that is already loaded.

    Reached through the engine's own package rather than reimplemented: the
    rendering is the contract the decision weights were trained on, and a second
    copy would drift into a model that reads a different prompt than the one
    measured.
    """
    package = type(engine).__module__.rsplit(".", 1)[0]
    return importlib.import_module(f"{package}.prompt")


def _question_key(questions: dict) -> str:
    return hashing.sha256_canonical_json(questions)


def _code_kind(codes: Sequence[tuple[str, int]]) -> str:
    """Which alphabet the codes come from, for a K curve to be readable.

    A K that crosses 26 changes the alphabet (letters -> two digits -> ...), and
    with it what "nominate the option" can even mean.
    """
    if not codes:
        return "none"
    first = codes[0][0]
    if len(first) == 1 and first.isalpha() and first.isupper():
        return "letters"
    if first.isdigit():
        return "digits"
    return first


def _normalize(answer: dict) -> dict:
    """d1's typed answer -> the shape `schema` and the metrics read.

    One gap is closed here rather than left to `record_for`: `noul` comes back
    with `noul` only, so `confidence` would default to 0.0 and every OER on that
    primitive would be zero by construction. The pair is written out too, so the
    answer is self-describing like laya's.
    """
    out = dict(answer)
    if out.get("type") == "noul":
        p = float(out["noul"])
        out.setdefault("confidence", round(max(p, 1.0 - p), 4))
        out.setdefault("probabilities", {"false": round(1.0 - p, 4), "true": round(p, 4)})
    return out
