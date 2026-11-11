"""Isolation layer around the Laya model.

This is the **only** module allowed to import ``laya`` / touch its internals
(``_to_internal``, ``_check_question``, ``build_sequence``, the Router). Every
experiment, attack and metric goes through :class:`LayaEngine`, so when the
upstream API shifts, only this file changes.

Batching contract
-----------------
``predict_batch`` groups cases by question signature and issues one Laya
``predict_batch`` per group. On typed-decisions that pays: the 400 test states
share 4 question schemas, so a baseline run costs 4 forward passes instead of
400. On the datasets the attack sweeps use it does **not** pay, because
``loaders.load_cases`` gives each case its own option set (per-case rng), every
group holds one state, and ``batch_size`` has no effect at all: laya applies a
single ``questions`` dict to all the states of a call, so states can only share
a forward pass when their questions are identical. ``batching_report`` records
which of the two happened, because the number in the run config would otherwise
describe a knob that did nothing.

Laya's OOM handler silently moves the model to CPU, so after every batch we
verify the device did not change and scream if it did.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Sequence

from layaadv import hashing
from layaadv.schema import Case, Prediction, answer_label

# Primary target of the study (see 10-modello/checkpoint-laya.md).
DEFAULT_CHECKPOINT = "convaiinnovations/laya-typed-decisions"

_CFG_KEYS = ("max_len", "head_max_len", "device", "dtype", "model_id_or_path")


QUANTIZATION = ("none", "int8", "nf4")

# I sottoinsiemi di marker che `marker_patch_profiles` sa ripristinare. "all" e'
# l'intero insieme, "gold" e' il marker dell'opzione corretta, "target" quello
# dell'opzione che la frase nomina (l'indice lo passa il chiamante, perche' dipende
# dall'attacco e non dal caso).
_MARKER_SUBSETS = ("all", "gold", "target")


def _quantize_linear(lin: Any, spec: str, compute_dtype: Any) -> Any:
    """One `nn.Linear` -> its bitsandbytes counterpart, weights carried over.

    Not `transformers.integrations.bitsandbytes.replace_with_bnb_linear`: that one
    builds the replacement on the ``meta`` device and leaves the weights to a
    checkpoint load afterwards, which is the `from_pretrained` path. Laya builds
    the model and loads the weights itself, so the transfer has to happen here.
    """
    import bitsandbytes as bnb

    bias = lin.bias is not None
    if spec == "nf4":
        q = bnb.nn.Linear4bit(
            lin.in_features, lin.out_features, bias=bias,
            compute_dtype=compute_dtype, quant_type="nf4", compress_statistics=True,
        )
        q.weight = bnb.nn.Params4bit(
            lin.weight.data, requires_grad=False, quant_type="nf4", compress_statistics=True
        )
    else:
        q = bnb.nn.Linear8bitLt(lin.in_features, lin.out_features, bias=bias, threshold=6.0)
        q.weight = bnb.nn.Int8Params(lin.weight.data, requires_grad=False, has_fp16_weights=False)
    if bias:
        q.bias = q.bias.__class__(lin.bias.data, requires_grad=False)
    # bitsandbytes builds the quantization state lazily and only when the module is
    # moved onto the device: without this the first forward asserts
    # "quantization state not initialized".
    return q.to(lin.weight.device)


def quantize_encoder(model: Any, spec: str, *, compute_dtype: Any = None) -> dict:
    """Weight-only quantization of the encoder, in place.

    The conversion is weight-only and it touches **the encoder only**: the decision
    head (`scorer`, `act_head`, the marker layers) is what the experiments read, so
    quantizing it would blur the readout instead of the backbone. It is also what
    LLM serving quantizes in practice.

    The count of converted layers goes in the result: "int4" in a run config that
    quietly quantized 3 of 206 layers would be a lie.

    `compute_dtype` is the dtype the dequantized math runs in, so it defaults to the
    model's own precision rather than to fp16.
    """
    if spec in (None, "", "none"):
        return {"quantization": "none"}
    if spec not in QUANTIZATION:
        raise ValueError(f"quantizzazione sconosciuta {spec!r}; noti: {list(QUANTIZATION)}")

    import torch

    encoder = model.encoder
    total = sum(1 for _, m in encoder.named_modules() if type(m) is torch.nn.Linear)
    replaced = 0
    for name, module in list(encoder.named_modules()):
        for child_name, child in list(module.named_children()):
            if type(child) is not torch.nn.Linear:
                continue
            setattr(module, child_name, _quantize_linear(child, spec, compute_dtype))
            replaced += 1
    return {
        "quantization": spec,
        "encoder_linear_total": total,
        "encoder_linear_converted": replaced,
        "compute_dtype": str(compute_dtype) if compute_dtype is not None else "model default",
    }


class CachedTokenizer:
    """Memoizing proxy for the tokenizer: laya re-tokenizes the same head once per state.

    `build_sequence` recomputes the whole head (question text plus every option)
    for **each** state, although the head only depends on the question. At K=16
    that head is 95 of 111 tokens, and 17 of the 18 tokenizer calls a state
    triggers are byte-identical to the previous state's.

    Measured on a 32-state chunk: python side 14.7ms -> 1.3ms (MASSIVE it-IT) and
    52.0ms -> 6.2ms (typed-decisions), i.e. -12% and -5% on the whole forward, with
    `input_ids`, `attention_mask` and `marker_pos` **bit-identical**. It is a
    proxy, not a reimplementation: only `__call__` is intercepted, and every
    other attribute is forwarded, so a change in laya's use of the tokenizer shows
    up as a missing attribute instead of as a silently wrong encoding.

    The cache is bounded because the states themselves are all distinct: the
    redundancy is in the head (few keys, hit every time), not in the states.
    """

    def __init__(self, tok: Any, maxsize: int = 4096) -> None:
        self._tok = tok
        self._maxsize = maxsize
        self._cache: dict[tuple, tuple] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tok, name)

    def __call__(self, text, **kwargs) -> dict:
        key = (text, tuple(sorted(kwargs.items()))) if kwargs else (text,)
        ids = self._cache.get(key)
        if ids is None:
            ids = tuple(self._tok(text, **kwargs)["input_ids"])
            if len(self._cache) >= self._maxsize:
                self._cache.clear()  # bounded by dropping everything, not by LRU bookkeeping
            self._cache[key] = ids
        # A fresh list: `build_sequence` slices what it gets back, and a cached
        # list shared across calls would let one caller corrupt every later one.
        return {"input_ids": list(ids)}


class LayaTokenBudget:
    """Token budget and probes on the sequence: the surface every laya-family
    backend shares.

    Stays here, and not in `LayaEngine`, because `build_sequence` and
    `render_options` have the same signature on every backend in the family:
    duplicating them would mean two implementations to keep aligned with laya's
    private API.

    A backend that implements this class can't fail with an ``AttributeError`` in
    the middle of an experiment: if a piece is missing it's missing immediately
    and with a name that says which.

    To provide: ``self._tok`` (tokenizer), ``self._lens() -> (max_len, head_max_len)``,
    ``self._internal(questions) -> {qid: internal form}``.
    """

    _tok: Any

    def count_tokens(self, text: str) -> int:
        """State length in model tokens (attack budget checks)."""
        return len(self._tok(text, add_special_tokens=False)["input_ids"])

    def state_budget(self, questions: dict[str, Any]) -> dict[str, int]:
        """Token room left for the state, per question: ``max_len - head - 1``.

        The binding constraint of the truncation attack
        (20-attack-surface/truncation-stato.md).

        The ``-1`` is not cosmetic: ``build_sequence`` reserves one token for the
        SEP that closes the state (``room = max_len - len(ids) - 1``), so the
        number returned here is the room laya really leaves, not the head size.
        """
        from laya.agent import build_sequence

        max_len, head_max_len = self._lens()
        internal = self._internal(questions)
        budget = {}
        for qid, q in internal.items():
            ids, _ = build_sequence(self._tok, "", q, max_len=max_len, head_max_len=head_max_len)
            budget[qid] = max(0, max_len - len(ids) - 1)
        return budget

    def probe_sequence(
        self, state: str, questions: dict[str, Any], *, marker: str, truncate_left: bool = False
    ) -> list[dict]:
        """Build the real token sequence per question and test marker survival.

        The only supported way to inspect truncation: if ``marker`` sits at the
        end of the state and disappears from the decoded sequence, the signal
        fell outside the budget.
        """
        from laya.agent import build_sequence

        max_len, head_max_len = self._lens()
        internal = self._internal(questions)
        rows = []
        for qid, q in internal.items():
            ids, markers = build_sequence(
                self._tok, state, q, max_len=max_len, head_max_len=head_max_len, truncate_left=truncate_left
            )
            rows.append(
                {
                    "qid": qid,
                    "seq_tokens": len(ids),
                    "state_tokens": self.count_tokens(state),
                    "option_markers": len(markers),
                    "marker_visible": marker in self._tok.decode(ids, skip_special_tokens=False),
                }
            )
        return rows

    def probe_head(self, questions: dict[str, Any]) -> list[dict]:
        """Does the question head fit ``head_max_len``? (E00 head-budget gate)."""
        from laya.agent import build_sequence

        max_len, head_max_len = self._lens()
        internal = self._internal(questions)
        rows = []
        for qid, q in internal.items():
            ids, markers = build_sequence(self._tok, "", q, max_len=max_len, head_max_len=head_max_len)
            head_tokens = len(ids) - 1  # drop the trailing SEP that opens the state
            rows.append(
                {
                    "qid": qid,
                    "head_max_len": head_max_len,
                    "built_head_tokens": head_tokens,
                    "option_markers": len(markers),
                    "head_fits": head_tokens <= head_max_len,
                }
            )
        return rows

    def render_options(self, questions: dict[str, Any]) -> dict[str, list[str]]:
        """Option labels as the model sees them (E00 head-budget probe)."""
        from laya.agent import render_options

        return {qid: render_options(q) for qid, q in self._internal(questions).items()}

    def head_text(self, questions: dict[str, Any]) -> dict[str, dict]:
        """The built head per question: its length in tokens and its decoded text.

        What the model actually reads of the question, not what the caller asked for.
        `loaders.check_head_budget` needs exactly this to tell a fitted head from a
        clipped one, and reaching into `engine._agent` for it made the audit work on
        one backend only.
        """
        from laya.agent import build_sequence

        max_len, head_max_len = self._lens()
        internal = self._internal(questions)
        out: dict[str, dict] = {}
        for qid, q in internal.items():
            ids, markers = build_sequence(self._tok, "", q, max_len=max_len, head_max_len=head_max_len)
            out[qid] = {
                "head_tokens": len(ids) - 1,  # drop the SEP that opens the state
                "markers": len(markers),
                "text": self._tok.decode(ids, skip_special_tokens=False),
            }
        return out

    def decision_temperature(self, primitive: str, k: int) -> dict:
        """Which temperature laya applies to a decision with ``k`` options.

        Logged by every K sweep, because it is not constant: `temp_bucket` switches
        at K=5/10/11, so a curve that crosses those boundaries compares numbers
        produced under different sharpness. On `laya-typed-decisions` the `choice:11+`
        bucket ships 0.1006, clamped to 0.5, i.e. 2x sharper logits than the
        `choice:6-10` bucket: the OER curve moves at the boundary with no attack
        involved.
        """
        from laya.common import QTYPES, temp_bucket

        qt = QTYPES[primitive]
        bucket = temp_bucket(qt, k)
        raw = self._temperature_by_options_raw.get(bucket, self._temperature_raw[qt])
        return {
            "primitive": primitive,
            "options": k,
            "bucket": bucket,
            "temperature_raw": raw,
            "temperature_applied": self._temperature_by_options.get(
                bucket, self._temperature[qt]
            ),
        }


class LayaEngine(LayaTokenBudget):
    """Load one checkpoint and score cases against it (batched)."""

    name = "laya"

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        device: str | None = None,
        batch_size: int = 16,
        fast: bool = False,
        compile: bool = False,
        quantize: str = "none",
        variant: str = "fp32",
    ) -> None:
        import laya  # lazy: importing layaadv must stay torch-free (tests, --help)

        # Simmetrico al rifiuto di `quantize` su deberta-onnx: ogni backend conosce una
        # sola maniera di scegliere la precisione. Ignorare l'altra in silenzio
        # produrrebbe quattro configurazioni diverse sullo stesso grafo, cioe' una tabella
        # di "la quantizzazione non cambia niente" costruita a mano da un errore di plumbing.
        if variant and variant != "fp32":
            raise ValueError(
                f"variant={variant!r} seleziona i grafi ONNX di deberta-onnx. Per la famiglia "
                "laya la precisione si sceglie con quantize (none/int8/nf4)."
            )

        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.compiled = bool(compile)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._agent = laya.load(checkpoint, device=device, fast=fast)
        self.quantization = quantize_encoder(
            self._agent.model, quantize, compute_dtype=self._agent.dtype
        )
        if self.compiled:
            # Same two lines `Agent(compile=True)` does; `laya.load()` does not
            # forward the flag, so without this the option is unreachable.
            #
            # It is **not** free of consequences: fusion reorders the reductions,
            # so the probabilities move (measured max |Δconfidence| 0.0031 on
            # typed-decisions, 2 of 15 answers unchanged) and it costs ~55s of
            # warmup on the first run, because 96 real cases produce 71 distinct
            # sequence shapes. It buys 1.44x and costs bit-identical
            # reproducibility, so it belongs in the run identity: a compiled run
            # and an eager one are not the same experiment and must not share a
            # run_id.
            import torch

            self._agent.model.encoder.config.reference_compile = True
            self._agent.model = torch.compile(self._agent.model)
        # Laya warns at load about uncalibrated temperatures — E00 logs them.
        self.load_warnings = [str(w.message) for w in caught]
        self._device_seen = str(getattr(self._agent, "device", ""))
        self._weight_sha256: str | None = None
        # The agent's own tokenizer is on the hot path (`_encode_state` runs per
        # state per question), so the cache has to sit on *that* one, not on a
        # copy kept by the engine.
        self._agent.tok = self._tok = CachedTokenizer(self._agent.tok)
        # Le temperature servono a `decision_temperature`: le applicate (clampate
        # da laya) e quelle dichiarate dal checkpoint, che possono essere fuori range.
        self._temperature = self._agent.temperature
        self._temperature_raw = self._agent.temperature_raw
        self._temperature_by_options = self._agent.temperature_by_options
        self._temperature_by_options_raw = self._agent.temperature_by_options_raw
        # `None` = mai rifitta. Va in `run_identity` perche' la temperatura muove le
        # confidenze, e le confidenze sono la metrica che gli OER leggono.
        self._temperature_applied: float | None = None

    def fit_and_apply_temperature(self, temperature: float) -> dict:
        """Apply a refitted temperature to every decision, and declare that it happened.

        The laya checkpoints ship temperatures outside the range the loader accepts
        (`choice:11+ = 0.1006`, clamped to 0.5 on `laya-typed-decisions`), so the confidence
        of the affected entries carries an asterisk and every OER on this campaign is
        comparable only *within* the laya family. Fitting the temperature on the clean case and
        applying it is the deployment step of E16: without it the defence is a number computed
        in an experiment, and the OER a reader would compare against another model is not the
        same quantity.

        The applied value goes into `run_identity`, so a run with a refitted temperature gets a
        different `run_id` from the stock one: a temperature changes the confidences, and the
        confidences are the metric.
        """
        value = float(temperature)
        # La struttura di laya e' precisa e sostituirla rotta il decode: `temperature` e' una
        # **lista** di un elemento per tipo primitivo (l'ordine e' quello di `common.QTYPES`),
        # `temperature_by_options` un dizionario per bucket. Rimettere uno scalare su
        # `temperature` fa fallire `self.temperature[qt]` con "float is not subscriptable",
        # quindi qui si cambiano i valori e la struttura resta quella che laya si aspetta.
        table = getattr(self._agent, "temperature", None)
        if isinstance(table, list):
            for i in range(len(table)):
                table[i] = value
        by_options = getattr(self._agent, "temperature_by_options", None)
        if isinstance(by_options, dict):
            for key in by_options:
                by_options[key] = value
        self._temperature_applied = value
        return {
            "temperature_applied": value,
            "temperature_shipped_raw": self._temperature_raw,
            "torch_dtype": str(next(self._agent.model.parameters()).dtype),
        }

    # ----------------------------------------------------------------- predict
    def predict(self, case: Case) -> Prediction:
        """Score a single case (thin wrapper: one code path for everything)."""
        return self.predict_batch([case])[0]

    def predict_batch(
        self, cases: list[Case], batch_size: int | None = None
    ) -> list[Prediction]:
        """Score many cases, grouped by identical questions.

        Returns predictions in the same order as ``cases``.
        """
        if not cases:
            return []
        size = batch_size or self.batch_size
        out: list[Prediction | None] = [None] * len(cases)
        for members in group_cases(cases):
            questions = members[0][1].questions
            states = [case.state for _, case in members]
            results = self._agent.predict_batch(states, questions, batch_size=size)
            if len(results) != len(states):
                raise RuntimeError(
                    f"laya returned {len(results)} results for {len(states)} states"
                )
            for (i, case), res in zip(members, results):
                answers = res.get("answers") or {}
                out[i] = Prediction(
                    id=case.id,
                    answers=answers,
                    labels={qid: answer_label(a) for qid, a in answers.items()},
                    usage=res.get("usage") or {},
                )
            self._check_device()
        missing = [cases[i].id for i, p in enumerate(out) if p is None]
        if missing:
            raise RuntimeError(f"missing predictions for: {missing}")
        return [p for p in out if p is not None]

    @staticmethod
    def batching_report(cases: Sequence[Case]) -> dict:
        """How the batching **actually** went, as opposed to how it was requested.

        `batch_size` only has an effect when several states share one question
        signature. `laya.predict_batch` applies a single `questions` dict to every
        state, so a group of one state can only ever be one forward pass whatever
        `batch_size` says. That is the normal case on the datasets the attack
        sweeps use: `loaders.load_cases` draws each case's option set from a
        **per-case** rng, so no two cases share a signature and every group has one
        member. A run config that records `batch_size: 16` is then describing a
        knob that did nothing.

        Recorded per run so the number in the config can be read against the
        number that happened. It goes in the result payload and not in
        `run_identity`: the truth about a run is worth recording even when it
        doesn't change which run this is, and adding it to the identity would
        invalidate every existing resume for no gain in reproducibility.
        """
        groups = group_cases(list(cases))
        sizes = sorted((len(g) for g in groups), reverse=True)
        return {
            "states": len(cases),
            "question_groups": len(groups),
            "max_states_per_group": sizes[0] if sizes else 0,
            "batch_size_effective": sizes[0] if sizes else 0,
            "batch_size_requested": None,  # filled by evaluate, which knows the argument
            "batching_is_inert": bool(sizes) and sizes[0] == 1,
        }

    def _check_device(self) -> None:
        """Laya's OOM fallback demotes the model to CPU without raising."""
        device = str(getattr(self._agent, "device", ""))
        if device != self._device_seen:
            warnings.warn(
                f"laya fell back from {self._device_seen} to {device} (OOM?): "
                "throughput is now CPU-bound, lower --batch-size",
                RuntimeWarning,
                stacklevel=2,
            )
            self._device_seen = device

    def runtime_state(self) -> dict:
        """Where the model is right now, and in what precision.

        This is not the same as `env_fingerprint`'s device: that one reports the GPU
        whenever CUDA is available *at all*, so a run that laya demoted to CPU mid-way
        still says "RTX 3080". Speed samples take the value from here, because the
        demotion costs about 10x and a latency without it is not comparable.
        """
        return {
            "device": str(getattr(self._agent, "device", "")),
            "dtype": str(getattr(self._agent, "dtype", "")),
        }

    def run_identity(self) -> dict:
        """What changes the numbers, for a run config.

        `backend` was missing until now, which made an ONNX run and a native run on
        the same checkpoint, dataset and K share a `run_id`: a resume would then read
        the other's predictions and mix two implementations under one number. Same
        for `compile`, which moves the probabilities by ~0.003.
        """
        return {
            "backend": self.name,
            "compiled": self.compiled,
            "fast": bool(self._agent._fast),
            # Cosa significa `record["confidence"]` su questo backend. Il campo non e' una
            # grandezza sola: su k=2 (noul) i backend ci mettono max(p, 1-p), su k>2
            # l'entropia normalizzata, e le due non coincidono (0.726 contro 0.152 a parita'
            # di distribuzione). Senza questa dichiarazione un `OER@0.75` letto su una riga
            # CLM e su una riga laya sembra lo stesso numero e non lo e'. Fuori dal run_id per
            # la regola di `batching_report`: non cambia quale run e' la stessa, e aggiungerla
            # all'identita' invaliderebbe ogni resume senza guadagno di riproducibilita'.
            "confidence_definition": "max(p,1-p) if k==2 else 1-H(p)/log(k)",
            # La temperatura entra nell'identita': se cambia, le confidenze cambiano, e ogni
            # OER letto da una run con T rifitta non e' confrontabile con una stock.
            "temperature_applied": self._temperature_applied,
            "temperature_shipped": self._temperature_raw,
            "quantization": self.quantization["quantization"],
            "encoder_linear_converted": self.quantization.get("encoder_linear_converted"),
            # Il device va nell'identità perche' laya puo' tornare su CPU da solo (OOM), e un
            # run su GPU e uno su CPU dello stesso checkpoint avrebbero lo stesso run_id: il
            # resume del secondo leggerebbe le previsioni del primo e una latenza da 30 ms
            # verrebbe pubblicata come se fosse stata misurata sulla 3080. Si registra il
            # device *richiesto*, non quello corrente: `runtime_state` cambia se il modello
            # viene demoto, e l'identità di un run in corso non puo' cambiare sotto i piedi.
            "device": str(getattr(self._agent, "device", "")),
        }

    # -------------------------------------------------------------- attention
    def attention_structure(self) -> dict:
        """The encoder's attention layout, read from the loaded model.

        The notes claim "1/3 global, 2/3 local window 128"
        (10-modello/architettura-laya.md, 20-attack-surface/truncation-stato.md).
        A claim about the model is read from the model here, layer by layer, and
        the module and the config are checked against each other: with 28 layers
        "one in three global" is 10 of 28 = 0.357, not 1/3, and `local_attention`
        is a half-window (transformers documents `sliding_window =
        local_attention // 2`), so a note that says "window 128" is describing
        `local_attention` and not the distance a local layer can actually reach.

        Per layer: the type, the module's own `sliding_window` (None on a global
        layer), and whether module and config agree. `layers` is the layout an
        experiment needs to split a profile into global and local without knowing
        the checkpoint.
        """
        import torch  # noqa: F401  (il controllo sui moduli richiede torch a runtime)

        encoder = self._agent.model.encoder
        config = encoder.config
        modules = [(name, mod) for name, mod in encoder.named_modules()
                   if type(mod).__name__ == "ModernBertAttention"]
        layer_types = list(config.layer_types)
        if len(modules) != len(layer_types):
            raise RuntimeError(
                f"{len(modules)} moduli di attenzione per {len(layer_types)} layer_types: "
                "il modello caricato non e' leggibile da questo estrattore"
            )
        layers, disagreements = [], []
        for index, ((name, mod), kind) in enumerate(zip(modules, layer_types)):
            declared = getattr(encoder.layers[index], "attention_type", None)
            window = getattr(mod, "sliding_window", None)
            if declared != kind:
                disagreements.append(f"layer {index}: modulo {declared!r} vs config {kind!r}")
            if (window is None) != (kind == "full_attention"):
                disagreements.append(
                    f"layer {index}: {kind} con sliding_window={window!r} sul modulo"
                )
            layers.append({
                "layer": index,
                "module": name,
                "layer_type": kind,
                "sliding_window_module": window,
            })
        n_global = sum(1 for row in layers if row["layer_type"] == "full_attention")
        return {
            "n_layers": len(layers),
            "layers": layers,
            "layer_types": layer_types,
            "n_global": n_global,
            "n_local": len(layers) - n_global,
            "fraction_global": n_global / len(layers),
            "fraction_local": (len(layers) - n_global) / len(layers),
            "num_attention_heads": config.num_attention_heads,
            "hidden_size": config.hidden_size,
            "local_attention": getattr(config, "local_attention", None),
            "sliding_window": getattr(config, "sliding_window", None),
            "global_attn_every_n_layers": getattr(config, "global_attn_every_n_layers", None),
            "attn_implementation": config._attn_implementation,
            "module_disagreements": disagreements,
        }

    def attention_profiles(
        self, cases: Sequence[Case], specs: Sequence[dict], *, batch_size: int = 1
    ) -> list[dict]:
        """Attention that each case's sequence pays to a token span, per layer.

        One forward per case (``batch_size`` is accepted and pinned to 1 on
        purpose: the batch composition moves a bf16 prediction, and this
        experiment compares its own forward against the stock run's label, so the
        two have to see the same shapes).

        The encoder is switched to the eager attention implementation for the
        duration of the call and restored afterwards: SDPA does not return the
        weights at all (it warns and hands back ``None``). Eager in bf16 is not
        bit-identical to the fused kernel, which is why every row carries
        ``argmax_match``: the forward that produces the attention has to agree
        with the prediction the run stored, or the profile belongs to a different
        decision than the one published.

        ``specs[i]`` describes the spans of case ``i``::

            {"spans": [{"name": "head", "tokens": 9, "sentence": "..."},
                       {"name": "tail", "tokens": 9, "sentence": None}]}

        ``tokens`` is the span length and ``sentence`` the text the attacker wrote,
        or ``None`` for a position-matched control (a span of the same length where
        the sentence *would* be, with nothing in it). Where the span is, and which
        rule found it, is :meth:`_locate_span`; every row carries ``located_by``,
        ``text_match`` (the span decodes back to the sentence) and ``decoded``, so a
        span measured on the wrong tokens is visible instead of silent.

        Per span and per layer the rows carry:
        ``all``      mass from every real query,
        ``markers``  mass from the option ``[MASK]`` positions, the ones the
                     decision head reads,
        ``state``    mass that lands on the state region at all (from every
                     query), so a profile can be read as a share.
        """
        from laya.agent import _amp_context  # private: solo qui

        if len(cases) != len(specs):
            raise ValueError(f"{len(cases)} casi per {len(specs)} specifiche di span")
        size = max(1, int(batch_size))
        if size != 1:
            raise ValueError(
                "attention_profiles gira un caso per forward: in bf16 la composizione del "
                "batch sposta le previsioni, e il controllo di coerenza con il run stock "
                "perderebbe il suo significato"
            )

        import torch

        model = self._agent.model
        encoder = model.encoder
        layout = self.attention_structure()
        attn_modules = [mod for _, mod in encoder.named_modules()
                        if type(mod).__name__ == "ModernBertAttention"]
        if len(attn_modules) != layout["n_layers"]:
            raise RuntimeError("l'layout dell'attenzione non corrisponde ai moduli caricati")

        captured: list[torch.Tensor] = []

        def hook(_module, _args, output):
            weights = output[1] if isinstance(output, tuple) else None
            if weights is None:
                raise RuntimeError(
                    "l'attenzione non e' stata restituita: il percorso non e' eager, e senza "
                    "i pesi il profilo sarebbe un numero inventato"
                )
            captured.append(weights)

        handles = [mod.register_forward_hook(hook) for mod in attn_modules]
        previous = encoder.config._attn_implementation
        encoder.config._attn_implementation = "eager"
        try:
            with torch.no_grad(), _amp_context(self._agent.device, self._agent.dtype):
                rows = [
                    self._attention_row(case, spec, captured, layout)
                    for case, spec in zip(cases, specs)
                ]
        finally:
            for handle in handles:
                handle.remove()
            encoder.config._attn_implementation = previous
        self._check_device()
        return rows

    def attention_receptive_field(
        self, case: Case, *, probes: Sequence[int] = (0, 20, 60, 120, 200)
    ) -> dict:
        """How far a local layer can actually see, measured on the weights.

        The notes say "local window 128" (10-modello/architettura-laya.md,
        20-attack-surface/truncation-stato.md). transformers documents
        ``sliding_window = local_attention // 2`` and masks with
        ``abs(query - key) <= sliding_window``, so a local layer reaches ±64 tokens
        while a global one reaches the whole sequence: the note's number is
        `local_attention`, not the distance. That is a claim about the model, so it
        is read off the attention weights here, not off a docstring: for each probe
        query, the farthest key the layer gives a non-zero weight.

        Masked positions are exactly zero (the softmax sees ``-inf``), so "non-zero"
        is the reachable set and not a threshold someone chose.
        """
        import torch
        from laya.agent import _amp_context  # private: solo qui
        from laya.common import collate_items

        questions = dict(case.questions)
        ids = list(questions)
        internal = self._internal(questions)
        items = self._agent._encode_state(case.state, ids, internal)
        batch = collate_items([items], self._agent.tok.pad_token_id)
        layout = self.attention_structure()

        captured: list[torch.Tensor] = []

        def hook(_module, _args, output):
            captured.append(output[1] if isinstance(output, tuple) else None)

        encoder = self._agent.model.encoder
        modules = [mod for _, mod in encoder.named_modules()
                   if type(mod).__name__ == "ModernBertAttention"]
        handles = [mod.register_forward_hook(hook) for mod in modules]
        previous = encoder.config._attn_implementation
        encoder.config._attn_implementation = "eager"
        try:
            with torch.no_grad(), _amp_context(self._agent.device, self._agent.dtype):
                del captured[:]
                device = self._agent.device
                self._agent.model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device),
                    batch["marker_mask"].to(device),
                    batch["qtype"].to(device),
                )
        finally:
            for handle in handles:
                handle.remove()
            encoder.config._attn_implementation = previous
        self._check_device()
        valid = int(batch["attention_mask"][0].sum())
        rows = []
        for index, weights in enumerate(captured):
            if weights is None:
                raise RuntimeError(f"livello {index}: attenzione non restituita")
            layer = weights[0].float()
            per_probe = {}
            for q in probes:
                if q >= valid:
                    continue
                keys = (layer[0, q] > 0).nonzero(as_tuple=True)[0]
                per_probe[str(q)] = int((keys - q).abs().max()) if keys.numel() else None
            rows.append({
                "layer": index,
                "layer_type": layout["layers"][index]["layer_type"],
                "seq_tokens": valid,
                "reach": per_probe,
            })
        return {"seq_tokens": valid, "probes": list(probes), "layers": rows}

    def _attention_row(self, case: Case, spec: dict, captured: list, layout: dict) -> dict:
        """One case's attention profile: forward, span location, aggregation.

        Split out of :meth:`attention_profiles` so the whole per-case path lives
        in one place and the try/finally around the encoder's attention
        implementation stays readable.
        """
        import torch
        from laya.common import collate_items

        del captured[:]
        questions = dict(case.questions)
        ids = list(questions)
        internal = self._internal(questions)
        items = self._agent._encode_state(case.state, ids, internal)
        batch = collate_items([items], self._agent.tok.pad_token_id)
        device = self._agent.device
        logits, _act = self._agent.model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["marker_pos"].to(device),
            batch["marker_mask"].to(device),
            batch["qtype"].to(device),
        )
        if len(captured) != layout["n_layers"]:
            raise RuntimeError(
                f"catturati {len(captured)} tensori di attenzione su {layout['n_layers']} livelli"
            )

        seq = items[0]["ids"]
        valid = len(seq)
        state_start = self._state_start(case)
        if not 0 < state_start < valid:
            raise RuntimeError(f"stato fuori dalla sequenza: start={state_start}, len={valid}")
        n_markers = int(batch["marker_mask"][0].sum())
        marker_pos = [int(p) for p in batch["marker_pos"][0, :n_markers].tolist()]
        options = list(case.questions[ids[0]]["criteria"])
        k = min(n_markers, int(logits.shape[1]))
        argmax = int(logits[0, :k].float().argmax())

        queries_all = slice(0, valid)
        queries_markers = torch.tensor(marker_pos, device=device, dtype=torch.long)
        cols_state = slice(state_start, valid - 1)

        spans = {}
        for entry in spec.get("spans", []):
            spans[entry["name"]] = self._span_profile(
                entry, seq=seq, state_start=state_start, valid=valid,
                markers=marker_pos, captured=captured, queries_all=queries_all,
                queries_markers=queries_markers, cols_state=cols_state,
            )
        return {
            "seq_tokens": valid,
            "state_tokens": max(0, valid - 1 - state_start),
            "head_tokens": state_start,
            "n_markers": n_markers,
            "marker_first": marker_pos[0] if marker_pos else None,
            "marker_last": marker_pos[-1] if marker_pos else None,
            "argmax": argmax,
            "argmax_label": options[argmax] if argmax < len(options) else None,
            "spans": spans,
        }

    def _state_start(self, case: Case) -> int:
        """Index where the state begins in the built sequence.

        Read from the sequence laya builds for the *empty* state, which is the
        head plus the SEP that opens the state, minus that SEP. Not computed from
        the question text: the head length is whatever the tokenizer made of it.
        """
        from laya.agent import build_sequence

        max_len, head_max_len = self._lens()
        qid, question = next(iter(case.questions.items()))
        internal = self._internal({qid: question})
        empty, _ = build_sequence(self._tok, "", internal[qid], max_len=max_len,
                                  head_max_len=head_max_len)
        return len(empty) - 1

    def _locate_span(
        self, entry: dict, *, seq: list[int], state_start: int, valid: int
    ) -> tuple[int, int, str, str]:
        """Where a span is, and which rule found it.

        Three rules, best first, and the row says which one was used:

        1. **ids**, a unique occurrence of the sentence's own token ids.
        2. **diff** against a reference state. The sentence's ids differ from what
           the sentence becomes inside the state (tokenisation is not composable: the
           junction merges a token, so a tail sentence matched by its own ids is not
           found at all), but the *boundary* is exact: the attacked state's tokens
           share a prefix (sentence at the tail) or a suffix (sentence at the head)
           with the reference's, and the first divergence is where the sentence
           begins.
        3. **position**, the first/last ``tokens`` tokens of the state region: the
           position-matched control, where there is no sentence to find.

        Neither sentence rule wins in general, so the one that **decodes back to the
        sentence** is chosen. Kept out of :meth:`_span_profile` because the control arm
        needs the same answer without a forward: a control span one or two tokens
        shorter than the attacked one would measure the length, not the presence of
        the sentence.

        Returns ``(start, end, located_by, decoded)``.
        """
        name = entry["name"]
        sentence = entry.get("sentence")
        reference = entry.get("reference_state")
        requested = int(entry.get("tokens") or 0)
        state_tokens = max(0, valid - 1 - state_start)
        positional_n = max(1, min(requested, state_tokens))
        positional_start = (
            state_start if name == "head" else
            valid - 1 - positional_n if name == "tail" else None
        )
        if positional_start is None:
            raise ValueError(f"posizione sconosciuta: {name!r}")

        candidates: list[tuple[int, int, str]] = []
        if sentence:
            token_ids = list(self._tok(sentence, add_special_tokens=False)["input_ids"])
            found = _find_subsequence(seq, token_ids) if token_ids else []
            if len(found) == 1:
                candidates.append((found[0], len(token_ids), "ids"))
        if sentence and reference:
            ref = list(self._tok(reference, add_special_tokens=False)["input_ids"])
            own = list(seq[state_start : valid - 1])
            if name == "tail":
                # own = clean + junction + frase: il confine e' il prefisso comune
                start = state_start + _common_prefix_length(own, ref)
                candidates.append((start, max(1, valid - 1 - start), "diff"))
            else:
                # own = frase + junction + clean: il confine e' il suffisso comune,
                # e la frase sta *prima*, quindi lo span comincia con lo stato.
                candidates.append(
                    (state_start, max(1, len(own) - _common_prefix_length(own[::-1], ref[::-1])),
                     "diff")
                )
        candidates.append((positional_start, positional_n, "position"))

        def decoded_of(start: int, n: int) -> str:
            return self._tok.decode(list(seq[start : start + n]), skip_special_tokens=True).strip()

        wanted = sentence.strip() if isinstance(sentence, str) else None
        start, n, located_by = next(
            (c for c in candidates if wanted and decoded_of(c[0], c[1]) == wanted),
            candidates[0],
        )
        return start, start + n, located_by, decoded_of(start, n)

    def attention_span_tokens(
        self, case: Case, *, position: str, sentence: str, reference_state: str
    ) -> int:
        """How many tokens this sentence occupies in the built sequence.

        The control arm needs the *real* length of the attacked span: the sentence
        tokenizes longer inside the state than in isolation (the junction adds a token
        or two), so counting it alone gives a control one fifth shorter than the thing
        it controls for, and the difference would be measured as attention.
        """
        questions = dict(case.questions)
        ids = list(questions)
        internal = self._internal(questions)
        state = (
            f"{sentence} {reference_state}" if position == "head"
            else f"{reference_state} {sentence}"
        )
        armed = case.__class__(id=case.id, state=state, questions=case.questions, gold=case.gold)
        items = self._agent._encode_state(armed.state, ids, internal)
        seq = items[0]["ids"]
        entry = {"name": position, "tokens": 0, "sentence": sentence,
                 "reference_state": reference_state}
        start, end, _located, _decoded = self._locate_span(
            entry, seq=seq, state_start=self._state_start(armed), valid=len(seq)
        )
        return end - start

    def _span_profile(
        self, entry: dict, *, seq: list[int], state_start: int, valid: int,
        markers: list[int], captured: list, queries_all, queries_markers, cols_state,
    ) -> dict:
        """Aggregate the attention that lands on one located span.

        The location is :meth:`_locate_span`; here it is only summed. The mass onto a
        span of ``n`` tokens is a share of one query's attention (rows sum to one over
        the visible keys), averaged over heads and over the query set. ``state`` (the
        mass onto the whole state region) is returned so the share can be read against
        what the sequence pays to the state at all.

        Slices, not index tensors, wherever the range is contiguous: an indexed copy of
        the query dimension would materialise 16 x queries x keys floats per layer,
        which is the difference between a run of minutes and one of hours at N=2974.
        """
        sentence = entry.get("sentence")
        start, end, located_by, decoded = self._locate_span(
            entry, seq=seq, state_start=state_start, valid=valid
        )
        n = end - start
        distance = (
            sum(abs(m - start) for m in markers) / len(markers) if markers else None
        )
        cols_span = slice(start, end)

        all_mass, marker_mass, state_mass = [], [], []
        for weights in captured:
            layer = weights[0].float()  # [heads, queries, keys]
            if layer.shape[-1] < valid:
                raise RuntimeError(
                    f"l'attenzione copre {layer.shape[-1]} chiavi per {valid} token: il "
                    "batch non e' uno per caso"
                )
            # Indice esplicito sulla dimensione delle query: `layer[tensor]` indicizza
            # le teste, non le query, e con 16 marker su 16 teste l'indice fuori
            # range non sempre solleva (qui solleva, ed e' il motivo della guardia).
            all_mass.append(float(layer[:, queries_all][..., cols_span].sum(-1).mean()))
            marker_mass.append(float(layer[:, queries_markers][..., cols_span].sum(-1).mean()))
            state_mass.append(float(layer[:, queries_all][..., cols_state].sum(-1).mean()))
        return {
            "name": entry["name"],
            "n": n,
            "start": start,
            "end": start + n,
            "located_by": located_by,
            # Controllo leggibile a occhio: lo span decodificato deve essere la frase
            # iniettata. `None` quando non c'e' frase (controllo posizionale).
            "text_match": (
                (decoded == sentence.strip())
                if isinstance(sentence, str) and sentence.strip() else None
            ),
            "decoded": decoded[:80],
            "distance_to_markers": round(distance, 2) if distance is not None else None,
            "all": [round(v, 6) for v in all_mass],
            "markers": [round(v, 6) for v in marker_mass],
            "state": [round(v, 6) for v in state_mass],
        }

    # -------------------------------------------------------- marker state patching
    def marker_patch_profiles(
        self, cases: Sequence[Case], attacked: Sequence[Case], *,
        layers: Sequence[int], markers: Sequence[str] = ("all",),
        target_index: Sequence[int | None] | None = None,
    ) -> list[dict]:
        """Restore the clean hidden states at the option markers, per patch point.

        The causal counterpart of :meth:`attention_profiles`: instead of reading
        where the model looks, this **replaces the pathway** that carries the
        attack into the states the decision head reads, and measures how much of
        the flip comes back. The measure is a mediation one
        (60-interpretabilita/mediazione-causale.md): the treatment is the
        injected state, the mediator is the hidden state at the option markers,
        the response is the decision.

        For each pair ``(cleancase, attackedcase)`` one clean forward captures the
        marker states at every patch point, then one forward per
        (patch point, marker subset) writes them back into the attacked run:

        ``layer = l``    the input of encoder layer ``l`` (layers before ``l``
                         still compute the attacked states),
        ``layer = len``  the encoder's own output, i.e. the input to the decision
                          head.

        The deepest patch is **not** guaranteed to recover: the head runs its own
        two transformer layers over the whole sequence and gathers the markers
        only afterwards, so it still sees the attacked non-marker positions. That
        is the point of the measure, not a defect.

        ``markers`` selects which marker states to restore: ``all`` (every
        option), ``gold`` (the correct one) and ``target`` (the one the sentence
        names, from ``target_index``). Comparing them separates "the attack
        writes into the named option" from "the attack damages the correct one".

        Off with ``--fast`` and ``--compile``: both bypass or may bypass the layer
        hooks this intervention is made of, and a run where every patch did
        nothing would report "no effect" instead of an error
        (the `--flag accepted and ignored` failure class).
        """
        if getattr(self._agent, "_fast", None) is not None:
            raise RuntimeError(
                "marker_patch_profiles non interviene con --fast: laya.fast sostituisce "
                "`model.forward` con una implementazione propria, quindi i ganci sui livelli "
                "dell'encoder non scatterebbero e ogni patch verrebbe ignorata in silenzio. "
                "Rilanciare senza --fast."
            )
        if self.compiled:
            raise RuntimeError(
                "marker_patch_profiles non interviene su un modello compilato: l'intervento "
                "passa da ganci sui moduli e un grafo compilato non garantisce che scattino. "
                "Rilanciare senza --compile."
            )
        if len(cases) != len(attacked):
            raise ValueError(f"{len(cases)} casi puliti per {len(attacked)} attaccati")
        wanted = [int(layer) for layer in layers]
        n_layers = len(self._agent.model.encoder.layers)
        for layer in wanted:
            if not 0 <= layer <= n_layers:
                raise ValueError(
                    f"patch point {layer} fuori range: 0..{n_layers} "
                    f"({n_layers} livelli, piu' l'uscita dell'encoder)"
                )
        if not wanted:
            raise ValueError("nessun patch point da misurare")
        unknown = [name for name in markers if name not in _MARKER_SUBSETS]
        if unknown:
            raise ValueError(f"marker sconosciuti: {unknown}; noti: {list(_MARKER_SUBSETS)}")
        if target_index is None and "target" in tuple(markers):
            raise ValueError("markers 'target' richiede target_index")
        if target_index is not None and len(target_index) != len(cases):
            raise ValueError(f"{len(target_index)} target_index per {len(cases)} casi")

        rows = []
        for index, (clean, armed) in enumerate(zip(cases, attacked)):
            target = target_index[index] if target_index is not None else None
            rows.append(self._marker_patch_case(clean, armed, wanted, tuple(markers),
                                                target, n_layers))
        return rows

    def _marker_patch_case(
        self, clean: Case, armed: Case, layers: list[int], markers: tuple[str, ...],
        target: int | None, n_layers: int,
    ) -> dict:
        """One case's patch rows: clean capture, the unpatched attacked forward, then
        every (patch point, marker subset). One row per intervention, all as floats
        and ints: no tensor leaves this module.
        """
        qid = next(iter(armed.questions))
        options = list(armed.questions[qid]["criteria"])
        gold_label = armed.gold[qid]
        if gold_label not in options:
            raise RuntimeError(f"gold {gold_label!r} non è fra le opzioni {options}")

        clean_batch, clean_pos, n_markers = self._marker_batch(clean)
        attack_batch, attack_pos, n_attack = self._marker_batch(armed)
        # I marker stanno nella testa, che non cambia fra i due stati: se le posizioni
        # divergessero, la patch scriverebbe sui token sbagliati e il recupero sarebbe
        # un numero inventato.
        if n_attack != n_markers or attack_pos.tolist() != clean_pos.tolist():
            raise RuntimeError(
                f"marker diversi fra pulito e attaccato ({clean_pos.tolist()} contro "
                f"{attack_pos.tolist()}): il caso {armed.id} non è confrontabile"
            )
        positions = clean_pos
        subsets = {"all": list(range(n_markers)), "gold": [options.index(gold_label)]}
        if "target" in markers:
            if target is None or not 0 <= int(target) < n_markers:
                raise RuntimeError(
                    f"marker del bersaglio {target!r} fuori range su {n_markers} marker"
                )
            subsets["target"] = [int(target)]
        for name in markers:
            if not subsets.get(name):
                raise RuntimeError(f"sottoinsieme {name!r} vuoto su {n_markers} marker")

        clean_index, captured = self._marker_capture(clean_batch, options, positions, n_layers)
        attacked_index = self._marker_decision(attack_batch, options)

        rows = []
        for layer in layers:
            source_all = captured[layer]
            for name in markers:
                indices = subsets[name]
                handle = self._register_marker_patch(layer, positions[indices],
                                                     source_all[:, indices])
                try:
                    patched_index = self._marker_decision(attack_batch, options)
                finally:
                    handle.remove()
                rows.append({
                    "layer": layer,
                    "markers": name,
                    "argmax": patched_index,
                    "label": options[patched_index],
                    # `recovered` e' sempre contro la decisione pulita, non contro il
                    # gold: un caso in cui anche il pulito sbaglia non ha nulla da
                    # recuperare, e l'ASR lo esclude per costruzione.
                    "recovered": patched_index == clean_index,
                })
        return {
            "clean": {"argmax": clean_index, "label": options[clean_index]},
            "attacked": {"argmax": attacked_index, "label": options[attacked_index]},
            "n_markers": n_markers,
            "marker_index": {name: indices[0] for name, indices in subsets.items()},
            "patches": rows,
        }

    def _marker_batch(self, case: Case) -> tuple[Any, Any, int]:
        """One case as the batch the decision model takes: batch of one, one state.

        One case per forward, like :meth:`attention_profiles`: the marker positions
        come from the batch, and a patch has to know exactly which positions the
        head will read. ``batch_size`` is inert here for the same reason it is on
        the attack sweeps (one question signature per case).
        """
        from laya.common import collate_items

        questions = dict(case.questions)
        ids = list(questions)
        internal = self._internal(questions)
        items = self._agent._encode_state(case.state, ids, internal)
        batch = collate_items([items], self._agent.tok.pad_token_id)
        n_markers = int(batch["marker_mask"][0].sum())
        return batch, batch["marker_pos"][0, :n_markers], n_markers

    def _marker_decision(self, batch: Any, options: list[str]) -> int:
        """The index this batch's decision head picks.

        The argmax of the logits, which is the same decision
        :meth:`predict_batch` reports for a `choice` question (the answer is the
        argmax over the option probabilities).
        """
        import torch
        from laya.agent import _amp_context

        device = self._agent.device
        with torch.no_grad(), _amp_context(device, self._agent.dtype):
            logits, _act = self._agent.model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["marker_pos"].to(device),
                batch["marker_mask"].to(device),
                batch["qtype"].to(device),
            )
        k = int(batch["marker_mask"][0].sum())
        index = int(logits[0, :k].float().argmax())
        if not 0 <= index < len(options):
            raise RuntimeError(f"argmax {index} fuori dalle {len(options)} opzioni")
        self._check_device()
        return index

    def _marker_capture(
        self, batch: Any, options: list[str], positions: Any, n_layers: int
    ) -> tuple[int, dict]:
        """The clean forward that keeps the marker states at every patch point.

        Returns the clean decision and, per patch point, the marker states:
        ``0..n_layers-1`` are the inputs of each encoder layer, ``n_layers`` is the
        encoder's own output (the input to the decision head). Values are cloned in
        the model's own dtype, so writing them back is exact rather than
        approximately restored.
        """
        encoder = self._agent.model.encoder
        captured: dict[int, Any] = {}
        handles = []

        def keep_input(index):
            def hook(_module, args):
                captured[index] = args[0][:, positions].clone()
            return hook

        for index, module in enumerate(encoder.layers):
            handles.append(module.register_forward_pre_hook(keep_input(index)))

        def keep_output(_module, _args, output):
            captured[n_layers] = output.last_hidden_state[:, positions].clone()

        handles.append(encoder.register_forward_hook(keep_output))
        try:
            decision = self._marker_decision(batch, options)
        finally:
            for handle in handles:
                handle.remove()
        missing = sorted(set(range(n_layers + 1)) - set(captured))
        if missing:
            raise RuntimeError(
                f"patch point senza stati catturati: {missing}. Il modello caricato non "
                "espone i livelli che questo intervento si aspetta."
            )
        return decision, captured

    def _register_marker_patch(self, layer: int, positions: Any, source: Any):
        """The intervention: write `source` into the marker positions at a patch point.

        Every patch point is an index, ``n_layers`` being the encoder's output: a
        forward **pre** hook on the layer patches its input, a forward hook on the
        encoder patches the value the decision head receives. Both write into a
        clone and return it, so the values captured on the clean forward stay the
        clean ones.
        """
        encoder = self._agent.model.encoder
        if layer == len(encoder.layers):

            def patch_head_input(_module, _args, output):
                hidden = output.last_hidden_state
                patched = hidden.clone()
                patched[:, positions] = source.to(patched.dtype)
                return output.__class__(last_hidden_state=patched)

            return encoder.register_forward_hook(patch_head_input)

        target = encoder.layers[layer]

        def patch_layer_input(_module, args, kwargs):
            hidden = args[0]
            patched = hidden.clone()
            patched[:, positions] = source.to(patched.dtype)
            return (patched, *args[1:]), kwargs

        return target.register_forward_pre_hook(patch_layer_input, with_kwargs=True)

    # -------------------------------------------------------- head knockout (E23b)
    def head_knockout_profiles(
        self, clean: Sequence[Case], attacked: Sequence[Case], *,
        components: Sequence[tuple[int, int]],
    ) -> list[dict]:
        """The decision with an attention head removed, per component and per case.

        The complement of :meth:`marker_patch_profiles`: that one asks *where* the
        pathway is cut, this one asks *which head* carries it. For each pair
        ``(cleancase, attackedcase)`` it takes the clean and the attacked decision,
        then re-runs both with one head's contribution removed.

        A knockout **zeroes the head's slice of the input to its layer's ``Wo``**,
        which is exactly "this head contributes nothing": `ModernBertAttention`
        reshapes the attention output as ``[batch, seq, n_heads, head_dim]`` before
        ``Wo``, so zeroing ``head * head_dim : (head + 1) * head_dim`` removes the
        head and nothing else. Zero-ablation, not mean-ablation: it is the harsher
        of the two and it needs no reference corpus, so it is the one that can be
        recomputed exactly.

        ``components`` are ``(layer, head)`` pairs over the **encoder** (28 x 16 =
        448 here), not over the decision head's own two layers.

        Per case the row carries the two reference decisions, whether the clean one
        is correct, and the exceptions only: a component whose two knockouts change
        nothing is absent. A run over 448 components on 150 cases is then a small
        record instead of 448 rows mostly containing "unchanged".
        """
        if getattr(self._agent, "_fast", None) is not None:
            raise RuntimeError(
                "head_knockout_profiles non interviene con --fast: laya.fast sostituisce "
                "`model.forward`, quindi i ganci sui moduli dell'attenzione non "
                "scatterebbero e ogni knockout verrebbe ignorato in silenzio. "
                "Rilanciare senza --fast."
            )
        if self.compiled:
            raise RuntimeError(
                "head_knockout_profiles non interviene su un modello compilato: "
                "l'intervento passa da ganci sui moduli e un grafo compilato non "
                "garantisce che scattino. Rilanciare senza --compile."
            )
        if len(clean) != len(attacked):
            raise ValueError(f"{len(clean)} casi puliti per {len(attacked)} attaccati")
        wanted = [(int(layer), int(head)) for layer, head in components]
        if not wanted:
            raise ValueError("nessuna testa da knockare")
        n_layers = len(self._agent.model.encoder.layers)
        n_heads = self._n_heads()
        for layer, head in wanted:
            if not 0 <= layer < n_layers or not 0 <= head < n_heads:
                raise ValueError(
                    f"testa ({layer}, {head}) fuori range: 0..{n_layers - 1} livelli, "
                    f"0..{n_heads - 1} teste"
                )
        return [
            self._head_knockout_case(c, a, wanted, n_heads) for c, a in zip(clean, attacked)
        ]

    def _n_heads(self) -> int:
        """How many heads the loaded encoder has, read from the module and not the config."""
        wo = self._agent.model.encoder.layers[0].attn.Wo
        return wo.in_features // self._agent.model.encoder.layers[0].attn.head_dim

    def _head_knockout_case(
        self, clean: Case, armed: Case, components: list[tuple[int, int]], n_heads: int
    ) -> dict:
        """One case's knockout rows: the two references, then the exceptions."""
        clean_batch, attack_batch, options, gold_label = self._ko_batches(clean, armed)

        clean_index = self._marker_decision(clean_batch, options)
        attacked_index = self._marker_decision(attack_batch, options)
        exceptions = []
        for layer, head in components:
            row: dict = {"l": layer, "h": head}
            clean_ko = self._head_zeroed_decision(clean_batch, options, layer, head)
            if clean_ko != clean_index:
                row["clean"] = options[clean_ko]
            attack_ko = self._head_zeroed_decision(attack_batch, options, layer, head)
            if attack_ko != attacked_index:
                row["attack"] = options[attack_ko]
            if len(row) > 2:
                exceptions.append(row)
        return {
            "clean": options[clean_index],
            "attacked": options[attacked_index],
            # Il recupero si conta solo dove c'era qualcosa da recuperare: un caso che
            # il pulito sbaglia gia' non e' nell'ASR, e "tornare indietro" li'
            # significherebbe solo "il modello ha riperso la sua predizione".
            "clean_correct": options[clean_index] == gold_label,
            "n_components": len(components),
            "exceptions": exceptions,
        }

    def _head_zeroed_decision(
        self, batch: Any, options: list[str], layer: int, head: int
    ) -> int:
        """The decision this batch makes with one attention head removed."""
        handle = self._register_head_zero(layer, head)
        try:
            return self._marker_decision(batch, options)
        finally:
            handle.remove()

    def heads_knockout_together(
        self, clean: Case, armed: Case, components: Sequence[tuple[int, int]]
    ) -> dict:
        """The pair of decisions with a whole **set** of heads removed.

        The companion of :meth:`head_knockout_profiles`: that one removes a head at a
        time, and a mechanism spread over several heads looks like "no single head
        matters". This one removes the set together, which is what tells a distributed
        mechanism from an irrelevant one, and the only number a defence acting on those
        heads would act on.

        `together_clean` is `None` when removing the set leaves the clean decision
        where it was: same convention of `exceptions` in
        :meth:`head_knockout_profiles`, so a record dice una cosa sola.
        """
        clean_batch, attack_batch, options, gold_label = self._ko_batches(clean, armed)
        clean_index = self._marker_decision(clean_batch, options)
        attacked_index = self._marker_decision(attack_batch, options)
        together_clean = self.heads_zeroed_decision(clean_batch, options, list(components))
        together_attack = self.heads_zeroed_decision(attack_batch, options, list(components))
        return {
            "clean": options[clean_index],
            "attacked": options[attacked_index],
            "clean_correct": options[clean_index] == gold_label,
            "together": options[together_attack],
            "together_clean": (None if together_clean == clean_index
                               else options[together_clean]),
        }

    def _ko_batches(
        self, clean: Case, armed: Case
    ) -> tuple[Any, Any, list[str], str]:
        """The two batches of a knockout pair, and the options the labels index into.

        Shared by the per-head and the whole-set knockouts: both need the clean and the
        attacked forward on the same case, and both need the marker positions to agree
        (they live in the head, which the attack does not touch, so a disagreement
        means the case is not comparable).
        """
        qid = next(iter(armed.questions))
        options = list(armed.questions[qid]["criteria"])
        gold_label = armed.gold[qid]
        if gold_label not in options:
            raise RuntimeError(f"gold {gold_label!r} non è fra le opzioni {options}")
        clean_batch, clean_pos, n_markers = self._marker_batch(clean)
        attack_batch, attack_pos, n_attack = self._marker_batch(armed)
        if n_attack != n_markers or attack_pos.tolist() != clean_pos.tolist():
            raise RuntimeError(
                f"marker diversi fra pulito e attaccato ({clean_pos.tolist()} contro "
                f"{attack_pos.tolist()}): il caso {armed.id} non è confrontabile"
            )
        return clean_batch, attack_batch, options, gold_label

    def heads_zeroed_decision(
        self, batch: Any, options: list[str], components: Sequence[tuple[int, int]]
    ) -> int:
        """The decision with **several** heads removed at once.

        The companion of :meth:`head_knockout_profiles`: that one removes a head at a
        time to find which ones matter, this one removes a set together, which is the
        only way to tell "no single head matters" from "the effect is spread over a set
        that does matter" (the difference between redundancy and a distributed
        mechanism, and the number a defence would act on).
        """
        handles = [self._register_head_zero(layer, head) for layer, head in components]
        try:
            return self._marker_decision(batch, options)
        finally:
            for handle in handles:
                handle.remove()

    def _register_head_zero(self, layer: int, head: int):
        """The knockout itself: zero head `head`'s slice of the input to `Wo`.

        A forward **pre** hook on the attention module's output projection. The width of
        the slice comes from the module (`head_dim`), not from the config, and the hook
        is registered per knockout and removed right after, so the next one starts from
        the intact model.
        """
        attn = self._agent.model.encoder.layers[layer].attn
        head_dim = int(attn.head_dim)
        start = head * head_dim
        stop = start + head_dim

        def zero_head(_module, args, kwargs):
            hidden = args[0]
            patched = hidden.clone()
            patched[..., start:stop] = 0
            return (patched, *args[1:]), kwargs

        return attn.Wo.register_forward_pre_hook(zero_head, with_kwargs=True)

    def head_slice_layout(self, batch: Any, options: list[str]) -> dict:
        """Is the slice this knockout zeroes really that head's contribution?

        The knockout zeroes head ``h``'s slice of the input to ``Wo``. An independent
        way to remove the same head is to zero the **value** rows of head ``h`` in
        ``Wqkv``: the head then attends to nothing, its output is exactly zero, and its
        contribution to ``Wo`` is exactly zero. Two different interventions, one
        intended effect: if the two disagree on any head, the slice is on the wrong
        columns and every "this head carries the attack" number would be about another
        head.

        Run once per experiment, on one case, and published: the check is what makes the
        head index mean something.
        """
        attn = self._agent.model.encoder.layers[0].attn
        n_heads = self._n_heads()
        head_dim = int(attn.head_dim)
        reference = self._marker_decision(batch, options)
        by_slice, by_qkv = [], []
        for head in range(n_heads):
            by_slice.append(self._head_zeroed_decision(batch, options, 0, head))
            handle = self._register_qkv_head_zero(0, head, n_heads, head_dim)
            try:
                by_qkv.append(self._marker_decision(batch, options))
            finally:
                handle.remove()
        agree = [i for i, (a, b) in enumerate(zip(by_slice, by_qkv)) if a == b]
        return {
            "n_heads": n_heads,
            "head_dim": head_dim,
            "reference_decision": options[reference],
            "knockout_by_wo_slice": [options[i] for i in by_slice],
            "knockout_by_qkv_values": [options[i] for i in by_qkv],
            "heads_where_the_two_agree": len(agree),
            "layout_ok": len(agree) == n_heads,
        }

    def _register_qkv_head_zero(self, layer: int, head: int, n_heads: int, head_dim: int):
        """The independent path: zero head `head`'s value rows in `Wqkv`.

        `Wqkv`'s output is laid out ``[batch, seq, 3, n_heads, head_dim]`` (the reshape
        in `ModernBertAttention.forward`), so the value rows of head ``h`` are the
        columns ``2 * n_heads * head_dim + h * head_dim ...``.
        """
        wqkv = self._agent.model.encoder.layers[layer].attn.Wqkv
        start = 2 * n_heads * head_dim + head * head_dim
        stop = start + head_dim

        def zero_values(_module, args, kwargs):
            x = args[0]
            patched = x.clone()
            patched[..., start:stop] = 0
            return (patched, *args[1:]), kwargs

        return wqkv.register_forward_pre_hook(zero_values, with_kwargs=True)

    # ------------------------------------------------------------------ router
    @staticmethod
    def route(states: list[str], questions: dict[str, Any]) -> list[dict]:
        """Route-only decisions (no checkpoint load): which model would answer.

        10-modello/checkpoint-laya.md — the router is part of the attack
        surface (20-attack-surface/routing-script-bypass.md).
        """
        import laya

        router = laya.Router(default="english", auto_task_detection=False, max_loaded=0)
        out = []
        for state in states:
            decision = router.route(state, questions)
            detection = getattr(decision, "detection", None)
            if detection is not None and hasattr(detection, "keys"):
                detection = dict(detection)
            out.append(
                {"state": state[:60], "model": decision.model, "reason": decision.reason,
                 "detection": detection}
            )
        return out

    # ----------------------------------------------------------- reproducibility
    @property
    def cfg(self) -> dict:
        """Config subset relevant to the experiments (budget, device, dtype)."""
        return {
            k: self._agent.cfg.get(k)
            for k in _CFG_KEYS
            if hasattr(self._agent, "cfg") and k in self._agent.cfg
        }

    def fingerprint(self) -> dict:
        """Versions + config + weight hash: logged by every experiment."""
        path = self._weight_path()
        return {
            "checkpoint": self.checkpoint,
            "cfg": self.cfg,
            "device": str(getattr(self._agent, "device", "")),
            "weight_path": str(path) if path else None,
            "weight_sha256": self.weight_sha256(),
        }

    def weight_sha256(self) -> str | None:
        """sha256 of model.safetensors (local path or HF cache), cached."""
        if self._weight_sha256 is None:
            path = self._weight_path()
            if path is not None:
                self._weight_sha256 = hashing.sha256_file(path)
        return self._weight_sha256

    def _weight_path(self) -> Path | None:
        local = Path(self.checkpoint)
        if local.is_dir() and (local / "model.safetensors").exists():
            return local / "model.safetensors"
        try:
            from huggingface_hub import hf_hub_download

            return Path(hf_hub_download(self.checkpoint, "model.safetensors"))
        except Exception:
            return None

    def close(self) -> None:
        """Release the checkpoint (E01 sweeps several models in one process)."""
        del self._agent
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ----------------------------------------------------------------- helpers
    def _lens(self) -> tuple[int, int]:
        return (
            int(self._agent.cfg.get("max_len", 1024)),
            int(self._agent.cfg.get("head_max_len", 256)),
        )

    def _internal(self, questions: dict[str, Any]) -> dict[str, dict]:
        """Validate + normalize questions exactly once per call (Laya's own path)."""
        out = {}
        for qid, q in questions.items():
            self._agent._check_question(qid, q)
            out[qid] = self._agent._to_internal(q)
        return out


def open_backend(spec: str, checkpoint: str, **kwargs):
    """Open a decision backend by name, so the model is a hot-swappable parameter.

    ``spec`` selects the backend (``laya`` native, ``laya-onnx`` via ONNX Runtime).
    Only ``laya`` and the laya-family ONNX export expose the same typed decision
    interface; the pair-encoder backends raise at construction if the checkpoint
    has no decision head, instead of pretending.

    Kept here (not only in ``backends.py``) because this module is the single
    place allowed to import the heavy backends, and every experiment goes
    through it.
    """
    from layaadv.model.backends import resolve_backend

    name = resolve_backend(spec)
    if name == "LayaEngine":
        return LayaEngine(checkpoint, **kwargs)
    if name == "ClmLlamaBackend":
        # Terza categoria: nessun peso in processo, due server già su. Va aperto dal
        # nome e non con getattr, altrimenti `--model clm-llama` muore con
        # AttributeError qui, cioè dopo che il flag è stato accettato: la classe di
        # buco di `--model` accettato e ignorato, che è la più costosa del progetto.
        from layaadv.model.clm_backend import ClmLlamaBackend

        return ClmLlamaBackend(checkpoint, **kwargs)
    if name == "D1Backend":
        # Stesso motivo di ClmLlamaBackend, e con un altro dettaglio: d1 liquida i
        # selettori di precisione e i percorsi `compile`/`fast` di laya **prima** di
        # qualsiasi I/O, quindi `--model d1-liquid --quantize nf4` muore qui con il
        # nome del knob sbagliato invece di girare su un bf16 chiamato nf4.
        from layaadv.model.d1_backend import D1Backend

        return D1Backend(checkpoint, **kwargs)
    from layaadv.model import backends

    return getattr(backends, name)(checkpoint, **kwargs)


def open_from_args(args, checkpoint: str) -> Any:
    """Backend e parametri presi dagli argomenti della CLI.

    Un solo punto in cui un esperimento apre un modello: `--model` non puo' essere
    accettato e ignorato. Se serve un checkpoint diverso lo si passa qui, e la
    firma resta quella del backend scelto.
    """
    return open_backend(
        getattr(args, "model", None) or "laya",
        checkpoint,
        device=getattr(args, "device", None),
        batch_size=getattr(args, "batch_size", 16),
        compile=bool(getattr(args, "compile", False)),
        fast=bool(getattr(args, "fast", False)),
        quantize=getattr(args, "quantize", None) or "none",
        # `variant` è la precisione dei grafi ONNX (deberta-onnx). Senza inoltrarlo il
        # backend carica il suo default: quattro configurazioni diverse avrebbero prodotto
        # quattro righe identiche e la falsa conclusione "la quantizzazione non cambia niente".
        variant=getattr(args, "variant", None) or "fp32",
    )


def question_signature(questions: dict[str, Any]) -> str:
    """Stable grouping key for batching (see hashing.sha256_canonical_json)."""
    return hashing.sha256_canonical_json(questions)


def group_cases(cases: list[Case]) -> list[list[tuple[int, Case]]]:
    """Partition cases by identical questions, preserving input order inside
    each group (one Laya ``predict_batch`` call per group)."""
    groups: dict[str, list[tuple[int, Case]]] = {}
    for i, case in enumerate(cases):
        groups.setdefault(question_signature(case.questions), []).append((i, case))
    return list(groups.values())


def _find_subsequence(seq: Sequence[int], needle: Sequence[int]) -> list[int]:
    """Every start index where ``needle`` appears contiguously in ``seq``.

    Used to locate an injected sentence inside the sequence laya actually built.
    Tokenisation is not compositional, so the sentence's ids can differ by a token
    where it joins the state; that is why the caller also keeps a positional
    definition and records which one it used, instead of trusting the match.
    """
    if not needle or len(needle) > len(seq):
        return []
    first = needle[0]
    hits = []
    for start in range(0, len(seq) - len(needle) + 1):
        if seq[start] != first:
            continue
        if list(seq[start : start + len(needle)]) == list(needle):
            hits.append(start)
    return hits


def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    """How many leading tokens two sequences share.

    This is the boundary of an injected sentence, and it is exact where a
    token-id match is not: a sentence at the tail of a state shares a prefix with
    the clean state's tokens and diverges exactly where the sentence begins, even
    when the junction merges the sentence's first token into the previous word.
    """
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index
