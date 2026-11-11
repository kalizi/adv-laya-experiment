"""CLM (Contrastive Language Models) served over HTTP by llama.cpp + clm-serve.

Why this is a separate category
-------------------------------
The ONNX backends are *in-process*: they load a graph and return logits. CLM
cannot be, on a 10 GB card. `Qwen3-8B-Q6_K` is 6.54 GiB of weights and
`-ngl 36` measured **9658 MiB used, 269 MiB free** on the RTX 3080, so the
projection heads do not fit in VRAM next to it (`--device cpu --action-cache 0`
are mandatory, not preferences). The in-process `clm.Engine` makes this worse,
not better: it moves the heads to CUDA by default and dies asking for 24 MiB,
because `--device` is a *server* flag it never reads.

So this backend talks to two already-running servers and owns no weights:

    llama-server  --embedding --pooling last ...   :8090   Qwen3-8B encoder
    clm-serve     --device cpu --action-cache 0     :8700   heads + softmax

Consequences that are conditions, not details, and all of them go into
``run_identity`` because each one moves the numbers:

1. **Quantisation is a condition.** The GGUF file *is* the precision. There is
   no `--quantize` knob here and asking for one raises: the file on disk already
   is Q6_K. The level comes from the server's own `/props` (`model_ftype`), not
   from the file name.
2. **The runtime is a condition.** llama.cpp bf16 does not reproduce vLLM bf16
   to the digit (`czl/CLM-v0.1-8B-GGUF` measured the same invoice case coming out
   more confident on llama.cpp). Every comparison is against a same-runtime
   reference, so `build_info` is recorded, not assumed.
3. **The logits are 100x cosine.** `exp(logit_scale)` is 100.81 clamped to 100.0
   (`clm/heads.py:95`), i.e. temperature 0.01. `argmax` is invariant to that, so
   ASR is governed by the same geometry as laya; the *confidence* channel is not,
   which is why `confidence_definition` is declared and why the margin metric
   exists.

Composition of state and candidate texts is **not reimplemented**: `clm.schema.
build_pairs` is imported, because that function encodes the layout the heads
were trained on (`context + "\n\n" + question`, candidates verbatim, key when the
description is empty). Two copies of that contract would drift silently, which
is the same reason `LayaTokenBudget` exists for laya.

Deliberately **not** a subclass of `LayaTokenBudget`: every method there either
imports `laya.agent` or encodes laya's marker-head semantics, and inheriting it
would leave a laya fallback reachable on a model that has no marker head.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Sequence

from layaadv import hashing, paths
from layaadv.schema import Case, Prediction

DEFAULT_ENCODER_URL = "http://127.0.0.1:8090"
DEFAULT_CLM_URL = "http://127.0.0.1:8700"


def _get_json(url: str, timeout: float = 10.0) -> dict:
    """Small JSON GET. Fails loud: a silent empty dict would look like a valid
    server that simply declares nothing, and the run would proceed undeclared."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.load(response)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"CLM backend: cannot reach {url} ({type(error).__name__}: {error}). "
            "Both servers must be up before the run: "
            "llama-server --embedding --pooling last -m <gguf> --port 8090 and "
            "clm-serve --device cpu --action-cache 0 --port 8700."
        ) from error


class ClmLlamaBackend:
    """CLM-v0.1-8B behind ``clm-serve``, encoder behind ``llama-server``."""

    name = "clm-llama"

    def __init__(
        self,
        checkpoint: str | None = None,
        *,
        base_url: str | None = None,
        encoder_url: str | None = None,
        service_model: str = "clm-latest",
        device: str | None = None,
        batch_size: int = 16,
        compile: bool = False,
        fast: bool = False,
        quantize: str = "none",
        variant: str = "fp32",
        config_path: str | None = None,
        **_,
    ) -> None:
        if checkpoint and checkpoint.startswith("convaiinnovations/laya"):
            raise ValueError(
                "CLM ha ricevuto un checkpoint Laya: passa --checkpoint clm-v0.1-8B "
                "(o l'identità CLM effettivamente servita)"
            )
        # Symmetric with deberta-onnx refusing `--quantize` and LayaEngine refusing
        # `--variant`: this backend knows exactly one way to choose its precision and
        # it is the GGUF file on disk. Ignoring the other selector silently produced
        # the "quantisation changes nothing" table in E15, i.e. a null result built
        # by hand out of a plumbing error.
        if quantize not in (None, "", "none"):
            raise ValueError(
                "per clm-llama la precisione e' il file GGUF che si avvia: non c'e' un "
                "selettore --quantize. Q6_K e Q5_K_M sono due file diversi e due condizioni "
                "diverse, e vanno dichiarati come tale. --quantize vale solo per la "
                "famiglia laya."
            )
        if variant not in (None, "", "fp32"):
            raise ValueError(
                "clm-llama non ha una famiglia di grafi: --variant vale solo per "
                "deberta-onnx (l'export open-jev)."
            )
        if compile or fast:
            raise ValueError(
                "clm-llama non gira laya: --compile e --fast sono accelerazioni del suo "
                "forward e non hanno un corrispondente qui. Accettarli e ignorarli e' "
                "il tipo di errore piu' costoso del progetto."
            )

        self.checkpoint = checkpoint or service_model
        self.batch_size = batch_size
        self.service_model = service_model
        self._clm_url = (base_url or os.environ.get("CLM_BASE_URL") or DEFAULT_CLM_URL).rstrip("/")
        self._encoder_url = (
            encoder_url or os.environ.get("CLM_ENCODER_URL") or DEFAULT_ENCODER_URL
        ).rstrip("/")
        self._encoder_props = _get_json(f"{self._encoder_url}/props")
        self._clm_health = _get_json(f"{self._clm_url}/health")
        if not self._clm_health.get("embedder", False):
            raise RuntimeError(
                "clm-serve è su ma il suo embedder non risponde: "
                f"{self._clm_health}. Il backend misurerebbe l'assenza di un encoder come "
                "se fosse una previsione."
            )

        self._cfg = self._load_contract(config_path)
        self._load_warnings: list[str] = []
        self._observe_device(device)
        self._tok = None
        self._tok_id: str | None = None
        self._sha_cache: dict[str, str] = {}
        self.compiled = False

    # ------------------------------------------------------------------ contract
    def _load_contract(self, config_path: str | None) -> dict:
        """`clm_config.json`: the pooling/embedding/scale contract no GGUF key carries.

        The GGUF bakes `pooling_type` but nothing declares the temperature, the
        normalisation or which side truncates. That file is the only place all of
        it is written down, so a run without it is a run whose geometry is guessed.
        """
        candidates: list[Path] = []
        if config_path:
            candidates.append(Path(config_path).expanduser())
        env = os.environ.get("CLM_CONFIG")
        if env:
            candidates.append(Path(env).expanduser())
        model_dir = os.environ.get("CLM_MODEL_DIR")
        candidates.append(Path(model_dir).expanduser() / "clm_config.json" if model_dir else None)
        candidates.append(paths.CLM_CONFIG)
        for path in [c for c in candidates if c is not None]:
            if path.exists():
                cfg = json.loads(path.read_text())
                cfg["_path"] = str(path)
                return cfg
        raise FileNotFoundError(
            "clm_config.json non trovato: dichiara pooling, normalizzazione, "
            f"max_context_tokens, logit_scale e il lato da cui si tronca. Cercato in "
            f"{[str(c) for c in candidates if c is not None]}. Senza questo contratto i "
            "numeri sono indovinati."
        )

    @property
    def cfg(self) -> dict:
        return self._cfg

    @property
    def load_warnings(self) -> list[str]:
        out = list(self._load_warnings)
        scale = self.logit_scale
        out.append(
            f"clm: logit_scale={scale} (temperatura {1.0 / scale:.5f}). L'argmax e' "
            "invariante alla scala quindi l'ASR non ne dipende, ma la confidenza sì: "
            "una soglia fissa su una probabilita' non discrimina piu'. "
            "Usare oer_margin e dichiarare il punto operativo."
        )
        if str(self._cfg.get("truncation", "")).startswith("head"):
            out.append(
                "clm: il serving tronca dalla TESTA dello stato (asimmetria train/serve "
                "dichiarata da upstream): un'iniezione in prepend e' quella a rischio "
                "di essere tagliata, append no."
            )
        return out

    @property
    def logit_scale(self) -> float:
        """`exp(logit_scale)`, the multiplier on cosine before the softmax."""
        return float(self._cfg.get("logit_scale", 100.0))

    @property
    def max_context_tokens(self) -> int:
        return int(self._cfg.get("max_context_tokens", 2048))

    @property
    def head_max_len(self) -> int:
        """CLM has no marker head. The only window is the encoder context.

        Returned as the "head" bound so `loaders.check_head_budget` sweeps and
        stops at a wall instead of running to K=40 on a checkpoint whose options
        do not fit. It is a different wall from laya's (state, not option head)
        and `batching_report`/`state_budget` say which.
        """
        return self.max_context_tokens

    def _lens(self) -> tuple[int, int]:
        return (self.max_context_tokens, self.head_max_len)

    # -------------------------------------------------------------------- device
    def _observe_device(self, requested: str | None) -> str:
        """Where the encoder actually is, checked rather than assumed.

        `llama-server` does not expose `n_gpu_layers` on `/props`, so the claim
        rests on the launch flags plus the **measured** VRAM occupancy of the
        compute processes. The second is the falsifiable half: a 6.5 GiB weight
        file cannot be resident in a process holding a few hundred MiB, so
        occupancy near the file size is evidence of the GPU, and occupancy near
        zero would mean a silent CPU demotion. In that case this returns "cpu"
        and warns, because every latency in the run would then be off by an order
        of magnitude while `env_fingerprint` still advertised an RTX 3080 -- the
        device-outside-the-run_id bug, in its slowest form.
        """
        vram = _process_vram_mib()
        weights_mib = self._weight_bytes() // (1 << 20)
        ftype = self._encoder_props.get("model_ftype", "?")
        resident = vram >= 0.8 * weights_mib if weights_mib else vram > 0
        where = "gpu" if resident else "cpu"
        self._device_seen = (
            f"llama.cpp:{where}({ftype}) vram={vram}MiB weights={weights_mib}MiB"
        )
        if not resident:
            self._load_warnings.append(
                f"clm: l'encoder NON sembra in GPU (vram {vram} MiB occupati contro un "
                f"file da {weights_mib} MiB). O gira su CPU, o su una GPU diversa "
                "da quella dichiarata: ogni latenza di questa run e' non confrontabile. "
                "Il device richiesto era "
                f"{requested or 'automatico'}."
            )
        return self._device_seen

    def _weight_bytes(self) -> int:
        """Size on disk of the GGUF the server says it loaded."""
        name = str(self._encoder_props.get("model_path") or "")
        for candidate in (Path(name), paths.CLM_DIR / name):
            if candidate.exists():
                return candidate.stat().st_size
        return 0

    def runtime_state(self) -> dict:
        """Per-sample device, for the latency sidecar.

        Two devices are honest here and saying so is the point: the encoder is on
        the GPU, the heads are on CPU because there was no VRAM left for them. A
        single `device` string would hide a split that changes the latency.
        """
        return {
            "device": self._device_seen,
            "dtype": f"gguf:{self._encoder_props.get('model_ftype', '?')}",
            "heads_device": "cpu",
        }

    def batching_report(self, cases: Sequence[Case]) -> dict:
        """CLM's batching is request concurrency, not a padded forward pass.

        `llama-server` has `total_slots` continuous-batching slots, so the lever
        is how many cases are in flight, not `--batch-size` in laya's sense. Unlike
        massive-it on laya, this is **not** inert: it is recorded as
        `concurrency` and whether it moves predictions is measured, not assumed
        (see `tools/clm_concurrency_check.py`).
        """
        groups = {}
        for case in cases:
            groups.setdefault(
                hashing.sha256_canonical_json(case.questions), 0
            )
            groups[hashing.sha256_canonical_json(case.questions)] += 1
        sizes = sorted(groups.values(), reverse=True)
        return {
            "states": len(cases),
            "question_groups": len(groups),
            "max_states_per_group": sizes[0] if sizes else 0,
            "batch_size_effective": min(self.batch_size, self.encoder_slots),
            "batch_size_requested": self.batch_size,
            "batching_is_inert": False,
            "concurrency": min(self.batch_size, self.encoder_slots),
            "encoder_slots": self.encoder_slots,
            "note": (
                "CLM non fa un forward con padding: ogni caso e' una richiesta HTTP e "
                "il lever e' la concorrenza sui slot di llama-server."
            ),
        }

    @property
    def encoder_slots(self) -> int:
        return int(self._encoder_props.get("total_slots", 1) or 1)

    # ----------------------------------------------------------------- tokenizer
    def _tokenizer(self):
        """Qwen3 tokenizer, from the `base_model` the contract declares.

        `czl/CLM-v0.1-8B-GGUF` verified `llama-tokenize` and `AutoTokenizer` emit
        byte-identical ids with no BOS and no EOS, so counting tokens with the HF
        tokenizer is counting what the encoder saw. The heads were trained on
        `add_special_tokens=False` text, so a BOS here would be a silent mismatch.
        """
        if self._tok is None:
            from transformers import AutoTokenizer

            base = str(self._cfg.get("base_model") or "Qwen/Qwen3-8B")
            self._tok = AutoTokenizer.from_pretrained(base)
            self._tok_id = base
        return self._tok

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer()(text, add_special_tokens=False)["input_ids"])

    # -------------------------------------------------------------------- budget
    def state_budget(self, questions: dict[str, Any]) -> dict[str, int]:
        """Room left in the encoder window for the state, per question.

        Unlike laya there is no marker head and no SEP: the state head reads
        `state + "\\n\\n" + instructions` as one sequence truncated at
        `max_context_tokens`. So the budget is the window minus the question text.
        """
        from clm.schema import build_pairs

        empty = {qid: build_pairs("", {qid: q})[qid][0] for qid, q in questions.items()}
        return {
            qid: max(0, self.max_context_tokens - self.count_tokens(text))
            for qid, text in empty.items()
        }

    def head_text(self, questions: dict[str, Any]) -> dict[str, dict]:
        """Options as CLM feeds them: one independent sequence per candidate.

        `check_head_budget` asks "do the options and the gold survive". For CLM
        the answer is per-candidate against the same 2048 window, so the honest
        report is the longest candidate and its own length. Options are short, so
        this reports no wall -- which is the correct answer, not a missing check.
        """
        from clm.schema import candidates

        out: dict[str, dict] = {}
        for qid, q in questions.items():
            keys, texts = candidates(q)
            lengths = [self.count_tokens(t) for t in texts]
            out[qid] = {
                "text": "\n".join(texts),
                "head_tokens": max(lengths) if lengths else 0,
                "option_tokens": dict(zip(keys, lengths)),
                "rendering": "un candidato per sequenza, senza prefisso",
            }
        return out

    def render_options(self, questions: dict[str, Any]) -> dict[str, list[str]]:
        from clm.schema import candidates

        return {qid: list(candidates(q)[1]) for qid, q in questions.items()}

    # -------------------------------------------------------------------- probes
    def probe_sequence(
        self, state: str, questions: dict[str, Any], *, marker: str, truncate_left: bool = False
    ) -> list[dict]:
        """Does a marker survive CLM's truncation? Real probe, not a stub.

        CLM's serving path keeps the **first** `max_context_tokens` tokens
        (`"truncation": "head"` in the contract), so `truncate_left=True` is the
        opposite of what it does. This is the probe that makes the upstream
        train/serve asymmetry measurable on our own states.
        """
        from clm.schema import build_pairs

        tok = self._tokenizer()
        rows = []
        for qid, (state_text, _keys, _texts) in build_pairs(state, questions).items():
            ids = tok(state_text, add_special_tokens=False)["input_ids"]
            kept = ids[: self.max_context_tokens]
            rows.append(
                {
                    "qid": qid,
                    "seq_tokens": len(ids),
                    "state_tokens": self.count_tokens(state),
                    "kept_tokens": len(kept),
                    "marker_visible": marker in tok.decode(kept),
                    "truncate_left": bool(truncate_left),
                    "note": "CLM tronca dalla testa: il marker in coda sparisce per primo",
                }
            )
        return rows

    def probe_head(self, questions: dict[str, Any]) -> list[dict]:
        raise NotImplementedError(
            "clm-llama non ha una testa a marker: le opzioni sono candidate indipendenti, "
            "non marcatori in una sequenza condivisa. Usa probe_sequence."
        )

    def decision_temperature(self, primitive: str, k: int) -> float:
        """CLM's effective temperature, for the per-K row.

        `1 / logit_scale` = 0.01. There is **one** temperature for every primitive
        and every K, unlike laya's `temperature_by_options` buckets that switch at
        K=5/10/11. Recording it per K is therefore constant here, and that constancy
        is the useful fact: an E03 curve on CLM does not cross a temperature change,
        so absolute OER across K is comparable on CLM in a way it is not on
        `laya-typed-decisions`.
        """
        return round(1.0 / self.logit_scale, 6)

    # ------------------------------------------------------------------ identity
    def weight_sha256(self) -> str | None:
        """sha256 of the GGUF actually loaded, from the path the server reports.

        Cached: the file is 6.5 GiB and hashing it three times per K is a minute
        of pure waste. The name of the loaded model is taken from `/props`, not
        from what we asked for, so a server left running on a different file is
        caught instead of silently measured.
        """
        path = str(self._encoder_props.get("model_path") or "")
        if not path:
            return None
        if path not in self._sha_cache:
            candidates = [
                Path(path),
                Path(os.environ.get("CLM_MODEL_DIR", "")) / path if os.environ.get("CLM_MODEL_DIR") else None,
                paths.CLM_DIR / path,
            ]
            found = next((c for c in candidates if c is not None and c.exists()), None)
            self._sha_cache[path] = (
                hashing.sha256_file(found) if found else f"unresolved:{path}"
            )
        return self._sha_cache[path]

    def head_sha256(self) -> str | None:
        """sha256 of the projection head checkpoint, when it is on disk.

        Encoder and heads are separate artefacts and only the pair is a model:
        CLM's own docs say a head "only makes sense with the encoder and pooling it
        was trained against", so a run that records one without the other is not
        reproducible.
        """
        path = paths.CLM_HEADS
        key = str(path)
        if not path.exists():
            return None
        if key not in self._sha_cache:
            self._sha_cache[key] = hashing.sha256_file(path)
        return self._sha_cache[key]

    def run_identity(self) -> dict:
        """What changes the numbers, for a run config."""
        return {
            "backend": self.name,
            "compiled": False,
            "fast": False,
            # CLM's own API defines it as top-1 minus the mean of the others, i.e.
            # a margin. A third definition next to laya's two, which is exactly why
            # it has to be written down: an `OER@0.75` read off a CLM row and one
            # read off a laya row would look like the same number.
            "confidence_definition": "p1 - mean(p2..pk)  [CLM API definition]",
            "quantization": self._encoder_props.get("model_ftype", "unknown"),
            "variant": None,
            "logit_scale": self.logit_scale,
            "temperature_effective": self.decision_temperature("choice", 0),
            "pooling": self._cfg.get("encoder_pooling", "last-token"),
            "embedding_normalisation": self._cfg.get("embedding_normalisation", ""),
            "truncation": self._cfg.get("truncation", ""),
            "max_context_tokens": self.max_context_tokens,
            "service_model": self.service_model,
            "encoder_runtime": self._encoder_props.get("build_info", "unknown"),
            "encoder_model": self._encoder_props.get("model_path", "unknown"),
            "encoder_slots": self.encoder_slots,
            "base_model": self._cfg.get("base_model", "unknown"),
            "device": self._device_seen,
            "heads_device": "cpu",
        }

    def fingerprint(self) -> dict:
        return {
            "checkpoint": self.checkpoint,
            "encoder_weight_sha256": self.weight_sha256(),
            "heads_sha256": self.head_sha256(),
            "contract": self._cfg.get("_path"),
            "encoder_build": self._encoder_props.get("build_info"),
            "encoder_ftype": self._encoder_props.get("model_ftype"),
            "device": self._device_seen,
        }

    # ---------------------------------------------------------------- prediction
    def predict_batch(
        self, cases: list[Case], batch_size: int | None = None
    ) -> list[Prediction]:
        """One HTTP `system_one` call per case, up to `concurrency` in flight.

        Concurrency is the only batching lever: llama-server continuous-batches
        across `total_slots`. Whether that moves predictions is measured by
        `tools/clm_concurrency_check.py` rather than assumed, because a changed
        batch composition moved bf16 predictions by ~0.003 on laya.
        """
        from concurrent.futures import ThreadPoolExecutor

        from clm import Choice, CLMClient, Noul, Score

        if not cases:
            return []
        workers = max(1, min(batch_size or self.batch_size, self.encoder_slots))

        def as_question(q: dict[str, Any]):
            kind = q.get("type")
            if kind == "choice":
                return Choice(instructions=q.get("instructions", ""), criteria=dict(q["criteria"]))
            if kind == "noul":
                crit = q.get("criteria") or {}
                return Noul(instructions=q.get("instructions", ""), criteria=dict(crit) or None)
            if kind == "score":
                return Score(instructions=q.get("instructions", ""), criteria=list(q["criteria"]))
            raise ValueError(f"unknown question type {kind!r}")

        def one(case: Case) -> Prediction:
            client = CLMClient(base_url=self._clm_url)
            questions = {qid: as_question(q) for qid, q in case.questions.items()}
            result = client.system_one(state=case.state, questions=questions)
            answers: dict[str, Any] = {}
            labels: dict[str, str] = {}
            for qid, ans in result.answers.items():
                kind = case.questions[qid].get("type")
                probs = {str(k): float(v) for k, v in (ans.probabilities or {}).items()}
                if kind == "noul":
                    answers[qid] = {
                        "type": "noul",
                        "noul": float(probs.get("true", 0.0)),
                        "confidence": float(ans.confidence),
                    }
                    labels[qid] = "true" if float(probs.get("true", 0.0)) >= 0.5 else "false"
                else:
                    # argmax on the returned distribution, keyed by option name.
                    # Not `ans.choice`: that is CLM's own label, and reading it would
                    # make our `answer_label` a pass-through instead of the rule E01
                    # fixes for every backend.
                    if not probs:
                        raise RuntimeError(
                            f"clm: nessuna probabilita' per {qid!r} su {case.id}: "
                            f"{ans}. Un backend che restituisce una risposta senza "
                            "distribuzione fa saltare Brier, ECE e OER senza dirlo."
                        )
                    best = max(probs, key=lambda k: (probs[k], k))
                    answers[qid] = {
                        "type": kind,
                        "choice": best,
                        "probabilities": probs,
                        "confidence": float(ans.confidence),
                    }
                    labels[qid] = best
            usage = {}
            try:
                usage = {
                    "input_tokens": int(result.usage.input_tokens),
                    "latency_ms": float(result.latency_ms or 0.0),
                }
            except AttributeError:
                usage = {}
            return Prediction(id=case.id, answers=answers, labels=labels, usage=usage)

        if workers == 1:
            return [one(c) for c in cases]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(one, cases))

    def predict(self, case: Case) -> Prediction:
        return self.predict_batch([case], batch_size=1)[0]

    def close(self) -> None:
        """Nothing to close: this backend owns no weights and no server.

        It refuses to start them too, on purpose. A backend that silently spawns
        an 8B model would make the process table the only record of which
        quantisation ran, and the experiment's own config would be a lie.
        """


def _process_vram_mib() -> int:
    """Total MiB currently held by compute processes on the GPU, or 0 if unknown.

    Used only as evidence for `device`, never as the device itself: it cannot say
    *which* model is resident, only that something large is.
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return sum(int(x) for x in out.stdout.split() if x.strip().isdigit())
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0
