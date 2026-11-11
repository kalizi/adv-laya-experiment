"""Hot-swappable decision backends.

`engine.py` resta l'unico modulo che tocca le API private di laya. Qui stanno
i **backend alternativi**, che espongono la stessa interfaccia di
:class:`LayaEngine` (``predict``, ``predict_batch``, ``count_tokens``,
``render_options``, ``close``, ``weight_sha256``) cosi' gli esperimenti non
distinguono il backend e si possono scambiare a caldo con `--model`.

Due famiglie:

- **laya-family**: stessa testa a marker di Laya. Il backend ONNX (laya-multilingual
  esportato) ha esattamente la firma `input_ids, attention_mask, marker_pos,
  marker_mask, qtype` e restituisce `logits, act_probs` gia' calcolati, quindi è
  drop-in. È il backend che rilancia la batteria E03.
- **encoder a coppie** (DeBERTa/open-jev): firma `seg, pair_q, pair_opt` con un
  logit per (domanda, opzione). Non è drop-in: ha bisogno del proprio wrapper che
  assembla le coppie. Serve per testare la **generalizzazione architetturale** della
  legge dei token fuori dalla famiglia Laya.

Import pesanti (torch, onnxruntime, laya) sono tutti lazy: importare
``layaadv.model.backends`` non deve tirare dentro nulla, altrimenti i test e
``--help`` diventano pesanti come il progetto si aspetta.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

from layaadv.model.engine import LayaTokenBudget, group_cases
from layaadv.schema import Case, Prediction, answer_label


@runtime_checkable
class DecisionBackend(Protocol):
    """Interfaccia minima che un backend deve offrire per essere hot-swappable.

    Include la superficie di budget che gli esperimenti usano davvero
    (``check_head_budget``, ``state_budget``, il gate di E00): senza questa parte
    un backend che non la implementa fallisce con un ``AttributeError`` a metà
    run, cioè dopo che i numeri sono gia' parzialmente scritti.
    """

    name: str

    def predict(self, case: Case) -> Prediction: ...

    def predict_batch(self, cases: list[Case], batch_size: int | None = None) -> list[Prediction]: ...

    def count_tokens(self, text: str) -> int: ...

    def render_options(self, questions: dict[str, Any]) -> dict[str, list[str]]: ...

    def close(self) -> None: ...

    def weight_sha256(self) -> str | None: ...

    def state_budget(self, questions: dict[str, Any]) -> dict[str, int]: ...

    def probe_sequence(
        self, state: str, questions: dict[str, Any], *, marker: str, truncate_left: bool = False
    ) -> list[dict]: ...

    def probe_head(self, questions: dict[str, Any]) -> list[dict]: ...

    @property
    def cfg(self) -> dict: ...

    @property
    def load_warnings(self) -> list[str]: ...


# --------------------------------------------------------------------- registry
def _resolve_graph(model_dir: str, name: str):
    """Il grafo puo' stare alla radice del repo o nella sottocartella `onnx/`.

    Le esportazioni HF mettono il .onnx sotto `onnx/` e il tokenizer alla radice,
    quindi i due non vivono insieme: si cerca in entrambi i posti invece di
    imporre un layout a chi esporta.
    """
    from pathlib import Path

    base = Path(model_dir)
    for cand in (base / "onnx" / name, base / name):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"{name} non trovato sotto {base} ne' in onnx/")


def _load_laya_config(model_dir: str) -> dict:
    """`laya_config.json` dell'export: max_len, head_max_len, temperature.

    L'export ONNX porta gia' questo file accanto al grafo, quindi non c'è niente da
    dedurre: i budget e le temperature sono dichiarati, non stimati. Se manca si
    solleva, perche' un fallback silenzioso (i vecchi 1024/256) tronca il checkpoint
    `laya` che dichiara 512/192 senza dire nulla.
    """
    from pathlib import Path

    path = Path(model_dir) / "laya_config.json"
    if not path.exists():
        raise FileNotFoundError(
            f"laya_config.json non trovato sotto {model_dir}: senza i budget e le "
            "temperature dichiarati dal checkpoint questo backend non puo' produrre "
            "numeri confrontabili con il backend nativo"
        )
    return json.loads(path.read_text())


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _temperature_warnings(temperature: list, by_options: dict) -> list[str]:
    """Stessi warning che laya emette al load (E00 li cerca per nome).

    Sono la traccia del fatto che il checkpoint dichiara temperature fuori range:
    sotto 1.0 la distribuzione viene affossata e la confidence letta su uno sbaglio
    non significa quello che sembra.
    """
    from laya.common import TEMP_MAX, TEMP_MIN, clamp_temperature

    entries = [(k, v, clamp_temperature(v)) for k, v in by_options.items()]
    entries += [(f"temperature[{i}]", t, clamp_temperature(t)) for i, t in enumerate(temperature)]
    rejected = []
    for name, raw, applied in entries:
        try:
            if float(raw) == applied:
                continue
        except (TypeError, ValueError):
            pass
        rejected.append(f"{name}={raw!r} -> {applied:g}")
    if not rejected:
        return []
    return [
        "laya: this checkpoint ships invalid temperatures or values outside "
        f"[{TEMP_MIN:g}, {TEMP_MAX:g}]; using {', '.join(rejected)}. Treat confidence "
        "from the affected entries as uncalibrated."
    ]


_BACKENDS: dict[str, str] = {
    "laya": "LayaEngine",  # backend nativo, implementato in engine.py
    "laya-onnx": "OnnxLayaBackend",
    "deberta-onnx": "OnnxPairBackend",
    # Quarta famiglia: `d1` di Liquid AI, un LFM causale post-trainato che legge la
    # risposta da un softmax sui token-codice delle opzioni. Non sta in `backends.py`
    # perche' non eredita `LayaTokenBudget` (ogni metodo di quella classe o importa
    # `laya.agent` o codifica la testa a marker, che d1 non ha) e perche' ereditaria
    # lascerebbe un fallback su laya raggiungibile.
    "d1-liquid": "D1Backend",
    # Terza categoria: nessun peso in processo, due server gia' su. Non sta in
    # `backends.py` perche' non eredita `LayaTokenBudget` (ogni metodo di quella
    # classe o importa `laya.agent` o codifica la testa a marker, che CLM non ha)
    # e perche' ereditaria lascerebbe un fallback su laya raggiungibile.
    "clm-llama": "ClmLlamaBackend",
}


def resolve_backend(spec: str) -> str:
    """Risolve uno spec backend ('laya', 'laya-onnx', ...) in un nome di classe."""
    key = (spec or "laya").strip().lower()
    if key not in _BACKENDS:
        raise KeyError(f"backend sconosciuto {spec!r}; noti: {sorted(_BACKENDS)}")
    return _BACKENDS[key]


# ------------------------------------------------------------------ laya-family ONNX
class OnnxLayaBackend(LayaTokenBudget):
    """Backend laya-family via ONNX Runtime (esportazione di un checkpoint Laya).

    Stessa firma di :class:`LayaEngine` a livello di API, quindi gli esperimenti
    esistenti (E00-E14) funzionano senza modifiche: è un `predict_batch` che
    costruisce le entry laya e le passa al grafo.

    Tre cose devono combaciare con il backend nativo, altrimenti i numeri non
    sono confrontabili pur sembrando uguali:

    1. **Le temperature.** laya divide i logit per ``temperature_by_options``
       prima del softmax (``laya/agent.py``). Senza quel termine le probabilita'
       divergono proprio sui checkpoint che lo hanno valori sotto 1, cioe' dove
       la distribuzione è piu' affossata. Vengono lette da ``laya_config.json``
       nell'export, con lo stesso ``clamp_temperature`` di laya.
    2. **La confidence.** laya la calcola da entropia normalizzata
       (``confidence_from_probs``). Senza, `record_for` mette 0.0 su ogni
       decisione e OER/ECE/mean_conf_wrong diventano zero per costruzione.
    3. **Tutte le domande.** Una riga del grafo per (stato, domanda), come
       ``collate_items``. Rispondere solo alla prima domanda darebbe un
       ``accuracy`` su un quinto dei dati senza che nulla lo segnali.

    Verificato su `laya-multilingual` (8 casi typed-decisions, 5 domande ciascuno,
    40 decisioni): **40/40 label identiche al backend nativo**, `accuracy` e
    `OER@0.75` uguali, `confidence` entro 0.01 su 37/40 con max 0.019. La residua
    è numerica, non logica: il grafo gira in fp32 e il nativo in autocast bf16.
    Ogni label e ogni `accuracy` coincidono, quindi gli attacchi sono confrontabili;
    un delta di OER sotto ~0.02 non lo è.
    """

    name = "laya-onnx"

    def __init__(
        self,
        model_dir: str,
        device: str | None = None,
        batch_size: int = 16,
        fp16: bool = False,
        variant: str = "fp32",
        **_,
    ) -> None:
        import onnxruntime as ort  # lazy

        if variant and variant != "fp32":
            raise ValueError(
                f"variant={variant!r} e' il selettore dei grafi di deberta-onnx; questo "
                "backend ha una sola precisione per checkpoint, scelta da `fp16`."
            )
        from laya.agent import Agent, build_sequence
        from laya.common import QTYPES, clamp_temperature, confidence_from_probs, temp_bucket

        self.name = "laya-onnx"
        self.checkpoint = model_dir
        self.batch_size = batch_size
        self.compiled = False
        # solo i provider effettivamente disponibili: l'onnxruntime-gpu non è
        # installato, e chiedere CUDA in una CPU-only build solleva solo un warning.
        available = set(ort.get_available_providers())
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        suffix = "_fp16" if fp16 else ""
        self._graph = _resolve_graph(model_dir, f"laya{suffix}.onnx")
        self._session = ort.InferenceSession(str(self._graph), providers=providers)
        # tokenizer + costruzione sequenza: riuso laya perche' il formato è identico
        self._build_sequence = build_sequence
        self._qtypes = QTYPES
        self._agent_cls = Agent
        self._confidence_from_probs = confidence_from_probs
        self._temp_bucket = temp_bucket
        self._tok = self._load_tok(model_dir)

        lcfg = _load_laya_config(model_dir)
        self._laya_cfg = lcfg
        self._max_len = int(lcfg.get("max_len", 1024))
        self._head_max_len = int(lcfg.get("head_max_len", 256))
        self._temperature = [clamp_temperature(t) for t in lcfg.get("temperature", [1.0, 1.0, 1.0])]
        self._temperature_by_options = {
            k: clamp_temperature(v) for k, v in lcfg.get("temperature_by_options", {}).items()
        }
        self._temperature_raw = list(lcfg.get("temperature", [1.0, 1.0, 1.0]))
        self._temperature_by_options_raw = dict(lcfg.get("temperature_by_options", {}))
        # stessi warning di laya per temperature fuori range: E00 li cerca per nome
        self.load_warnings = _temperature_warnings(
            lcfg.get("temperature", []), lcfg.get("temperature_by_options", {})
        )
        self._device_seen = "onnx:" + self._session.get_providers()[0]

    @staticmethod
    def _load_tok(model_dir: str):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_dir)

    def _lens(self) -> tuple[int, int]:
        """Presi da ``laya_config.json``, non indovinati: ``laya`` dichiara
        512/192 e ``laya-typed-decisions`` 1024/256, quindi un valore fisso
        troncherebbe (o non troncherebbe) E04 sul checkpoint sbagliato."""
        return self._max_len, self._head_max_len

    @property
    def cfg(self) -> dict:
        return {
            "max_len": self._max_len,
            "head_max_len": self._head_max_len,
            "device": self._device_seen,
            "dtype": "onnxruntime",
            "model_id_or_path": self.checkpoint,
        }

    def _internal(self, questions: dict[str, Any]) -> dict[str, dict]:
        # `_check_question` e `_to_internal` sono @staticmethod su Agent: si
        # chiamano sulla classe, nessun agent da costruire, nessun peso torch
        # caricato, stesso output di `self._agent._internal(questions)`.
        return {qid: self._agent_cls._to_internal(self._check(qid, q)) for qid, q in questions.items()}

    def _check(self, qid: str, qdef: Any) -> Any:
        self._agent_cls._check_question(qid, qdef)
        return qdef

    # ------------------------------------------------------------------ predict
    def predict_batch(self, cases: list[Case], batch_size: int | None = None) -> list[Prediction]:
        """One graph row per (case, question), chunked by *states* like laya.

        The chunk is in states, not rows, so the batch composition matches the
        native backend: in bf16 a different composition can move a prediction.
        """
        if not cases:
            return []
        size = batch_size or self.batch_size
        out: list[Prediction | None] = [None] * len(cases)
        for members in group_cases(cases):
            answers: dict[int, dict] = {}
            usage: dict[int, dict] = {}
            for start in range(0, len(members), size):
                part = members[start : start + size]
                rows = []
                for i, case in part:
                    internal = self._internal(case.questions)
                    for qid, q in internal.items():
                        seq, markers = self._build_sequence(
                            self._tok, case.state, q, self._max_len, self._head_max_len
                        )
                        rows.append((i, qid, q, seq, markers))
                logits, act, n_tokens = self._forward(rows)
                for j, (i, qid, q, _seq, markers) in enumerate(rows):
                    answers.setdefault(i, {})[qid] = self._decode(
                        qid, q, logits[j], float(act[j, 0]), len(markers)
                    )
                    usage.setdefault(i, {"input_tokens": 0, "output_tokens": 0})
                    usage[i]["input_tokens"] += n_tokens // len(part)
            for i, case in members:
                ans = answers.get(i, {})
                out[i] = Prediction(
                    id=case.id,
                    answers=ans,
                    labels={qid: answer_label(a) for qid, a in ans.items()},
                    usage=usage.get(i, {}),
                )
        missing = [cases[i].id for i, p in enumerate(out) if p is None]
        if missing:
            raise RuntimeError(f"missing predictions for: {missing}")
        return [p for p in out if p is not None]

    def _forward(self, rows: list[tuple]) -> tuple:
        import numpy as np

        seqs = [r[3] for r in rows]
        mks = [r[4] for r in rows]
        maxlen = max(len(s) for s in seqs)
        maxopt = max(len(m) for m in mks)
        input_ids = np.zeros((len(rows), maxlen), dtype=np.int64)
        attn = np.zeros((len(rows), maxlen), dtype=np.int64)
        marker_pos = np.zeros((len(rows), maxopt), dtype=np.int64)
        marker_mask = np.zeros((len(rows), maxopt), dtype=bool)
        qtype = np.zeros(len(rows), dtype=np.int64)
        for i, (_idx, _qid, q, seq, mk) in enumerate(rows):
            input_ids[i, : len(seq)] = seq
            attn[i, : len(seq)] = 1
            marker_pos[i, : len(mk)] = mk
            marker_mask[i, : len(mk)] = True
            qtype[i] = self._qtypes[q["t"]]
        logits, act_probs = self._session.run(
            ["logits", "act_probs"],
            {
                "input_ids": input_ids,
                "attention_mask": attn,
                "marker_pos": marker_pos,
                "marker_mask": marker_mask,
                "qtype": qtype,
            },
        )
        return logits, act_probs, int(attn.sum())

    def _decode(self, qid: str, q: dict, row, act_probability: float, k: int) -> dict:
        """Logit row -> the same typed answer laya returns.

        The comparable label comes from the question's own criteria, never from
        ``render_options``: that renders ``"label: description"``, and splitting
        on the first colon mangles every label that contains one (E13's `prefix`
        arm is made of labels like ``"I7: calendar_query"``).
        """
        import numpy as np

        t_scale = self._temperature_by_options.get(
            self._temp_bucket(self._qtypes[q["t"]], k), self._temperature[self._qtypes[q["t"]]]
        )
        z = row[:k] / t_scale
        p = np.exp(z - z.max())
        p = p / p.sum()
        ext = {"act_probability": round(act_probability, 4)}
        if q["t"] == "choice":
            keys = list(q["crit"].keys())
            return {
                "type": "choice",
                "choice": keys[int(p.argmax())],
                "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                "confidence": round(self._confidence_from_probs(p, k), 4),
                "action": ext,
            }
        if q["t"] == "score":
            return {
                "type": "score",
                "score": round(float((np.arange(k) * p).sum()), 4),
                "legend": {str(i): c for i, c in enumerate(q["crit"])},
                "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                "confidence": round(self._confidence_from_probs(p, k), 4),
                "action": ext,
            }
        pt = float(p[1])
        return {
            "type": "noul",
            "noul": round(pt, 4),
            "confidence": round(max(pt, 1.0 - pt), 4),
            "action": ext,
        }

    def predict(self, case: Case) -> Prediction:
        return self.predict_batch([case])[0]

    def render_options(self, questions: dict[str, Any]) -> dict[str, list[str]]:
        from laya.agent import render_options

        return {qid: render_options(q) for qid, q in self._internal(questions).items()}

    def runtime_state(self) -> dict:
        """Same contract as the native backend: the provider actually in use.

        ONNX Runtime does not move the graph on OOM, so this cannot change mid-run,
        but the speed samples take the device from here rather than from
        `env_fingerprint`, which would report a GPU the graph is not on.
        """
        return {"device": self._session.get_providers()[0], "dtype": "onnxruntime"}

    # ----------------------------------------------------------- reproducibility
    def run_identity(self) -> dict:
        """`OnnxLayaBackend` non ne aveva, ed era il buco che `--model laya-onnx` apriva.

        Senza questo ogni esperimento che costruisce il config chiamando `run_identity()`
        sollevava `AttributeError` sul backend ONNX: `--model` accettava `laya-onnx` e il run
        moriva subito dopo. Stessa classe di buco di `--model` accettato e ignorato.
        """
        return {
            "backend": self.name,
            "compiled": False,
            "fast": False,
            "variant": "fp32",
            "device": self._device_seen,
            # vedi `LayaEngine.run_identity`: `confidence` non e' una grandezza sola e senza
            # dichiararla un OER@0.75 su questa riga non e' confrontabile con uno letto su
            # un backend che ci mette l'entropia normalizzata anche per k=2.
            "confidence_definition": "max(p,1-p) if k==2 else 1-H(p)/log(k)",
        }

    def fingerprint(self) -> dict:
        return {
            "checkpoint": self.checkpoint,
            "cfg": self.cfg,
            "device": self._device_seen,
            "weight_path": str(self._graph),
            "weight_sha256": self.weight_sha256(),
            "laya_config": self._laya_cfg,
        }

    def weight_sha256(self) -> str | None:
        """sha256 del grafo: è l'equivalente di `model.safetensors` per l'export."""
        if getattr(self, "_graph_sha256", None) is None:
            from layaadv import hashing

            try:
                self._graph_sha256 = hashing.sha256_file(self._graph)
            except OSError:
                return None
        return self._graph_sha256

    def close(self) -> None:
        return None


def _render(q):
    from laya.common import render_options as ro

    return ro(q)


# --------------------------------------------------------------- encoder a coppie
OPEN_JEV_MARKERS = ("[STATE]", "[Q]", "[OPT]")


def assemble_pairs(
    state_ids: Sequence[int],
    questions: Sequence[tuple[str, Sequence[int], Sequence[Sequence[int]]]],
) -> dict:
    """Index bookkeeping for open-jev's pair encoder, as a pure function.

    One sequence carries the state and every question; the head scores each
    (question, option) pair. `seg` is what makes that possible: it is the span slot
    of every token, -1 for the state and the markers, ``pairs + question index`` for
    question tokens and the pair index for option tokens.

    Kept free of any tokenizer, session or numpy so it can be tested as arithmetic
    (the graph interface itself needs a 3.4 GB export to exercise).

    ``questions`` is ``(qid, instruction_ids, option_ids_per_option)``.
    """
    total_pairs = sum(len(opts) for _, _, opts in questions)
    ids: list[int] = []
    seg: list[int] = []
    pair_q: list[int] = []
    pair_opt: list[int] = []
    groups: list[list[int]] = []
    for part in (state_ids,):
        ids.extend(part)
        seg.extend([-1] * len(part))
    for qi, (_qid, ins, options) in enumerate(questions):
        ids.extend(ins)
        seg.extend([total_pairs + qi] * len(ins))
        g: list[int] = []
        for opt in options:
            idx = len(pair_opt)
            ids.extend(opt)
            seg.extend([idx] * len(opt))
            pair_q.append(total_pairs + qi)
            pair_opt.append(idx)
            g.append(idx)
        groups.append(g)
    return {
        "input_ids": ids,
        "seg": seg,
        "pair_q": pair_q,
        "pair_opt": pair_opt,
        "groups": groups,
        "n_pairs": total_pairs,
    }


class OnnxPairBackend(LayaTokenBudget):
    """Backend a coppie (DeBERTa/open-jev), firma `seg, pair_q, pair_opt`.

    Non è drop-in sul marker head di laya, ma è il **counterpart architetturale**
    della stessa idea: un encoder che legge stato e tutte le domande in una sequenza e
    una testa che da' un logit per coppia (domanda, opzione). Cambiano l'architettura
    (DeBERTa-v3-large con span pooling invece di ModernBERT con marker) e la calibrazione
    (dichiara ECE 0.022 in-domain contro 0.417 di `laya-typed-decisions`), non la
    semantica: stesse tre primitive, stessa regola di argmax, e le risposte hanno la
    stessa forma, cosi' le metriche sono confrontabili.

    Serve per la domanda che tre checkpoint della stessa famiglia non possono
    rispondere: se l'attacco di E03 funziona anche su un modello calibrato e di
    architettura diversa, il finding è sul canale delle opzioni e non su questa rete.

    La precisione si sceglie **scegliendo il grafo**: l'export ne porta quattro dello
    stesso modello (fp32, fp16, q4, q4f16), quindi `quantize` non è accettato e al suo
    posto c'è `variant`.
    """

    name = "deberta-onnx"
    VARIANTS = {
        "fp32": "model.onnx",
        "fp16": "model_fp16.onnx",
        "q4": "model_q4.onnx",
        "q4f16": "model_q4f16.onnx",
    }

    def __init__(
        self,
        model_dir: str,
        device: str | None = None,
        batch_size: int = 16,
        variant: str = "fp32",
        quantize: str = "none",
        **_,
    ) -> None:
        import onnxruntime as ort  # lazy

        if quantize not in (None, "", "none"):
            raise ValueError(
                "per deberta-onnx la precisione si sceglie con --variant "
                f"({', '.join(self.VARIANTS)}): i grafi quantizzati sono gia' nell'export. "
                "La quantizzazione in torch vale solo per la famiglia laya (--quantize)."
            )
        if variant not in self.VARIANTS:
            raise ValueError(f"variant sconosciuta {variant!r}; noti: {list(self.VARIANTS)}")

        available = set(ort.get_available_providers())
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self._graph = _resolve_graph(model_dir, self.VARIANTS[variant])
        self._session = ort.InferenceSession(str(self._graph), providers=providers)

        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(model_dir)
        self._special = {
            m: self._tok(m, add_special_tokens=False)["input_ids"][0] for m in ("[CLS]", "[SEP]")
        }

        cfg = _read_json(Path(model_dir) / "open_jev_config.json")
        self._cfg = cfg
        self._temperature = float(cfg.get("temperature", 1.0))
        self._max_len = int(cfg.get("max_len", 512))
        self._max_state_tokens = int(cfg.get("max_state_tokens", 256))

        self.checkpoint = model_dir
        self.batch_size = batch_size
        self.compiled = False
        self.name = "deberta-onnx"
        self.variant = variant
        self.quantization = {"quantization": variant, "note": "precisione nativa del grafo"}
        self.load_warnings: list[str] = []
        self._device_seen = "onnx:" + self._session.get_providers()[0]
        self._warn_if_symlinked_data(self._graph)

    def _warn_if_symlinked_data(self, graph: Path) -> None:
        """ONNX Runtime rifiuta i dati esterni che risolvono fuori dalla directory del modello.

        Il caso reale: gli snapshot HF sono symlink nei blob e l'export mette i pesi in
        `model.onnx_data`, quindi l'errore arriva dieci righe dentro l'esperimento. Qui si
        dice prima, con il rimedio.
        """
        if not graph.with_suffix(graph.suffix + "_data").exists():
            return
        if graph.parent.resolve() != graph.parent:
            self.load_warnings.append(
                f"deberta-onnx: {graph.parent} contiene symlink e i dati esterni non si "
                "caricano (External data path escapes model directory). Copia lo snapshot "
                "in una directory vera (cp -rL) prima di girare."
            )

    # ------------------------------------------------------------- costruzione
    def _internal(self, questions: dict[str, Any]) -> dict[str, dict]:
        """Same normalized form as laya, via the same functions.

        The text the model reads has to be the text laya would have read, so any
        difference between the two backends is architecture and not prompt.
        """
        from laya.agent import Agent

        return {qid: Agent._to_internal(q) for qid, q in questions.items()}

    def _plan(self, state: str, questions: dict[str, Any]) -> dict:
        """Tokenize one state and every question into a single sequence, plus the pair map."""
        from laya.agent import render_options

        spec: list[tuple[str, list[int], list[list[int]]]] = []
        labels: dict[str, dict[str, list[str]]] = {}
        for qid, q in self._internal(questions).items():
            texts = render_options(q)
            labels[qid] = _labels_of(q, texts)
            spec.append(
                (
                    qid,
                    self._tok(f"[Q] {q['ins']}", add_special_tokens=False)["input_ids"],
                    [self._tok(f"[OPT] {t}", add_special_tokens=False)["input_ids"] for t in texts],
                )
            )
        state_ids = self._tok(f"[STATE] {state}", add_special_tokens=False)["input_ids"][1:]
        plan = assemble_pairs(state_ids, spec)
        plan["input_ids"] = [self._special["[CLS]"]] + plan["input_ids"] + [self._special["[SEP]"]]
        plan["seg"] = [-1] + plan["seg"] + [-1]
        self._trim_state(plan)
        plan["labels"] = labels
        return plan

    def _trim_state(self, plan: dict) -> None:
        """The state is the only elastic part: the questions are fixed.

        Same side as laya, which keeps the **head** of the state (`truncate_left=False`),
        so a tail injection dies in both backends instead of surviving in one. Which makes
        the budget the binding constraint of this model too: 512 total, 256 state, so a
        large K pushes the options out of the sequence and the answer stops being about
        anything.
        """
        seg = plan["seg"]
        end = 1
        while end < len(seg) and seg[end] == -1:
            end += 1
        state_len = min(end - 1, self._max_state_tokens)
        room = self._max_len - (len(plan["input_ids"]) - state_len)
        if room < state_len:
            keep = max(0, room)
            plan["input_ids"] = plan["input_ids"][: 1 + keep] + plan["input_ids"][end:]
            plan["seg"] = plan["seg"][: 1 + keep] + plan["seg"][end:]

    def _lens(self) -> tuple[int, int]:
        """The head budget here is not a declared `head_max_len`: there is no separate head.

        The sequence is `[CLS] state [Q] [OPT] ... [SEP]` and the state is the elastic part,
        so the questions plus every option get what is left of `max_len` once the state has
        taken its `max_state_tokens`. That leftover is the head budget, and it is what
        decides whether the last options survive.
        """
        return self._max_len, max(0, self._max_len - self._max_state_tokens)

    def head_text(self, questions: dict[str, Any]) -> dict[str, dict]:
        """The built head per question, measured with the **state at full length**.

        `check_head_budget` asks whether the option texts survive. For laya the head is
        built on its own, so option text gets clipped as soon as K grows. Here the sequence
        is shared with the state, so a head that only fits with an empty state would still
        lose its last options on a real message. Measuring with a full state is the
        conservative form, and it makes one guard behave the same on both backends.
        """
        qid = next(iter(questions))
        filler = " ".join(["parola"] * self._max_state_tokens)
        plan = self._plan(filler, {qid: questions[qid]})
        return {
            qid: {
                "head_tokens": len(plan["input_ids"]),
                "markers": plan["n_pairs"],
                "text": self._tok.decode(plan["input_ids"], skip_special_tokens=False),
            }
        }

    # ------------------------------------------------------------------ predict
    def predict_batch(self, cases: list[Case], batch_size: int | None = None) -> list[Prediction]:
        """One row per state, with all of its questions in the same sequence."""
        import numpy as np

        if not cases:
            return []
        size = batch_size or self.batch_size
        out: list[Prediction | None] = [None] * len(cases)
        for members in group_cases(cases):
            for start in range(0, len(members), size):
                part = members[start : start + size]
                plans = [self._plan(case.state, case.questions) for _i, case in part]
                maxlen = max(len(p["input_ids"]) for p in plans)
                feed = {
                    "input_ids": _pad1d(np, [p["input_ids"] for p in plans], maxlen),
                    "attention_mask": np.ones((len(plans), maxlen), dtype=np.int64),
                    "seg": _pad1d(np, [p["seg"] for p in plans], maxlen),
                    "pair_q": np.array([p["pair_q"] for p in plans], dtype=np.int64),
                    "pair_opt": np.array([p["pair_opt"] for p in plans], dtype=np.int64),
                }
                logits = self._session.run(["logits"], feed)[0]
                for j, (i, case) in enumerate(part):
                    out[i] = self._decode(case, plans[j], logits[j])
        missing = [cases[i].id for i, p in enumerate(out) if p is None]
        if missing:
            raise RuntimeError(f"missing predictions for: {missing}")
        return [p for p in out if p is not None]

    def _decode(self, case: Case, plan: dict, row) -> Prediction:
        """Logits -> the same answer shape laya returns, so the metrics are comparable.

        The comparable label comes from the index map built while the options were
        emitted, never by splitting the rendered text: a label containing a colon would
        be cut in half, which is exactly the bug that made the first ONNX laya backend
        score 0 on E13's `prefix` arm.
        """
        import numpy as np

        answers: dict[str, dict] = {}
        for group, qid in zip(plan["groups"], plan["labels"]):
            z = np.array([row[p] for p in group], dtype=np.float32) / self._temperature
            e = np.exp(z - z.max())
            p = e / e.sum()
            lab = plan["labels"][qid]
            qtype = str(case.questions[qid].get("type", "choice"))
            if qtype == "score":
                answers[qid] = {
                    "type": "score",
                    "score": round(float((np.arange(len(p)) * p).sum()), 4),
                    "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                    "confidence": round(_entropy_conf(p), 4),
                }
            elif qtype == "noul":
                # laya's noul is P(true) over the two rendered options, and the pair model
                # is documented with the options in (false, true) order.
                answers[qid] = {
                    "type": "noul",
                    "noul": round(float(p[1]), 4),
                    "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4),
                }
            else:
                answers[qid] = {
                    "type": "choice",
                    "choice": lab["order"][int(p.argmax())],
                    "probabilities": {n: round(float(v), 4) for n, v in zip(lab["order"], p)},
                    "confidence": round(_entropy_conf(p), 4),
                }
        return Prediction(
            id=case.id,
            answers=answers,
            labels={qid: answer_label(a) for qid, a in answers.items()},
            usage={"input_tokens": len(plan["input_ids"]), "output_tokens": 0},
        )

    def predict(self, case: Case) -> Prediction:
        return self.predict_batch([case])[0]

    def render_options(self, questions: dict[str, Any]) -> dict[str, list[str]]:
        from laya.agent import render_options

        return {qid: render_options(q) for qid, q in self._internal(questions).items()}

    def runtime_state(self) -> dict:
        """Same contract as the native backend: the provider actually in use, plus the
        graph's own precision. `env_fingerprint` would report a GPU the graph is not on."""
        return {"device": self._session.get_providers()[0], "dtype": self.variant}

    def run_identity(self) -> dict:
        """The pair backend's precision is the graph, so `variant` is the identity part a
        run config carries, and `quantize` is refused: it would be a name with no second
        precision underneath it."""
        return {
            "backend": self.name,
            "compiled": False,
            "fast": False,
            "variant": self.variant,
            # Su CUDA la fp16 è nativa e la q4 gira sui tensor core: la stessa tabella su CPU
            # dice il contrario (E15: fp16 1.8x piu' lenta di fp32 perche' non ci sono kernel
            # fp16 nativi). Device è quindi una condizione sperimentale, non un dettaglio, e
            # senza questo nel run_id un run GPU riprenderebbe i record di uno CPU.
            "device": self._device_seen,
            # vedi `LayaEngine.run_identity`: dichiarare cosa mette in `confidence`, perche'
            # questa coppia dichiara anche `temperature 1.05` e quindi non e' confrontabile
            # con laya su nessuna metrica di confidenza senza dire quale confidenza.
            "confidence_definition": "max(p,1-p) if k==2 else 1-H(p)/log(k)",
        }

    # ----------------------------------------------------------- reproducibility
    def fingerprint(self) -> dict:
        return {
            "checkpoint": self.checkpoint,
            "variant": self.variant,
            "device": self._device_seen,
            "graph": str(self._graph),
            "graph_sha256": self.weight_sha256(),
            "open_jev_config": self._cfg,
        }

    def weight_sha256(self) -> str | None:
        """sha256 del grafo: the equivalent of `model.safetensors` for an export."""
        if getattr(self, "_graph_sha256", None) is None:
            from layaadv import hashing

            self._graph_sha256 = hashing.sha256_file(self._graph)
        return self._graph_sha256

    def count_tokens(self, text: str) -> int:
        return len(self._tok(text, add_special_tokens=False)["input_ids"])

    def close(self) -> None:
        return None


def _labels_of(q: dict, texts: Sequence[str]) -> dict[str, list[str]]:
    """Comparable labels for the rendered options, kept in emission order.

    `choice` takes the label itself, `score` the level index as a string (which is what
    laya's `probabilities` keys are), `noul` the (false, true) order laya renders. The map
    is built *here*, from the question, so nothing downstream ever reads a label back out
    of a string.
    """
    if q["t"] == "choice":
        return {"order": list(q["crit"].keys()), "texts": list(texts)}
    if q["t"] == "score":
        return {"order": [str(i) for i in range(len(q["crit"]))], "texts": list(texts)}
    return {"order": ["false", "true"], "texts": list(texts)}


def _entropy_conf(p) -> float:
    """The confidence laya reports: 1 - H(p)/log(k).

    Comparing an OER across two backends only means something if the confidence means
    the same thing, and the pair model declares its own calibration (temperature 1.05).
    """
    import math

    import numpy as np

    k = len(p)
    if k < 2:
        return 1.0
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
    return float(max(0.0, min(1.0, 1.0 + (p * np.log(p)).sum() / math.log(k))))


def _pad1d(np, rows: list[list[int]], width: int):
    """Right-pad into a 2-D int64 matrix: the graph's `batch` axis is fixed, the sequence isn't."""
    out = np.zeros((len(rows), width), dtype=np.int64)
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out
