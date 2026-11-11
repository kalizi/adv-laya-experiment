---
tags: [strumenti, backend, modelli, infrastructure]
stato: fatto
data: 2026-09-30
---

# Backend hot-swappable

Il modello è un **parametro**, non una convenzione. `--model` seleziona il backend;
`open_backend(spec, checkpoint)` è l'unico punto in cui un esperimento apre un modello,
e resta dentro `model/engine.py` per mantenere l'isolamento degli import pesanti.

## Backend disponibili

| spec | cosa | dove | stato |
|---|---|---|---|
| `laya` (default) | backend nativo, GPU | `convaiinnovations/laya-*` | verificato, è quello di tutti gli esperimenti |
| `laya-onnx` | stessa testa a marker via ONNX Runtime, CPU | export laya-multilingual | **verificato: 40/40 label identiche al PyTorch**, vedi sotto |
| `deberta-onnx` | open-jev: coppie (domanda, opzione), span pooling | `data/onnx/open-jev-deberta-v3-large` | **implementato**, vedi sotto |

## Cosa significa "identico" (misurato, non supposto)

Su `laya-multilingual`, 8 casi di typed-decisions con 5 domande ciascuno (40 decisioni),
nativo in autocast bf16 contro grafo in fp32:

| grandezza | risultato |
|---|---|
| label per decisione | **40/40 identiche** |
| `accuracy` | identica |
| `OER@0.75` | identica |
| `confidence` | entro 0.01 su 37/40, max \|Δ\| 0.019 |
| `probs` | coincidono a ~1e-3 |
| `input_tokens` | identico |

La residua è **numerica** (fp32 contro bf16), non logica: label, accuracy e OER
coincidono, quindi due run sui due backend sono confrontabili. Un delta di OER sotto
~0.02 non lo è.

> Corretto il 2026-10-01. Il backend rispondeva alla **sola prima domanda** del caso e
> non emetteva `confidence`: su typed-decisions avrebbe riportato 8 decisioni invece di
> 40 e `OER`/`ECE`/`confidenza sul torto` **zero per costruzione**, cioe' "il guardrail non
> si rompe" senza aver misurato nulla. Le temperature e i budget (`max_len`,
> `head_max_len`) sono ora letti da `laya_config.json` dell'export invece che essere
> indovinati: sono diversi per checkpoint (`laya` dichiara 512/192,
> `laya-typed-decisions` 1024/256) e un valore fisso troncherebbe E04 sul modello
> sbagliato. `--model` non era collegato a nessun backend: il flag e l'help promettevano
> una cosa che nessun esperimento eseguiva.

## Uso
```bash
uv run adv e03 --model laya-onnx --checkpoint <dir-export> --k 16 --limit 1000
```
`--checkpoint` resta il path HF o locale; per `laya-onnx` è la directory dell'export
ONNX (il grafo può stare nella radice o in `onnx/`, il tokenizer alla radice).

## Modelli scaricati in locale
- `killkli/open-jev-laya-multilingual-onnx` — Laya multilingual in ONNX (1.84 GiB)
- `onnx-community/open-jev-deberta-v3-large-ONNX` — DeBERTa-v3-large, encoder a coppie (3.22 GiB)
- `argos1111/modernbert-ja-310m-jev` — ModernBERT 310M, **una sola classe** (`LABEL_0`): è un
  sequence classifier, non un modello a opzioni, quindi non è un backend drop-in. Download
  mantenuto per riferimento, non utilizzato.

## Il backend a coppie, e perché è il counterpart e non un encoder qualsiasi

`onnx-community/open-jev-deberta-v3-large-ONNX` non è un encoder generico: è un
**counterpart architetturale** di laya. Un DeBERTa-v3-large legge stato e tutte le domande
in una sequenza e la testa dà un logit per coppia (domanda, opzione) con span pooling.
Stesse tre primitive, stessa regola di argmax, e le risposte hanno la stessa forma di quelle
di laya, così le metriche sono confrontabili. Cambiano encoder, testa e **calibrazione**:
dichiara in carta ECE 0.022 in-domain, contro 0.417 di `laya-typed-decisions`; **misurati da noi su `banking77` sono pero' 0.300 e 0.025**, cioe' l'ordine inverso ([[E15-quantizzazione]]).

La firma del grafo è `input_ids, attention_mask, seg, pair_q, pair_opt` → `logits`. `seg` è
lo slot di ogni token: `-1` su stato e marker, `pairs + indice domanda` sui token della
domanda, l'indice della coppia su quelli dell'opzione. L'uguaglianza è in
`backends.assemble_pairs`, che è aritmetica pura e quindi testata senza grafo.

Tre cose da sapere prima di usarlo:

1. **L'export va dereferenziato una volta.** Gli snapshot HF sono symlink nei blob e
   ONNX Runtime rifiuta i dati esterni che risolvono fuori dalla directory del modello
   ("External data path escapes model directory"). `cp -rL` in `data/onnx/` una volta sola.
   Il backend avvisa da solo se trova la situazione.
2. **La precisione è il grafo, non `--quantize`.** L'export porta quattro varianti dello
   stesso modello: `fp32`, `fp16`, `q4`, `q4f16`. `--quantize` su questo backend solleva,
   perché accettarlo produrrebbe un "int4" che è un grafo fp32 con un nome falso.
3. **Il dataset è `banking77` con `label_text`.** Con l'id intero le opzioni sono "63",
   "13" e il modello, addestrato sui nomi degli intent, non ha niente su cui decidere.
4. **Serve `onnxruntime-gpu`, non `onnxruntime`.** Il pacchetto CPU funziona, ma la tabella
   di E15 misurata su CPU dice il contrario della tabella su GPU (la fp16 costa 1.8x la fp32
   senza kernel nativi, e su GPU è 1.3x più veloce). I due pacchetti non convivono nella
   stessa venv.
5. **Il CUDA EP va verificato per nodo, non per la lista dei provider.**
   `get_providers()` restituisce l'ordine di priorità: una sessione che ricade sulla CPU per
   un operatore continua a dichiarare `CUDAExecutionProvider` in testa, e il device salvato
   nelle run sarebbe falso.
   ```bash
   uv run python tools/onnx_provider_audit.py
   ```
   Per questa export tutte le moltiplicazioni, `MatMulNBits` comprese, girano su CUDA.

## Note
- `requires-python` alzato a `>=3.11` perché `onnxruntime-gpu>=1.30` richiede 3.11+.
- Il modulo `backends.py` mantiene gli import pesanti lazy: importarlo non tira dentro
  torch né onnxruntime (test in subprocess).
- La firma del backend a coppie (`seg, pair_q, pair_opt`, un logit per domanda-opzione) è
  diversa dalla testa a marker di Laya: serve il proprio wrapper per assemblare le coppie,
  ed è il backend che manca per testare la generalizzazione architetturale della legge dei
  token fuori dalla famiglia Laya.

## Note collegate
[[E01-baseline]] · [[E03-option-injection]] · [[configurazione-ambiente]] · [[esperimenti]]