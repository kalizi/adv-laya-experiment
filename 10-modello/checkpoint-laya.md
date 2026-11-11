---
tags: [modello, checkpoint, router]
aliases: [Router]
---

# Checkpoint e Router

## I tre checkpoint
| id | encoder | params | context | per cosa |
|---|---|---|---|---|
| `convaiinnovations/laya` | ModernBERT-large | 421M | 512 | English, guardrails, email triage |
| `convaiinnovations/laya-multilingual` | mmBERT-base | 322M | 1024 (max 8192) | 100+ lingue, ~2.2× più veloce |
| `convaiinnovations/laya-typed-decisions` | ModernBERT-large | 421M | 1024 | le 4 workflow del benchmark |

```python
from laya import Router
router = Router()
router.predict(state, questions, model="typed-decisions")
# oppure
agent = laya.load("convaiinnovations/laya-typed-decisions")
```

> `Router()` NON seleziona `typed-decisions` automaticamente (a meno di `auto_task_detection=True`): è uno specialista su 4 workflow sintetici.

## Come sceglie il Router
1. Rilevamento script + function-word heuristic, **< 0.5 ms in Python puro, prima della forward pass**
2. Dispatch: script non-Latin → multilingual; Latin → English (best-effort anche per spagnolo/francese/italiano senza accenti)
3. Hint opzionale: `lang_guess=`, `Router(default=...)`; codice `en`/`eng`/`english` → English. `None` o vuoto → abstain
4. `model=`, `task=`, `lang=` espliciti vincono sempre

`Router()` tiene **2 checkpoint residenti** di default (niente reload a ogni alternanza di lingua). `max_loaded=1` → rebuild ~4.4 s median.

## Perché il Router esiste
Il confidence del modello **non segnala** che il checkpoint è illeggibile per quella lingua:
- English checkpoint, 51 lingue MASSIVE: macro acc **0.227**, macro ECE **0.733**, solo 23/51 sopra 3× random
- Khmer **0.000 @ 0.952 conf**, Hebrew 0.060, Armenian 0.050, Bengali 0.080 — tutti a 0.89–0.96 confidence
- La mean confidence **non scende mai sotto 0.885** ad alcun livello di accuracy

→ **Il routing va fatto prima del forward pass.** Vedi [[routing-script-bypass]].

## Multilingual vs English (numeri)
| | `laya` | `laya-multilingual` |
|---|---|---|
| macro acc (51 lingue) | 0.227 | **0.366** |
| macro ECE | 0.733 | **0.387** |
| >3× random | 23/51 | **45/51** |
| XNLI English | **0.860** | 0.843 |
| XNLI 14 altre | 0.521 | **0.731** |
| English suites macro | **0.684** | 0.619 |

**Route, non replace.**

## Note collegate
[[routing-script-bypass]] · [[laya]] · [[checkpoint-laya]] · [[calibrazione-fratturata]]
