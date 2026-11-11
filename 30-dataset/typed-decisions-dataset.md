---
tags: [dataset, benchmark, gold]
aliases: [typed-decisions]
---

# typed-decisions (dataset ufficiale)

Il benchmark indipendente su cui si gioca tutto.

- **Hub:** `huggingface.co/datasets/LocalLLaMA/typed-decisions`
- **Struttura:** 4 workflow sintetici
  - agent-trace observability
  - customer service
  - invoice processing
  - security incidents

## Split
| split | casi | decisioni |
|---|---|---|
| train | 1.200 | 6.000 (usato per il fine-tuning di `laya-typed-decisions`) |
| **test (ufficiale)** | **400** | **2.000** (5 domande per caso) |

Per la study: **attaccare il test split 400**. Il train serve solo per capire la distribution, non per addestrare (a meno di non dichiarare un attacco train-informed → [[black-box-vs-gray-box]]).

## Schema (per esempio)
Campi: `id`, `workflow`, `split`, `state`, `questions`, `gold`, `factors`, `n_questions`, `label_agreement`, poi 5 domande piane:

| domanda | tipo | tipo label |
|---|---|---|
| `outcome` | choice | 4 classi |
| `action` | choice | 4 classi |
| `needs_review` | noul | boolean + confidence + probs |
| `risk` | score | 4 livelli + score atteso |
| `urgency` | score | 4 livelli + score atteso |

Presenti anche: `*_confidence`, `*_probabilities`, `*_probability_true`, `*_score`, `label_agreement` (distribuzione del teacher), `outcome__label` ecc.

→ Composizione: **600 noul + 600 choice + 800 score = 2.000 decisioni** sul test.

## Baseline pubblicate (test split)
| modello | acc | soft acc | Brier | ECE | score MAE |
|---|---|---|---|---|---|
| `laya-typed-decisions` | **0.766** | 0.471 | 0.062 | 0.213 | 0.242 |
| Jev 1.13.0 (pubblicato) | 0.727 | 0.580 | 0.148 | 0.144 | 0.391 |
| teacher self-agreement ceiling | 0.735 | — | — | — | — |
| ModernBERT-base specialist | 0.646 | 0.542 | 0.119 | 0.179 | 0.444 |
| majority class | 0.461 | — | — | — | — |
| random | 0.318 | — | — | — | — |
| `laya` (base, zero-shot) | 0.362 | 0.332 | 0.316 | 0.175 | 0.694 |
| `laya-multilingual` (zero-shot) | 0.342 | 0.326 | 0.439 | 0.285 | 0.687 |

### Per workflow (specialist)
invoice 0.804 · security 0.766 · customer service 0.764 · agent-trace 0.730

### Per primitive (specialist)
| tipo | acc | ECE | n |
|---|---|---|---|
| noul | 0.857 | 0.192 | 600 |
| choice | 0.733 | 0.255 | 600 |
| score | 0.723 | 0.199 | 800 |

## Implicazioni per la study
1. **Target principale:** `laya-typed-decisions` (in-distribution, baseline alta 0.766 → il drop è significativo).
2. **Target secondario:** `laya` base (zero-shot 0.362 → baseline bassa, un drop di 20pp è quasi impossibile; qui servono OER e transfer, non il drop).
3. ⚠️ Esiste anche `pngwn/typed-decisions-v2`: corpus companion con label review/escalate **invertite** nella v1 (nanodiff finding 6). Verificare su quale versione sono stati misurati i numeri prima di confrontare → [[dataset-secondari]].
4. Il `label_agreement` del teacher dà una mappa degli esempi ambigui: utilissimo per filtrare il test set (attaccare gli esempi *non* ambigui).

## Download
```python
from datasets import load_dataset
ds = load_dataset("LocalLLaMA/typed-decisions", split="test")
```

## Note collegate
[[protocollo-dataset-valido]] · [[laya]] · [[esperimenti]] · [[metriche-attacco]]
