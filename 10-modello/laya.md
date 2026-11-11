---
tags: [modello, laya, open-source]
---

# Laya

Il clone Apache-2.0 di [[jev]]. Letteralmente "LAVA" con la Y.

## Scheda
- **Casa:** Convai Innovations (CEO Nandakishor M)
- **Licenza:** Apache 2.0 — pesi e codice pubblici
- **Pacchetto:** `pip install laya` (PyPI `laya`, Python ≥ 3.10 recenti)
- **Repo:** `github.com/NandhaKishorM/laya` · Hub: `convaiinnovations/laya`
- **Latenza:** ~33 ms / domanda singola (T4), 7.2 ms/question batched a 10
- **Throughput:** 103–332 questions/sec batched su una T4
- **Costo:** $0 self-hosted

## Perché è il target perfetto per la study
1. **Pesi scaricabili** → gray-box possibile (gradient, analisi interna)
2. **Wire-compatibile** → stesso schema di Jev, gli esperimenti valgono per entrambi
3. **Apache 2.0** → possiamo pubblicare ogni cosa senza problemi
4. **Ha bug noti** → superficie d'attacco già mappata in parte (vedi [[superficie-attacco]])

## Tre checkpoint
Vedi [[checkpoint-laya]]:

| id | encoder | params | context | use |
|---|---|---|---|---|
| `convaiinnovations/laya` | ModernBERT-large | 421M | 512 | English, generale |
| `convaiinnovations/laya-multilingual` | mmBERT-base | 322M | 1024 (fino a 8k) | 100+ lingue |
| `convaiinnovations/laya-typed-decisions` | ModernBERT-large | 421M | 1024 | le 4 workflow del benchmark |

```python
import laya
agent = laya.load("convaiinnovations/laya-typed-decisions")
result = agent.predict(state, questions)
print(result["answers"])
```

> ⚠️ Se `laya.load()` si blocca: `transformers` prova TensorFlow all'import. Usare `USE_TF=0`.

## Limiti (dichiarati, sono la nostra mappa dei territori)
- **Near chance zero-shot** sui typed-decisions se non fine-tuned: `laya` 0.362, `laya-multilingual` 0.342 (random 0.318, majority 0.461). La baseline 0.766 è del solo checkpoint `laya-typed-decisions`.
- **Over-confidence**: ECE raw 0.466 → 0.081 solo dopo temperature fitting per (type, option count).
- **`noul` segue le label** più dello stato → [[noul-label-bias]]
- **`score` ha bias di posizione** (multilingual) → [[score-position-bias]]
- **Choice >20 options soffre** per il budget → [[token-budget-head]]
- **Fuori dall'inglese il checkpoint English collassa** e resta confidente (Khmer: 0.000 acc @ 0.952 conf)

## Fine-tuning
Notebook Kaggle 2×T4 gratis, ~4–5 ore: dataset → RLCD → temperature → eval → push Hub. Utile anche per capire *come* è stato addestrato → [[rlcd-calibrazione]].

## Note collegate
[[checkpoint-laya]] · [[architettura-laya]] · [[superficie-attacco]] · [[typed-decisions-dataset]] · [[link-utili]]
