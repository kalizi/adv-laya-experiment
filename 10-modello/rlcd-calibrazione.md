---
tags: [modello, training, calibrazione, rlcd]
aliases: [RLCD]
---

# RLCD e calibrazione

## RLCD — Reinforcement Learning for Calibrated Decisions
Metodo di training condiviso da [[jev]] (non pubblicato) e [[laya]] (pubblicato).

Il ciclo di Laya:
1. La policy emette una **distribuzione** (non un'etichetta)
2. L'esplorazione aggiunge **rumore gaussiano a zero-media ai logits**
3. Il reward è uno **strictly proper scoring rule**:
   - log score + spherical score
   - ranked probability score per le domande ordinali `score`
4. Update: **REINFORCE con baseline di gruppo** (stile GRPO)
5. Multi-turn: **TD(λ=1.0)** su prefix slices (niente outcome leakage)
6. + soft cross-entropy contro le distribuzioni del teacher

### Perché conta
Con uno scoring rule strictly proper, **l'unico modo di massimizzare il reward atteso è riportare probabilità oneste**. Quindi le probabilità *dovrebbero* essere calibrate… salvo:
- temperature fitting non rifatto sul dominio → [[calibrazione-fratturata]]
- shift out-of-distribution → il modello resta confidente
- bias strutturali del decoder (label, posizione) → [[noul-label-bias]], [[score-position-bias]]

## Metriche di calibrazione
| metrica | cosa misura | numero Laya |
|---|---|---|
| **ECE** (Expected Calibration Error) | scostamento confidenza vs accuracy reale per bin | raw **0.466** → **0.081** dopo temperature fit |
| **Brier** | MSE tra probabilità predette ed evento | 0.062 (typed-decisions) |
| **soft accuracy** | match della distribuzione intera vs teacher | 0.471 (contro 0.580 di Jev) |
| **score MAE** | errore medio sul livello ordinale | 0.242 |

### Numeri chiave
- I checkpoint **shipped sono over-confident**: temperature `temperature_by_options` ereditate dal base e non ri-fitted per `laya-typed-decisions`.
- `laya-multilingual` arriva **senza temperature fitted** (T = [1.0, 1.0, 1.0]).
- Su tutti e due i checkpoint: rifittare una temperature per (question type, option count) su held-out muove ECE 0.466 → 0.081 (`laya`) e 0.314 → 0.106 (`laya-multilingual`).

### Conseguenza d'attacco
**La confidence del modello non è un guardrail affidabile.** Esempi noti:
- Khmer: 0.000 accuracy a 0.952 confidence
- ECE sul typed-decisions del checkpoint specialist: 0.213 (Jev: 0.144)

→ Un attacco che produce predizioni **sbagliate e confidenti** è più grave di uno che produce predizioni sbagliate e incerte. Vedi  e [[calibrazione-fratturata]].

## Confidence: formula diversa da Jev
- Laya: entropy-based `1 − H(p)/log(k)`
- Jev: peak-based normalization (documentato)

Stessa distribuzione → confidence diverse. **Non copiare le soglie di threshold da Jev.**

## Note collegate
[[laya]] · [[calibrazione-fratturata]] · [[metriche-attacco]] · [[rlcd-calibrazione]]
