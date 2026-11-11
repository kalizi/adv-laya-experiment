---
tags: [attacco, calibrazione, confidence]
---

# Calibrazione fratturata

L'over-confidence **è una superficie d'attacco**, non solo un difetto di reporting.

## Fatti
- I checkpoint shippati sono **over-confident**: ECE raw **0.466** (`laya`), **0.314** (`laya-multilingual`)
- Le temperature ereditate o mancanti: `laya-typed-decisions` usa `temperature_by_options` del base, `laya-multilingual` arriva con T = [1.0, 1.0, 1.0]
- Dopo temperature fitting per (type, option count): ECE 0.466 → **0.081**, 0.314 → **0.106**
- Confidence formula di Laya è entropy-based: `1 − H(p)/log(k)`

## Perché è grave per una study
Se un attacco produce predizioni **sbagliate ma confidenti (≥ 0.75)**, qualsiasi sistema che gatea sulla confidence (automation @ coverage) **non interviene**. Esempio dal model card: selective automation @ 50% coverage → 92.2% accuracy. Un attacco overconfident rompe l'assunto.

Casi noti:
- Khmer: 0.000 acc @ 0.952 confidence
- mean confidence **non scende sotto 0.885** per nessun livello di accuracy (English su 51 lingue)

## Attacchi derivati

### 1. Overconfident error rate (OER)
Metrica principale: frazione di esempi con `correct = false AND confidence ≥ τ` (τ = 0.75 o 0.85). Questo è il vero "fischi per fiaschi" della study → .

### 2. Confidence senza rifitting
Usare i pesi come shippati (nessuna temperature fit sul proprio dataset) è lo scenario realistico di deployment. Misurare baseline ECE e OER sul proprio dataset **prima** di attaccare → [[E01-baseline]].

### 3. Distribution sharpening
Se la confidence è entropy-based, attacchi che **spiano la distribuzione** (meno entropia) alzano la confidence anche se l'argmax è sbagliato → legare con [[option-marker-injection]] e [[token-budget-head]].

### 4. Far sembrare "calibrated" un errore sistematico
Un attacco che produce costantemente una classe sbagliata con p moderately peaked distribution può avere ECE basso localmente e accuracy catastrofica: ECE da solo non basta, serve OER + accuracy → [[metriche-attacco]].

## Da misurare
- [ ] ECE e OER puliti, per checkpoint, sul proprio dataset (senza refit)
- [ ] OER dopo attacco (qualsiasi): il gate di confidence salva o no?
- [ ] Confronto OER con e senza temperature fitting post-hoc

## Note collegate
[[rlcd-calibrazione]] · [[metriche-attacco]] ·  · [[esperimenti]]
