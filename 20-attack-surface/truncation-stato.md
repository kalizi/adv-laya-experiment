---
tags: [attacco, troncamento, budget]
---

# Troncamento dello stato

## Meccanismo
Budget per lo stato = `max_len − head_max_len`:

| checkpoint | stato budget (circa) |
|---|---|
| `laya` (English) | 512 − 192 = **~320 token** |
| `laya-multilingual` / `laya-typed-decisions` | 1024 − 256 = **~768 token** |

Il troncamento avviene di default sullo **stato**. Se la parte decisiva sta in coda, viene tagliata.

## Varianti

### A. Push-out (niente rumore)
Mantenere il gold label intatto ma **spingere la frase chiave oltre il budget** con filler semanticamente neutro (legalese, firme, citazioni, thread di email precedenti). Il modello vede uno stato innocuo → predice la classe di default.
- Necessita: il deployment deve effettivamente troncare (verificare con un marker in coda in [[E00-setup]])

### B. Signal dilution (stato nel budget)
Restare nel budget ma **diluire** il segnale con materiale irrilevante → non si perde la frase, ma il pooled representation e i `[MASK]` perdono grip. Misurabile come curva acc vs lunghezza.

### C. Early-position attack
L'attenzione è alternata: **un livello globale ogni 3** (10 su 28 = 0.357) e **18 livelli
locali**, e un livello locale raggiunge **±64 token** (`local_attention=128` è una mezza
finestra: maschera `|query − key| ≤ 64`, misurata sui pesi in
`results/provenance/e22_attention_structure.json`). Quindi il confine utile è **ogni 64
token**, non ogni 128, e un marcatore oltre 64 token dal segnale non lo vede nei livelli
locali. Da testare.

## Da misurare
- [ ] Marker in coda: appare nella tokenizzazione effettiva? (troncamento on/off)
- [ ] Curva acc vs lunghezza stato: 0.25×, 0.5×, 0.75×, 1.0×, 1.5× budget
- [ ] Push-out: drop e confidence
- [ ] Confronto posizione: segnale a token 50 vs 300 vs 700
- [ ] Interazione con [[token-budget-head]] (K alto + stato lungo)

## Note collegate
[[architettura-laya]] · [[token-budget-head]] · [[esperimenti]] · [[metriche-attacco]] · [[E22-attenzione-laya]]
