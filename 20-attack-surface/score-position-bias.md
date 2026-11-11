---
tags: [attacco, bug-noto, score, issue-131]
aliases: [issue #131, score bias]
---

# `score` position bias (issue #131)

Bug documentato: `laya-multilingual` **raramente sceglie il primo livello** elencato, in qualsiasi lingua.

## Sintomo
Su domande `score` del multilingual, il livello 0 (prima opzione della rubrica) è quasi mai selected. Qualsiasi stato la cui gold è "il livello più basso" viene sovrastimato.

## Perché è un attacco
- `score` è già la **primitive più debole** (SST-5: 0.282 multilingual, 0.372 English)
- Se il primo livello è strutturalmente penalizzato, **tutta la distribuzione slitta verso l'alto**
- Dataset il cui gold è skewato verso 0 → drop sistematico misurabile

## Strategie
1. **Dataset-level**: scegliere/creare uno split con molti gold al livello 0 (es. "nessuna urgenza", "nessun rischio")
2. **Stato-level**: scrivere stati che *dovrebbero* ricevere livello 0 e vedere se il modello lo assegna comunque
3. **Rubrica-order**: riordinare i `criteria` (se lecito: vedi ) spostando il livello vero in prima posizione → misurare il drop
4. Cross-checkpoint: il bug è dichiarato sul multilingual, vale anche su `laya` e `laya-typed-decisions`?

## Raccomandazione ufficiale di Convai
Per `score` in inglese usare `model="english"`; per le altre lingue **validare gli output su dati propri** prima di fidarsi. Il bias è dichiarato solo sul multilingual, ma la raccomandazione generale di validare `score` vale per tutti.

## Da misurare
- [ ] Distribuzione predetta vs distribuzione gold sul primo livello
- [ ] % di predizioni con argmax = 0 (atteso ≈ 0?)
- [ ] Effetto del riordino dei criteria
- [ ] MAE sul livello, per checkpoint
- [ ] Intera curva acc vs livello gold (0..K)

## Note collegate
[[primitive-typed]] · [[score-position-bias]] · [[rlcd-calibrazione]] · [[esperimenti]]
