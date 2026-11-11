---
tags: [paper, interpretabilita, bertology, 2020]
---

# A Primer in BERTology

**Rogers, Kovaleva & Rumshisky** · TACL 2020 · arXiv:2002.12327 · `paper/rogers-kovaleva-bertology-primer.pdf`

## Cosa contiene

La sintesi di **oltre 150 studi** su come funziona BERT: cosa impara, come lo
rappresenta, quali modifiche all'obiettivo di pre-training e all'architettura cambiano il
risultato, il problema dell'overparametrizzazione e della compressione. E' il paper da
leggere per tarare le aspettative prima di progettare un'analisi.

## Le conclusioni che ci riguardano

- **Euristiche superficiali**: i modelli si appoggiano a scoring superficiali su NLI,
  reading comprehension e classificazione. Da noi il sospetto equivalente e'
  [[E13-label-semantics]] (il testo delle opzioni pesa 27-29pp) e
  [[option-marker-injection]], che sono euristiche superficiali documentate.
- **Calibrazione e misure aggregate**: vanno ricalcolate per modello e dominio. E' la
  lezione che il progetto ha pagato con gli ECE dichiarati invece che misurati in
  [[E15-quantizzazione]].
- **Compressione/quantizzazione**: cambia i numeri, raramente la natura del risultato.
  Coerente con la nostra campagna su `Q5_K_M` e con E15 su laya.
- **Visualizzazione qualitativa**: tipicamente limitata ad analisi qualitative, e **non va
  interpretata come evidenza definitiva**. Questa frase e' la ragione per cui
  [[attribuzioni-e-sanity-check]] esiste.

## Cosa non trasferisce

La sintesi riguarda BERT e i suoi discendenti diretti con teste semplici. Laya e'
`ModernBERT-large` fully fine-tuned con una decision head che legge solo i `[MASK]`
([[architettura-laya]]): molte conclusioni sull'**uso** delle rappresentazioni vanno
rimisurate, perche' il punto di lettura e' diverso e l'addestramento non e' un
fine-tuning leggero.

## Note collegate
[[bertology-encoder-mascherati]] · [[cosa-guarda-bert]] · [[E13-label-semantics]]
