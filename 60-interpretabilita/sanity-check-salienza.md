---
tags: [paper, interpretabilita, attribuzioni, 2018]
---

# Sanity Checks for Saliency Maps

**Adebayo, Gilmer, Muelly, I. Goodfellow, Hardt & Kim** · NeurIPS 2018 · arXiv:1810.03292 · `paper/adebayo-sanity-checks-saliency.pdf`

## Cosa propone

Due test, entrambi facili da implementare, per capire se un metodo di spiegazione e'
utilizzabile:

1. **Model parameter randomization test**: la stessa mappa prodotta sul modello
   addestrato e su una rete con pesi **casuali** deve essere diversa. Se e' uguale, la
   mappa non dipende dal modello.
2. **Data randomization test**: addestrare la stessa architettura su etichette
   **permutate** e ripetere. Se la mappa non cambia, non dipende dai dati.

Il risultato e' che metodi molto popolari sono indipendenti da modello e dati, e le loro
mappe diventano indistinguibili da quelle di un **rilevatore di bordi**: uno strumento che
non ha bisogno ne' di addestramento ne' di modello, e che pero' produce output convincenti.
Fra i metodi testati, gradienti e GradCAM passano i test; Guided BackProp e le sue varianti
no.

## Perche' ci riguarda

E' il paper che giustifica il programma di controlli di [[E23-meccanismo-injection]]: uno
strumento di spiegazione va testato **su se stesso** prima di essere usato. Nel progetto
la stessa disciplina esiste gia' in altre forme (`tools/noise_floor.py` isola il rumore
prima di interpretare un delta; `tools/oer_audit.py` misura il raggio d'azione di una
metrica prima di citarla).

La versione pratica del test, per un modello che non possiamo riaddestrare facilmente come
Laya, e' il confronto con l'**ablazione**: se il token che l'attribuzione dichiara
importante non e' quello la cui rimozione cambia davvero la decisione, uno dei due
strumenti misura il testo e non la rete.

## Note collegate
[[attribuzioni-e-sanity-check]] · [[interpretabilita-transformer]]
