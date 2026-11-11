---
tags: [paper, interpretabilita, attribuzioni, 2021]
---

# Transformer Interpretability Beyond Attention Visualization

**Chefer, Gur & Wolf** · CVPR 2021 · arXiv:2012.09838 · `paper/chefer-transformer-interpretability.pdf`

## Cosa propone

Un modo di calcolare la **rilevanza** in un transformer, invece di leggere le mappe di
attenzione: assegna una rilevanza locale con Deep Taylor Decomposition e la propaga
attraverso i livelli, attraverso l'attenzione **e** le connessioni residue, con una
formulazione che **conserva la somma totale** della rilevanza. La conservazione e' una
verifica interna: se la somma non si conserva, la propagazione e' sbagliata.

Due proprieta' che contano piu' del nome:

- **Class-specific**: la stessa immagine (o frase) ha spiegazioni diverse per classi
  diverse. Molti metodi su transformer non lo sono, e restituiscono la stessa mappa per
  qualsiasi output.
- Funziona sia su testo sia su immagini, con codice pubblico.

## Perche' ci riguarda

E' il candidato naturale per la domanda che il progetto si pone: non "qual e' il token
importante", ma "qual e' il token importante **per l'opzione scelta invece di quella
giusta**". La proprieta' class-specific e' esattamente quella che serve quando un modello
produce piu' decisioni per forward, come Laya con le sue K opzioni.

La cosa da non fare: usarlo **senza** i controlli di [[sanity-check-salienza]]. Una
rilevanza conservata non e' una rilevanza fedele, e il confronto con l'ablazione resta il
giudice.

## Note collegate
[[attribuzioni-e-sanity-check]] · [[circuiti-e-interventi]]
