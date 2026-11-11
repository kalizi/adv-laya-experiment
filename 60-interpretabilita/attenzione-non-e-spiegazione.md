---
tags: [paper, interpretabilita, attenzione, 2019]
---

# Attention is not Explanation

**Jain & Wallace** · NAACL-HLT 2019 · arXiv:1902.10186 · `paper/jain-wallace-attention-not-explanation.pdf`

## Cosa misura

Se l'attenzione e' una spiegazione, dovrebbero valere due cose: i pesi dovrebbero
**correlare** con le misure gradienti di importanza delle feature, e distribuzioni di
attenzione **diverse** dovrebbero dare previsioni diverse. Gli autori testano entrambe su
8 dataset di classificazione e 2 di question answering, e nessuna delle due regge.

Il secondo test e' il piu' forte: si costruiscono distribuzioni di attenzione alternative
che danno **previsioni equivalenti** a quelle originali. Se due spiegazioni diverse
producono la stessa uscita, l'attenzione non e' *la* spiegazione.

## I numeri che contano

- La correlazione con le misure gradienti e' debole e instabile fra task.
- Le attenzioni alternative (fra cui attenzione uniforme) mantengono le prestazioni.
- Le mappe di attenzione non sono quindi un attributo del modello, sono una fra le
  possibili letture.

## Perche' ci riguarda

E22 ha misurato quanta' attenzione arriva sullo span dell'iniezione, e ha trovato che non
spiega l'ASR. Letto con questo paper, quel risultato e' **atteso**: la massa di attenzione
e' una variabile debole per costruzione. Il valore di E22 non e' la massa, sono le misure
sui pesi (struttura globale/locale, ±64 token) e il fatto che l'attacco regge anche quando
i livelli locali non vedono piu' nulla.

## Cosa NON dice

Non dice che l'attenzione sia inutile, ne' che i pesi non contengono informazione. Dice
che **non sono una spiegazione della predizione**. La distinzione torna nel titolo della
risposta: [[attenzione-non-non-e-spiegazione]].

## Note collegate
[[attenzione-come-spiegazione]] · [[attribuzioni-e-sanity-check]]
