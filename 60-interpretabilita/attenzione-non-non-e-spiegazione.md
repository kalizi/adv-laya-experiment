---
tags: [paper, interpretabilita, attenzione, 2019]
---

# Attention is not not Explanation

**Wiegreffe & Pinter** · EMNLP-IJCNLP 2019 · arXiv:1908.04626 · `paper/wiegreffe-pinter-attention-not-not-explanation.pdf`

## Cosa dice

La risposta a [[attenzione-non-e-spiegazione]], e il doppio negativo e' deliberato: la
conclusione dipende da **cosa** si intende per spiegazione, e il test va fatto su tutti
gli elementi del modello con un disegno rigoroso. L'argomento centrale e' che Jain &
Wallace non hanno **addestrato** un modello con le attenzioni avversarie: quelle
attenzioni sono quindi spiegazioni di un modello che non esiste, non spiegazioni
alternative dello stesso modello. "Existenza non implica esclusivita'", e l'attenzione
fornisce *una* spiegazione, non *la* spiegazione.

## I quattro test proposti

1. **Baseline ad attenzione uniforme**: la distribazione uniforme e' una spiegazione che
   non richiede apprendimento, quindi va confrontata per prima.
2. **Calibrazione sulla varianza**: ripetere l'addestramento con piu' seed e guardare
   quanto si muovono le mappe.
3. **Framework diagnostico con pesi congelati**: usare pesi pre-addestrati per isolare
   l'effetto dell'attenzione da quello dell'apprendimento.
4. **Protocollo avversario end-to-end**: addestrare davvero il modello con le attenzioni
   avversarie, invece di sostituirle a mano.

Risultato: anche le attenzioni avversarie affidabili non passano il diagnostico semplice,
quindi il lavoro precedente non ha dimostrato che l'attenzione sia inutile.

## Perche' ci riguarda

E' il motivo per cui E22 fa **due** cose invece di una: misura i profili di attenzione, e
poi misura il **comportamento** (l'ASR sugli stessi casi). Il controllo neutro (frase che
non nomina opzioni) e' l'equivalente del test uniforme: se una frase senza contenuto
ottiene lo stesso profilo, il profilo non distingue la nominazione. Ed e' esattamente
quello che e' successo.

## Note collegate
[[attenzione-come-spiegazione]] · [[E22-attenzione-laya]]
