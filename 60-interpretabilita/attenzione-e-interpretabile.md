---
tags: [paper, interpretabilita, attenzione, 2019]
---

# Is Attention Interpretable?

**Serrano & Smith** · ACL 2019 · arXiv:1906.03731 · `paper/serrano-smith-is-attention-interpretable.pdf`

## Cosa misura

Non "l'attenzione spiega la predizione" (la domanda di Jain & Wallace) ma la domanda piu'
bassa: l'attenzione identifica quali **input contestualizzati** hanno piu' impatto
sull'uscita? La manipolazione e' diretta: si alterano i pesi in modelli di
classificazione testo gia' addestrati e si guarda cosa cambia.

Risultato: pesi piu' alti tendono ad avere piu' effetto, ma la correlazione e' rumorosa e
fallisce in molti casi. In particolare, l'ordinamento dei pesi per **gradiente** predice
meglio il loro effetto dell'ordinamento per **grandezza**.

## La frase da tenere

L'attenzione predice in modo rumoroso l'importanza complessiva, e non e' un indicatore
fail-safe. Non e' un metodo ottimale per identificare quali elementi ascoltati sono
responsabili di un'uscita.

## Perche' ci riguarda

Ci da' un **ordinamento alternativo da confrontare** con le nostre mappe. Se E23 produce
una mappa di attenzione e una lista di teste rilevanti per knockout, e le due non
coincidono, uno dei due strumenti mente. Il confronto gradiente-vs-grandezza e' la
versione economica dello stesso controllo, e si puo' fare senza alcun intervento
addestrato.

E' anche la ragione per cui [[attribuzioni-e-sanity-check]] insiste sul confronto con
l'ablazione: l'ablazione e' l'unico dei tre che misura l'effetto e non una sua stima.

## Note collegate
[[attenzione-come-spiegazione]] · [[circuiti-e-interventi]]
