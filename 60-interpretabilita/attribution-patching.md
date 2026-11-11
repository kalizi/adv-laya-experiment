---
tags: [paper, interpretabilita, circuiti, 2023]
---

# Attribution Patching Outperforms Automated Circuit Discovery

**Syed, Rager & Conmy** · arXiv 2023 · arXiv:2310.10348 · `paper/syed-attribution-patching.pdf`

## Cosa fa

- Attribuzione (attribution patching) **sovrastima** il patching esatto ed e' molto piu'
veloce: un forward invece di uno per intervento.
- Confronta il metodo con ACDC, la procedura automatica di scoperta dei circuiti, e trova
  che e' competitivo nel recuperare i circuiti e sostanzialmente piu' veloce.

E' un risultato di **ingegneria della misura**: il costo di un intervento causale esatto
cresce con il numero di componenti, e approssimare il primo ordine cambia di molto il
costo e poco il responso.

## Perche' ci riguarda

Nel nostro caso il conto e' quello che blocca il disegno: 28 livelli × 16 teste = 448
componenti, e un patching esatto per componente su N=2974 casi sono ~1.3M forward. Con una
stima al primo ordine il costo crolla di ordini di grandezza, e la differenza fra le due
stime e' essa stessa una misura: dove divergono, l'effetto non e' lineare e va rifatto con
l'intervento esatto.

Questo e' il motivo per cui [[E23-meccanismo-injection]] prevede screening su sottoinsieme
e conferma su N pieno sui pochi sopravvivono, invece di un esaustivo su N pieno.
L'avvertenza ovvia: l'approssimazione va **validata** su un sottoinsieme, mai assunta.

## Note collegate
[[circuiti-e-interventi]] · [[circuito-ioi-gpt2]]
