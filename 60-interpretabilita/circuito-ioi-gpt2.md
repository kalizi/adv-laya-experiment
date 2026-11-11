---
tags: [paper, interpretabilita, circuiti, 2023]
---

# Interpretability in the Wild: il circuito IOI di GPT-2 small

**Wang, Variengien, Conmy, Shlegeris & Steinhardt** · ICLR 2023 · arXiv:2211.00593 · `paper/wang-ioi-circuit-gpt2-small.pdf`

## Cosa fa

Spiega **come** GPT-2 small esegue un compito naturale (indirect object identification:
"when Mary and John went to the store, John gave a drink to ..." → Mary). Il circuito
trovato comprende **26 teste** raggruppate in 7 classi, scoperte combinando analisi delle
mappe di attenzione, **path patching** e **knockout** delle teste. Alcune classi sono
intuitivamente leggibili (le "name mover" spingono il nome corretto, altre lo contrastano),
e il fatto che esistano teste che spingono in direzione opposta e' un promemoria che un
circuito non e' una somma di contributi positivi.

## I tre criteri, che sono la parte importante

La spiegazione viene valutata con tre criteri quantitativi, e vanno insieme:

- **Fedelta'**: il circuito riproduce il comportamento del modello completo.
- **Completezza**: il circuito contiene tutti i nodi usati per il comportamento.
- **Minimalita'**: non contiene nodi inutili.

Senza minimalita' qualunque sovrainsieme "spiega" il comportamento, ed e' il motivo per cui
un'ablazione esaustiva va interpretata con una soglia dichiarata, non a occhio.

## Perche' ci riguarda

E' il modello di [[E23-meccanismo-injection]], per tre ragioni: la scala e' paragonabile
(un modello piccolo, un comportamento specifico, un circuito discreto), il metodo e'
trasferibile a un encoder (path patching e knockout non richiedono un decoder), e i tre
criteri danno uno standard di rigore che il progetto puo' adottare senza inventarne uno.

L'avvertenza: gli autori dichiarano che i criteri **supportano** la spiegazione ma
lasciano lacune, e che l'approccio va scalato con cautela. Per noi la scala e' un problema
reale: 28 livelli × 16 teste = 448 componenti, quindi il disegno deve essere in due fasi
(screening su sottoinsieme, conferma su N pieno).

## Note collegate
[[circuiti-e-interventi]] · [[mediazione-causale]] · [[attribution-patching]]
