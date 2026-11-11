---
tags: [paper, interpretabilita, teste, 2019]
---

# Are Sixteen Heads Really Better than One?

**Michel, Levy & Neubig** · NeurIPS 2019 · arXiv:1905.10650 · `paper/michel-levy-neubig-sixteen-heads.pdf`

## Cosa misura

Quante teste di attenzione si possono togliere a tempo di inferenza senza perdere
prestazioni? Il risultato e' sorprendente: molte, e per alcuni livelli una sola testa
basta. Con pruning greedy su BERT si arriva a togliere il **40%** delle teste senza
degrado percettibile (su traduzione il 20%, oltre il quale crolla).

Un dettaglio rilevante: le teste importanti lo sono in modo **universale**, cioe' lo
stesso insieme conta su domini diversi. Questo e' un indizio che esista un piccolo
nocciolo di teste che porta il comportamento, e che sia individuabile.

## Perche' ci riguarda

E' il metodo piu' economico per passare da "l'attacco funziona" a "queste teste lo
fanno funzionare": si classificano le teste per importanza e si misura l'effetto
sull'**ASR** invece che sull'accuratezza. La predizione e' chiara e falsificabile: se
l'attacco e' portato da poche teste, toglierle deve ridurre l'ASR piu' di quanto riduca
l'accuratezza pulita.

L'avvertenza speculare e' nella letteratura vicina: [[teste-specializzate]] trova che le
teste che restano dopo il pruning sono quelle interpretabili, quindi un attacco portato da
teste inerti sarebbe sospetto. Se E23 trova che l'ASR non si muove togliendo teste, la
conclusione e' che il meccanismo e' **distribuito**, non che l'esperimento ha sbagliato.

## Note collegate
[[circuiti-e-interventi]] · [[teste-specializzate]]
