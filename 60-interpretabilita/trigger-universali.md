---
tags: [paper, avversariale, attacco, 2019]
---

# Universal Adversarial Triggers for Attacking and Analyzing NLP

**Wallace, Feng, Kandpal, Gardner & Singh** · EMNLP-IJCNLP 2019 · arXiv:1908.07125 · `paper/wallace-universal-adversarial-triggers.pdf`

## Cosa fa

Cerca sequenze di token **agnostiche rispetto all'input** che inducono una predizione
target quando sono concatenate a qualsiasi input del dataset, con una ricerca guidata dal
gradiente. I numeri sono il punto:

- su SNLI l'accuratezza passa da **89.94% a 0.55%** con un trigger di una parola;
- il **72%** delle domande "why" di SQuAD riceve la stessa risposta ("to kill american
  people") indipendentemente dalla domanda;
- su GPT-2 il trigger produce output tossici anche a partire da contesti neutri.

Il paper e' duplice per costruzione: attacco **e** strumento di analisi, perche' un trigger
che funziona su tutti gli input dice dove sta l'euristica del modello (i trigger confermano
per esempio che i modelli SNLI sfruttano i bias del dataset).

## Perche' ci riguarda

Distingue due cose che la nostra campagna ha tenuto separate: E03 **non** e' un trigger
universale (dipende dal caso: cambia l'opzione nominata), ma la tecnica da' un test
economico dell'ipotesi "esiste una scorciatoia superficiale". Se esistesse una sequenza
breve, indipendente dallo stato, che sposta la decisione verso un'opzione data, il
meccanismo sarebbe molto piu' superficiale di quanto l'ASR alto faccia sembrare.

L'uso come strumento di analisi e' il modello di [[perche-gli-attacchi-funzionano]]: un
attacco che generalizza troppo e' un rivelatore di euristica, non un'abilita'.

## Note collegate
[[perche-gli-attacchi-funzionano]] · [[esempi-avversari-feature]] ·
[[E03-option-injection]]
