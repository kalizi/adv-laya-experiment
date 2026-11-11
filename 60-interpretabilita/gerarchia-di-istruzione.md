---
tags: [paper, avversariale, difesa, 2024]
---

# The Instruction Hierarchy

**Wallace, Xiao, Leike, Weng, Heidecke & Beutel** · OpenAI · arXiv:2404.13208 · `paper/wallace-instruction-hierarchy.pdf`

## Cosa sostiene

La causa comune di prompt injection, jailbreak e sovrascrittura delle istruzioni e' che
gli LLM non hanno una **gerarchia di privilegi**: un testo di terze parti puo' sovrascrivere
quello di sistema perche' i due sono trattati allo stesso modo. Il rimedio proposto e'
addestrare la gerarchia (sistema > utente > terze parti) con due principi: *context
synthesis* (le istruzioni allineate vengono scomposte e ricomposte ai vari livelli) e
*context ignorance* (per le istruzioni in conflitto si allena il modello a rispondere come
se non le avesse viste). La robustezza migliora anche su tipi di attacco **non visti** in
addestramento.

Gli autori distinguono anche dominio aperto e dominio chiuso, che e' una distinzione utile:
un task chiuso (classificazione, riassunto) ha un esito verificabile, uno aperto no.

## Perche' ci riguarda

E' la formalizzazione piu' vicina al nostro caso. Il formato wire di Laya separa
**domanda, criteri e opzioni** dallo **stato** ([[primitive-typed]]): due campi, due
ruoli, ma nessuna garanzia che il modello abbia appreso la separazione. Se non l'ha
appresa, E03 non e' un attacco al testo: e' un attacco alla mancanza di privilegi fra
canali.

L'ipotesi e' verificabile, ed e' uno dei tre bracci di [[E23-meccanismo-injection]]: la
nominazione deve avere effetto **solo** quando sta nel campo "stato". Se la stessa frase
inserita nelle criteria produce lo stesso ribaltamento, il modello non ha alcuna nozione di
canale, e la difesa non puo' essere una regola sul campo (dovrebbe essere una regola sul
contenuto, che e' quello che [[E18-nomination-detector]] fa, con i suoi limiti lessicali).

## Note collegate
[[perche-gli-attacchi-funzionano]] · [[primitive-typed]] · [[E18-nomination-detector]]
