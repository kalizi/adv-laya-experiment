---
tags: [paper, interpretabilita, unita, 2023, openai]
---

# Language Models Can Explain Neurons in Language Models

**Bills, Cammarata, Mossing, Tillman, Gao, Goh, Sutskever, Leike, Wu & Saunders** · OpenAI, 2023 · `paper/bills-openai-language-models-explain-neurons.html`

## Cosa fa

Il lavoro di OpenAI su GPT-2 che viene citato come "lo studio del funzionamento interno".
Il metodo e' automatico e in due passi: GPT-4 scrive una spiegazione in linguaggio naturale
del comportamento di un neurone (dalle sue attivazioni piu' forti su un corpus), poi la
**punta** simulando le attivazioni attese su altre frasi e confrontandole con quelle vere.
Applicato a tutti i **307.200 neuroni di GPT-2**, con codice e dati rilasciati.

Risultati dichiarati: oltre 1.000 neuroni con punteggio ≥ 0.8, quindi spiegazioni che
secondo GPT-4 rendono conto della maggior parte del comportamento; la maggior parte dei
neuroni ben spiegati e' poco interessante; le spiegazioni restano peggiori di quelle umane;
cambiare la funzione di attivazione del modello spiegato migliora i punteggi.

## Il limite dichiarato dagli autori

Spiegano **cosa fa** un neurone come funzione del testo in ingresso, non quali effetti ha a
valle ne' quale meccanismo lo produce. E'una spiegazione del comportamento, non del
circuito.

## Perche' ci riguarda

E' l'antesignano dell'automatizzazione dell'interpretabilita', ed e' il riferimento da
citare quando nel paper si parla di spiegazioni automatiche. Per [[unita-di-analisi]] e'
anche il limite da superare: l'audit di [[audit-spiegazioni-neuroni]] mostra che le sue
spiegazioni sono molto meno affidabili di quanto il punteggio suggerisca.

Nota pratica per noi: su un modello 421M con hidden 1024 e 28 livelli, i neuroni sono
~30.000, e il metodo richiede un modello spiegante potente. E' fattibile in linea di
principio, ma e' il passo **dopo** che il meccanismo sia noto, non prima: spiegare neuroni
di un modello di cui non si conosce il comportamento interno produce descrizioni, non
comprensione.

## Note collegate
[[unita-di-analisi]] · [[audit-spiegazioni-neuroni]] · [[verso-la-monosemanticita]]
