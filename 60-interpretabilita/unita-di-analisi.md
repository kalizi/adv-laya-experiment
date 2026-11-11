---
tags: [interpretabilita, feature, neuroni, automatizzazione]
---

# L'unita' di analisi

Prima di misurare un meccanismo bisogna decidere **di cosa** e' fatto. Le opzioni in
letteratura sono almeno quattro, e la scelta determina cosa si puo' concludere.

| unita' | cos'e' | vantaggio | svantaggio |
|---|---|---|---|
| token | una parola nel testo | leggibile da chiunque | non e' un oggetto interno: il modello non ragiona sui token |
| testa di attenzione | una delle 16 teste di un livello | intervengo con un knockout, effetto causale diretto | ridondanza: togliere una testa spesso non cambia niente |
| direzione nello stato residuo | una direzione lineare | oggetto naturale per un confronto fra opzioni | polisemantica: la stessa direzione fa cose diverse in contesti diversi |
| feature (dizionario sparso) | direzione trovata da un autoencoder sparso | piu' monosemantica del singolo neurone | va **addestrata**, quindi e' un oggetto approssimato |

## Le quattro voci della cartella

- **[[spiegare-i-neuroni]]** (OpenAI, 2023): GPT-4 scrive spiegazioni in linguaggio
  naturale dei neuroni di **GPT-2** (307.200 neuroni) e le punteccia simulando le
  attivazioni. E' il riferimento che l'utente conosce come "OpenAI che studia il
  funzionamento di GPT-2". Risultati: oltre 1.000 neuroni con punteggio ≥ 0.8, ma la
  maggior parte poco interessante, e le spiegazioni restano peggiori di quelle umane;
  cambiare la funzione di attivazione del modello spiegato migliora i punteggi. Il metodo
  spiega **cosa fa** un neurone come funzione del testo, non quali effetti ha a valle.
- **[[verso-la-monosemanticita]]** (Anthropic): la risposta al problema della
  polisemanticita'. Su un transformer a un solo livello con MLP da 512 neuroni, gli
  autoencoder sparsi fino a 256x producono 131.072 feature, e il run A/1 con 4.096
  feature contiene feature interpretabili e sensibility testabili. E' la prova di
  principio che esiste un'unita' migliore del neurone, su un modello pero' molto piu'
  piccolo del nostro.
- **[[audit-spiegazioni-neuroni]]** (Huang et al.): la doccia fredda, ed e' il paper piu'
  utile dei tre. Audita il metodo di OpenAI su GPT-2 XL con due criteri (osservazionale:
  la spiegazione predice le attivazioni; causale: la spiegazione sopravvive a un
  intervento). Fra i neuroni che GPT-4 considera spiegati meglio (top 0.6%), la
  **precisione e' 0.64 e il recall 0.50**, e l'efficacia causale e' pressocche' nulla.
  Conclusione: le spiegazioni automatiche in linguaggio naturale non sono ancora
  affidabili, e i neuroni non sono probabilmente l'unita' giusta.
- **[[tuned-lens]]** (Belrose et al.): decodifica lo stato nascosto di **ogni** livello
  con una sonda affine addestrata per livello, invece di usare l'unembedding diretto
  (logit lens), perche' le feature si spostano con la profondita' (representational
  drift). E la prova che una lettura intermedia e' possibile e affidabile. Nota che ci
  riguarda da vicino: gli autori trovano che la traiettoria delle previsioni latenti
  **rileva input malevoli** con accuratezza alta, che e' una difesa possibile senza
  toccare i pesi.

## La scelta per questo progetto

Per [[E23-meccanismo-injection]] l'unita' primaria e' la **testa di attenzione** e il
**punto di lettura** (lo stato al `[MASK]`), per tre ragioni: sono oggetti su cui un
knockout e' un intervento causale diretto; sono pochi (28 × 16 teste, K opzioni); e non
richiedono di addestrare nulla, quindi non introducono un artefatto nostro dentro
l'analisi. Le direzioni nello stato residuo entrano come **seconda unita'** (il
confronto fra opzioni e' naturale in quello spazio), e le feature sparse restano
dichiarate fuori portata: 421M parametri e un autoencoder da addestrare e' un progetto
diverso, e l'audit di Huang et al. dice che non e' il primo strumento da prendere.

## Note collegate
[[circuiti-e-interventi]] · [[bertology-encoder-mascherati]] · [[letteratura-collaterale]]
