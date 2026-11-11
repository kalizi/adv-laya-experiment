---
tags: [interpretabilita, circuiti, patching, causalita]
---

# Circuiti e interventi

La famiglia di metodi che trasforma una correlazione interna in una causa: si **estrae**
un pezzo del grafo computazionale, lo si **sostituisce**, e si misura cosa succede
all'uscita. E' l'unica famiglia che produce spiegazioni testabili, ed e' il cuore del
programma [[E23-meccanismo-injection]].

## Il metodo, in ordine di apparizione

- **[[mediazione-causale]]** (Vig et al.) e' il vocabolario: trattamento, mediatore,
  risposta, effetti totale e mediato.
- **[[rome-editing]]** (Meng et al.) e' il **causal tracing**: si corre l'input, si
  ripristinano gli stati interni puliti in una posizione (layer, token) alla volta, e si
  guarda dove il ripristino recupera la risposta corretta. Su GPT trova che le
  associazioni fattuali passano per i moduli feed-forward **di medio livello**, nell'ultimo
  token del soggetto. L'idea del "ripristina un pezzo e vedi se torna" e' esattamente
  quello che serve per un'iniezione: il pezzo da ripristinare e' lo stato al `[MASK]`
  dell'opzione giusta.
- **[[circuito-ioi-gpt2]]** (Wang et al.) e' il modello da seguire per un'esperimento
  completo: 26 teste in 7 classi, trovate combinando analisi delle mappe, **path
  patching** e **knockout** delle teste, e validate con tre criteri espliciti:
  *fedelta'* (il circuito riproduce il comportamento), *completezza* (contiene tutti i
  nodi usati), *minimalita'* (non ne contiene di inutili). I tre criteri insieme sono la
  parte che rende la spiegazione una misura: senza minimalita' qualsiasi sovrainsieme
  "spiega" tutto.
- **[[attribution-patching]]** (Syed et al.) e' l'accelerazione: il patching esatto
  richiede un forward per intervento, e una sua approssimazione gradienti-based e'
  comparabile nel recuperare i circuiti e molto piu' veloce. Per noi e' la differenza
  fra una settimana e un pomeriggio.

## Perche' serve a un encoder con una testa decisionale

Laya ha un'architettura che si presta bene a questo metodo, meglio di un LLM:

- **Un solo punto di lettura per opzione**: la decision head legge gli stati ai token
  `[MASK]` ([[architettura-laya]]). Quindi il mediatore naturale e' uno stato in una
  posizione nota, non "qualche neurone da qualche parte".
- **Un'uscita confrontabile**: il margine fra l'opzione scelta e quella giusta, dalle
  `probs` gia' presenti nei record. Il patching si misura su un numero che il progetto
  calcola gia'.
- **Forward deterministico**: il modello e' deterministico su questo setup (0/150 label
  diverse fra run ripetute), quindi un delta dopo un intervento non e' rumore. E' la
  precondizione per usare interventi singoli, e va verificata di nuovo prima di
  interpretare ([[performance-velocita]]).

## Cosa questo metodo NON puo' fare qui

- **Non dimostra che il circuito trovato e' l'unico.** Due meccanismi diversi possono
  dare la stessa uscita sullo stesso dataset (e' il problema di esistenza vs
  esclusivita' di [[attenzione-non-non-e-spiegazione]], versione causale).
- **Non scala a tutti gli interventi su N pieno**: 28 livelli × 16 teste = 448 knockout,
  e su N=2974 casi sono 1.3M forward. Il disegno va quindi in due fasi: screening su un
  sottoinsieme, conferma su N pieno dei pochi componenti che sopravvivono (E23 lo
  formalizza).
- **I controlli non sono opzionali**: ogni intervento va ripetuto sul caso pulito, per
  separare "togliere questa testa rompe l'attacco" da "togliere questa testa rompe il
  modello".

## Note collegate
[[probe-e-controfattuali]] · [[unita-di-analisi]] · [[perche-gli-attacchi-funzionano]]
