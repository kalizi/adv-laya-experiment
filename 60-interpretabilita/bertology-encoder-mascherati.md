---
tags: [interpretabilita, bertology, encoder, probe]
---

# Bertology: cosa si sa degli encoder mascherati

`laya` e' `ModernBERT-large` fine-tuned con una decision head addestrata da zero
([[architettura-laya]]). ModernBERT e' un discendente diretto di BERT: stesso obiettivo
di pre-training (masked language modeling), stessa architettura a blocchi, attenzione
alternata globale/locale, RoPE. Quindi la letteratura "BERTology" e' la letteratura del
nostro backbone, con un'avvertenza: riguarda modelli **pre-addestrati** e usati con una
testa lineare, non un modello fully fine-tuned con una testa decisionale proprietaria.

## I tre riferimenti

- **[[cosa-guarda-bert]]** (Clark et al.): le teste di BERT sviluppano pattern
  interpretabili (token delimitatore, offset posizionali, attenzione diffusa) e alcune
  teste corrispondono a nozioni sintattiche reali (complemento oggetto, determinante del
  nome, coreferenza), con una sonda costruita **sulle mappe di attenzione** che arriva a
  77 UAS sul parsing. Utile a noi: la struttura sintattica vive nelle teste, non solo
  negli stati nascosti.
- **[[bert-riscopre-la-pipeline]]** (Tenney et al.): congelato l'encoder, l'informazione
  linguistica compare in un ordine localizzabile: POS, parsing, NER, ruoli semantici,
  coreferenza. Le decisioni individuali pero' possono essere riviste piu' in alto, quindi
  "dove vive l'informazione" e "quando viene usata" sono domande separate. Per noi: la
  decisione di Laya sta in testa ai 28 livelli piu' la decision head, ma **quale** livello
  prepara la scelta e' una domanda a cui rispondere con un probe.
- **[[bertology-primer]]** (Rogers et al.): la sintesi di oltre 150 studi, ed e' il paper
  piu' utile per tarare le aspettative. Le sue conclusioni ci riguardano quasi una per una:
  i modelli si appoggiano a **euristiche superficiali**; gli ECE e le misure di
  calibrazione vanno ricalcolati per modello e dominio; la compressione/quantizzazione
  cambia i numeri ma raramente il comportamento qualitativo; e la visualizzazione
  qualitativa **non e' evidenza**.

## Cosa trasferisce a Laya, e cosa no

Trasferisce:

- La struttura a livelli come luogo dove localizzare (decision depth, "a che profondita'
  lo stato nascosto del `[MASK]` sa gia' quale opzione verra' scelta").
- Il metodo dei probe con control task ([[control-task-probe]]).
- L'avvertenza sulle euristiche superficiali: un encoder mascherato fine-tuned su
  decisioni puo' risolvere il compito con una scorciatoia lessicale. E13 lo sospettava
  dal comportamento (il testo delle opzioni pesa 27-29pp), e qui c'e' il metodo per
  verificarlo dentro il modello.

Non trasferisce:

- I probe lineari sui token `[CLS]`: Laya non usa `[CLS]`, legge gli stati ai token
  `[MASK]` delle opzioni. Il punto di lettura e' diverso, quindi i probe vanno rifatti
  li'.
- Le analisi su BERT-base (12 livelli, attenzione piena): Laya ha 28 livelli, 10
  globali e 18 locali con finestra ±64, quindi la "profondita'" di un risultato di
  BERTology non si trasferisce in termini di indice di livello.

## La domanda che questa letteratura ci lascia

Bertology descrive cosa c'e' **nelle rappresentazioni**. Non descrive quale circuito lo
**usa** per decidere. Per un modello decisionale la differenza e' il punto: un'informazione
puo' essere presente nello stato nascosto e non essere letta dalla decision head. Il
passaggio da "c'e'" a "viene usato" richiede interventi causali
([[probe-e-controfattuali]]).

## Note collegate
[[architettura-laya]] · [[E13-label-semantics]] · [[probe-e-controfattuali]] ·
[[unita-di-analisi]]
