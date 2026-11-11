---
tags: [paper, interpretabilita, bertology, 2019]
---

# BERT Rediscovers the Classical NLP Pipeline

**Tenney, Das & Pavlick** · ACL 2019 · arXiv:1905.05950 · `paper/tenney-das-bert-rediscovers-pipeline.pdf`

## Cosa misura

Dove vive l'informazione linguistica in un encoder pre-addestrato, misurata con **edge
probing**: sonde addestrate su rappresentazioni congelate, un task per stadio della
pipeline classica. Risultato: l'ordine atteso compare ed e' localizzabile (POS, parsing,
NER, ruoli semantici, coreferenza), con informazioni piu' superficiali nei livelli bassi e
piu' semantiche in alto.

Il risultato piu' interessante e' l'**eccezione**: l'ordine regge in aggregato, ma le
decisioni individuali possono dipendere l'una dall'altra in modo arbitrario, rimandando
quelle ambigue o correggendosi con informazione di livello piu' alto. Cioe' "dove vive
l'informazione" e "quando viene usata" sono domande diverse.

## Perche' ci riguarda

Per [[E23-meccanismo-injection]] questo e' il modello della **primamedia**: a quale
profondita' lo stato al `[MASK]` dell'opzione "sa gia'" quale opzione verra' scelta. Se la
decisione di Laya e' una pipeline, il punto in cui la nominazione entra nel campo
dell'opzione dovrebbe essere localizzabile con un probe per livello.

L'avvertenza dell'eccezione va presa alla lettera: anche se troviamo il livello, il fatto
che l'informazione sia *presente* li' non dice che *venga usata* da quella testa. Il
passaggio richiede [[probe-e-controfattuali]].

## Note collegate
[[bertology-encoder-mascherati]] · [[control-task-probe]] · [[mediazione-causale]]
