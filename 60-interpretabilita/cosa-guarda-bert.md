---
tags: [paper, interpretabilita, bertology, 2019]
---

# What Does BERT Look At?

**Clark, Khandelwal, Levy & Manning** · BlackboxNLP@ACL 2019 · arXiv:1906.04341 · `paper/clark-khandelwal-what-does-bert-look-at.pdf`

## Cosa misura

Le 144 teste di BERT-base, con metodi applicabili a qualsiasi modello con attenzione. I
pattern trovati sono di tre tipi: attenzione a token **delimitatori**, a **offset
posizionali** fissi, o diffusa sull'intera frase. E alcune teste corrispondono a nozioni
sintattiche vere: complemento oggetto del verbo, determinante del nome, oggetto di
preposizione, coreferenti, con accuratezze alte.

In piu', una sonda costruita **sulle mappe di attenzione** (non sugli stati nascosti)
raggiunge 77 UAS sul dependency parsing: l'informazione sintattica sta anche nelle teste,
non solo nelle rappresentazioni.

## Perche' ci riguarda

E' la prova che una testa puo' avere un **ruolo leggibile** senza supervisione esplicita,
ed e' il motivo per cui ha senso cercare, in Laya, una testa (o poche) con un ruolo del
tipo "copia il token nominato". E' anche il riferimento da citare quando si descrive la
nostra misura di raggiungibilita' ±64 ([[E22-attenzione-laya]]): la struttura locale/globale
non e' un dettaglio di implementazione, cambia quali pattern possono esistere.

L'avvertenza, che viene da [[attenzione-non-e-spiegazione]] e da
[[bertology-primer]]: un pattern interpretabile in una testa non dimostra che quella testa
**causi** la predizione. La sonda su attenzione e' un risultato di disponibilita', non
d'uso.

## Note collegate
[[bertology-encoder-mascherati]] · [[teste-specializzate]] · [[E22-attenzione-laya]]
