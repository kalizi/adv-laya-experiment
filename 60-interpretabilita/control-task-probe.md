---
tags: [paper, interpretabilita, probe, 2019]
---

# Designing and Interpreting Probes with Control Tasks

**Hewitt & Liang** · EMNLP-IJCNLP 2019 · arXiv:1909.03368 · `paper/hewitt-liang-control-tasks.pdf`

## Cosa risolve

Una sonda abbastanza espressiva impara **qualsiasi** task su una rappresentazione
lossless: quindi un'accuratezza alta non dimostra che l'informazione sia **nella**
rappresentazione, puo' dimostrare solo che la sonda e' capace. Il rimedio e' il **control
task**: associare a ogni tipo di parola un'uscita **casuale**, costruita in modo che solo
la sonda possa impararla. Una buona sonda e' **selettiva**: alta sull'attivita' vera,
bassa sul control.

I numeri dell'esempio piu' citato: su ELMo, le sonde MLP arrivano a 97.3 sul part-of-speech
e **92.8 sul control**, una selettivita' di 4.5 punti. Con iperparametri scelti per la
selettivita' la sonda lineare arriva a 33.0 di selettivita'. E il dropout, la scorciatoia
standard, non migliora la selettivita'.

## Perche' ci riguarda

E' il paper che impedisce il falso positivo piu' costoso del programma: "lo stato al
`[MASK]` dell'opzione nominata codifica la nominazione" puo' essere vero per ragioni che
non c'entrano con la decisione. La verifica e' il control task abbinato, e il progetto ha
gia' i materiali per costruirlo: il **braccio neutro** di [[E22-attenzione-laya]] (stessa
cornice, nessuna opzione nominata) e il caso pulito sono input in cui la nominazione e'
presente ma la predizione non cambia.

Nota operativa: la selettivita' va riportata **insieme** all'accuratezza, sempre. Una
sonda con 0.95 di accuratezza e 3 punti di selettivita' sta misurando se stessa.

## Note collegate
[[probe-e-controfattuali]] · [[bertology-encoder-mascherati]] · [[E22-attenzione-laya]]
