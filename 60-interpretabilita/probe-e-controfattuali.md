---
tags: [interpretabilita, probe, causalita, metodo]
---

# Probe e controfattuali

Due famiglie di metodo per passare da "il modello si comporta cosi'" a "il modello
**usa** questa informazione". Sono le uniche due che producono affermazioni
falsificabili su un modello che non generiamo noi.

## Probe, con il control task

**[[control-task-probe]]** (Hewitt & Liang) risolve il problema classico dei probe: una
sonda abbastanza espressiva impara **qualsiasi** task su una rappresentazione
lossless, quindi un'accuratezza alta non dimostra che l'informazione ci sia. Il rimedio
e' il control task: associare a ogni tipo di parola un'uscita **casuale**, costruita in
modo che solo la sonda possa impararla. Una buona sonda e' **selettiva**: alta
sull'attivita' linguistica vera, bassa sul control. Il risultato e' istruttivo: con
iperparametri normali, le sonde MLP su ELMo arrivano a 97.3 sul POS e **92.8 sul
control**, una selettivita' di 4.5 punti. Una sonda poco selettiva sta misurando la
propria capacita', non la rappresentazione.

Per noi la conseguenza e' secca: un probe che dice "lo stato al `[MASK]` dell'opzione
nominata codifica la nominazione" vale zero senza il control task abbinato. E il dropout,
che e' la scorciatoia standard, non migliora la selettivita'.

## Mediazione causale

**[[mediazione-causale]]** (Vig et al.) porta dentro le reti neurali il formalismo della
mediazione causale: trattamento (intervento sull'input), mediatore (un neurone, una
testa, un livello), risposta (la previsione). Si misurano effetto totale e effetto
mediato, e le categorie che ne escono sono utili: l'effetto e' **sparse** (concentrato
in pochi componenti), **sinergico** (amplificato o represso da componenti diversi) e
**decomponibile**. E' il ponte fra analisi strutturale e comportamentale, ed e' il
formalismo che poi useranno tutti i lavori di circuiti ([[circuito-ioi-gpt2]] lo cita
esplicitamente come base).

## Come si usano qui, concretamente

La differenza fra le due famiglie, in una riga: il probe dice **se** un'informazione e'
disponibile in una rappresentazione; la mediazione dice **se** quella rappresentazione
causa la decisione. Per [[E23-meccanismo-injection]] servono entrambe, in quest'ordine:

1. **Probe**: lo stato al `[MASK]` dell'opzione nominata predice la scelta? Con control
   task (per esempio: predire l'opzione nominata da stati in cui la nominazione e'
   presente ma la decisione non cambia, come il braccio neutro di E22).
2. **Mediazione**: sostituendo quello stato con quello del caso pulito, la decisione
   tornerebbe corretta? Misurato **livello per livello**, questo da' la profondita' a cui
   l'attacco "si fissa".

Le due misure non si sostituiscono, e la coppia e' quella che evita il falso positivo
piu' costoso: uno stato che codifica la nominazione per ragioni che non hanno nulla a che
fare con la decisione.

## Note collegate
[[bertology-encoder-mascherati]] · [[circuiti-e-interventi]] · [[attribuzioni-e-sanity-check]]
