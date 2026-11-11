---
tags: [paper, interpretabilita, causalita, 2020]
---

# Causal Mediation Analysis for Interpreting Neural NLP

**Vig, Gehrmann, Belinkov, Qian, Nevo, Singer & Shieber** · BlackboxNLP 2020 · arXiv:2004.12265 · `paper/vig-causal-mediation-gender-bias.pdf`

## Cosa introduce

Il formalismo della **mediazione causale** dentro le reti neurali: trattamento (intervento
sull'input), mediatore (un neurone, una testa, un livello), risposta (la predizione). Si
misurano effetto totale ed effetto mediato, e il mediatore si puo' sostituire con la sua
versione controfattuale. Il caso studio e' il gender bias, ma il metodo e' generale.

I tre aggettivi con cui gli autori descrivono gli effetti trovati sono una checklist utile
per qualsiasi analisi successiva: l'effetto e' **sparse** (concentrato in pochi
componenti), **sinergico** (amplificato o represso da componenti diversi) e
**decomponibile** (in effetto diretto dall'input e indiretto attraverso i mediatori).

## Perche' ci riguarda

E' il vocabolario formale di [[circuiti-e-interventi]], e il paper citato esplicitamente
da [[circuito-ioi-gpt2]] come base del path patching. Per [[E23-meccanismo-injection]] da'
la struttura della misura principale: l'effetto totale e' l'ASR (che gia' abbiamo), e
l'effetto mediato e' la frazione di flip recuperata sostituendo un mediatore interno con
quello del caso pulito.

La lezione metodologica: il mediatore va scelto **prima** di guardare i dati, altrimenti
si cerca il componente che funziona. Nel nostro caso la scelta e' obbligata
dall'architettura: i punti di lettura sono gli stati ai `[MASK]` delle opzioni, quindi il
mediatore naturale e' quello, livello per livello.

## Note collegate
[[probe-e-controfattuali]] · [[circuiti-e-interventi]] · [[rome-editing]]
