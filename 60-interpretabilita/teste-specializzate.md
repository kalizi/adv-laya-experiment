---
tags: [paper, interpretabilita, teste, 2019]
---

# Analyzing Multi-Head Self-Attention

**Voita, Talbot, Moiseev, Sennrich & Titov** · ACL 2019 · arXiv:1905.09418 · `paper/voita-analyzing-multihead-self-attention.pdf`

## Cosa misura

Il contributo delle singole teste dell'**encoder** alla prestazione complessiva, con un
metodo che non e' l'ablazione uno-a-uno ma un **gate stocastico** con rilassamento
differenziabile della penalita' L0. Le teste vengono classificate per ruolo (posizionale,
sintattico, raro) e si osserva che le teste specializzate sono le **ultime** a essere
potate.

Il numero che colpisce: su WMT en-ru si potano **38 teste su 48 dell'encoder** con un
calo di solo **0.15 BLEU**. Cioe' la maggior parte delle teste dell'encoder e'
praticamente inerte.

## Perche' ci riguarda

Due insegnamenti, uno sul metodo e uno sul risultato.

**Sul metodo**: un gate addestrato e' un intervento *appreso*, quindi misura
un'importanza media sul dataset. Per un attacco specifico (E03 su un caso specifico) serve
l'ablazione **singola e controllata**, non una media. Il metodo si prende in prestito per
la **classificazione dei ruoli**, non per la potatura.

**Sul risultato**: se anche sul nostro backbone la maggior parte delle teste e' inerte,
allora un esperimento di knockout esaustivo (28 livelli × 16 teste) e' piu' economico di
quanto sembri: molti knockout non faranno nulla, e il numero di quelli che fanno qualcosa
e' gia' un risultato. E' esattamente il disegno di screening di
[[E23-meccanismo-injection]].

## Note collegate
[[sedici-teste-meglio-di-una]] · [[circuiti-e-interventi]]
