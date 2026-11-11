---
tags: [paper, avversariale, teoria, 2019]
---

# Adversarial Examples Are Not Bugs, They Are Features

**Ilyas, Santurkar, Tsipras, Engstrom, Tran & Madry** · NeurIPS 2019 · arXiv:1905.02175 · `paper/ilyas-adversarial-examples-features.pdf`

## Cosa sostiene

L'esistenza e la **trasferibilita'** degli esempi avversari si spiegano con le **feature
non-robuste**: pattern derivati dalla distribuzione dei dati che sono altamente predittivi
ma fragili, e quindi invisibili all'umano. Il ragionamento ha due conseguenze operative:

1. un attacco che sfrutta feature non-robuste di un classificatore **trasferisce** a un
   classificatore diverso che usa feature simili, perche' le feature sono nella
   distribuzione, non nei pesi;
2. la vulnerabilita' non e' un difetto di implementazione ma una conseguenza del
   paradigma di apprendamento supervisionato, con un disallineamento fra la nozione umana
   di robustezza e la geometria dei dati.

## Perche' ci riguarda

E' l'ipotesi piu' potente per un risultato che la campagna ha gia' misurato: l'attacco E03
**trasferisce**. Trasferisce fra checkpoint (`laya`, `laya-multilingual`,
`laya-typed-decisions`) e fra architetture (DeBERTa in open-Jev e CLM, in
[[E15-quantizzazione]], con ASR 0.47-0.54 su `banking77`). Se il meccanismo fosse una
singolarita' dei pesi di un checkpoint, non trasferirebbe.

La predizione che ne discende, e che [[E23-meccanismo-injection]] misura: l'attacco
dovrebbe appoggiarsi a una componente interna **condivisa** fra checkpoint (una direzione
nello stato residuo, o un pattern di teste), non a una testa unica. Se invece la componente
non e' condivisa, la spiegazione del trasferimento va cercata altrove.

Il limite va detto: il paper lavora su immagini e perturbazioni continue, e la nozione di
"feature fragile" su testo e' molto piu' difficile da definire. Trasferiamo il **ragionamento**
sul trasferimento, non la formalizzazione.

## Note collegate
[[perche-gli-attacchi-funzionano]] · [[trigger-universali]] · [[E15-quantizzazione]]
