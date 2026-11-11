---
tags: [paper, interpretabilita, unita, 2023]
---

# Eliciting Latent Predictions with the Tuned Lens

**Belrose, Furman, Smith, Halawi, Ostrovsky, McKinney, Biderman & Steinhardt** · arXiv 2023 · arXiv:2303.08112 · `paper/belrose-tuned-lens.pdf`

## Cosa fa

Il **logit lens** legge lo stato nascosto di un livello intermedio con l'unembedding del
modello, trattando la predizione latente come una predizione. Funziona poco: le feature si
spostano con la profondita' (representational drift), quindi la lettura e' distorta e per
molti modelli inutilizzabile. Il **tuned lens** addestra una trasformazione affine per
livello, con una loss di distillazione sull'uscita finale, e ottiene predizioni latenti
piu' rappresentative e meno distorte. Gli autori verificano con esperimenti causali che il
tuned lens usa feature simili a quelle del modello.

## Il risultato che ci riguarda subito

Gli autori riportano che la **traiettoria** delle predizioni latenti (come la predizione
cambia livello per livello) rileva input malevoli con accuratezza alta. Cioe' una difesa
che non tocca i pesi e non guarda il testo, ma guarda **quanto in fretta** il modello si
decide.

## Perche' ci riguarda

E' lo strumento per la domanda "a che profondita' la decisione e' presa", ed e' il
corrispettivo addestrato del probe per livello di [[bert-riscopre-la-pipeline]]. Per Laya
la lettura intermedia e' fattibile: la decision head da' una distribuzione sulle opzioni, e
applicarla livello per livello (con una sonda affine addestrata sui casi puliti) darebbe
una curva di decisione per profondita'.

La connessione con la difesa e' concreta e da verificare: se un input iniettato fa
"saltare" la decisione prima (a un livello piu' superficiale) di quanto faccia un input
pulito, la profondita' di decisione e' un segnale. E' un segnale **interno**, quindi molto
piu' difficile da falsificare di una regola lessicale come [[E18-nomination-detector]], ed
e' lo stesso tipo di idea del detector sulla nominazione, ma fondata sul meccanismo invece
che sulla lessicale.

## Note collegate
[[unita-di-analisi]] · [[bert-riscopre-la-pipeline]] · [[E18-nomination-detector]]
