---
tags: [paper, interpretabilita, unita, 2023]
---

# Towards Monosemanticity

**Bricken, Templeton, Batson, Chen, Jermyn, Conerly, Turner, Anil, Denison, Askell, Lasenby, Wu, Kravec, Schiefer, Maxwell, Joseph, Tamkin, Nguyen, McLean, Burke, Hume, Carter, Henighan & Olah** · Transformer Circuits Thread (Anthropic) 2023 · `paper/bricken-towards-monosemanticity.html`

## Cosa fa

La prova di principio che esiste un'unita' di analisi migliore del neurone. Su un
transformer a **un solo livello** con MLP da 512 neuroni, un autoencoder sparso con
espansione fino a **256x** (131.072 feature) decompone le attivazioni in feature
interpretabili; il run A/1 con 4.096 feature viene analizzato nel dettaglio, e le feature
mostrate sono sensibilmente piu' monosemantiche dei neuroni.

Il paper affronta anche il problema pratico: gli autoencoder sparsi fanno overfitting, e la
soluzione adottata e' una procedura di addestramento con dati enormi (8 miliardi di punti)
e una cura esplicita nel riportare i fallimenti, non solo i successi.

## Perche' ci riguarda

E' il candidato per l'unita' "feature" di [[unita-di-analisi]], ed e' la base di
[[letteratura-collaterale]] (i circuiti su feature). Ma per noi e' **fuori portata** nel
2026, per due ragioni concrete: servono attivazioni-addestramento su un encoder 421M
(l'addestramento di un SAE e' un progetto a se'), e l'audit di
[[audit-spiegazioni-neuroni]] dice che le spiegazioni di feature non sono automaticamente
affidabili.

Il valore per il progetto e' un'altra: la **dimostrazione** che la polisemanticita' e' un
artefatto della sovrapposizione, non una proprieta' necessaria. Questo cambia il modo di
leggere un risultato negativo: se in Laya nessun singolo componente controlla l'attacco,
la spiegazione piu' probabile e' la sovrapposizione, non la nostra incapacita' di cercare.

## Note collegate
[[unita-di-analisi]] · [[audit-spiegazioni-neuroni]] · [[letteratura-collaterale]]
