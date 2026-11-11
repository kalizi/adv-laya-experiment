---
tags: [interpretabilita, attenzione, metodo]
---

# L'attenzione non e' una spiegazione

Il primo istinto, di fronte a un transformer che sembra "leggere" una frase iniettata, e'
guardare i pesi di attenzione. E22 l'ha fatto, e ha trovato che la quantita' di attenzione
sullo span **non** spiega l'ASR. Questo mette il risultato in un contesto: non e' una
scoperta isolata, e' il caso particolare di una letteratura che va avanti dal 2019.

## La famiglia di risultati

Tre paper indipendenti, stesso anno, stessa conclusione a forza diversa:

- **[[attenzione-non-e-spiegazione]]** (Jain & Wallace): su 8 task di classificazione e 2
  di question answering, i pesi appresi sono **decorrelati** dalle misure gradienti di
  importanza delle feature, e si possono trovare **distribuzioni di attenzione diverse che
  danno previsioni equivalenti**. Se due spiegazioni diverse producono la stessa uscita,
  l'attenzione non e' *la* spiegazione.
- **[[attenzione-e-interpretabile]]** (Serrano & Smith): manipolando i pesi in modelli
  addestrati, l'ordinamento per **grandezza** del peso predice peggio l'effetto
  sull'uscita dell'ordinamento per **gradiente**. Attenzione rumoreggia, non garantisce.
- **[[attenzione-non-non-e-spiegazione]]** (Wiegreffe & Pinter): la controprova. Jain &
  Wallace non hanno addestrato un modello con le attenzioni avversarie, quindi quelle
  attenzioni non sono spiegazioni *plausibili* del modello; esistenza non implica
  esclusivita'. Il titolo e' un doppio negativo deliberato: la conclusione dipende da cosa
  si intende per spiegazione.

Poi la famiglia "come aggregare i pesi fra livelli":
[[flusso-di-attenzione]] (rollout e flow correlano meglio dei pesi grezzi con le misure
di ablazione), e le due famiglie di potatura [[sedici-teste-meglio-di-una]] ·
[[teste-specializzate]] dicono che molte teste sono **inerte**: toglierle non cambia il
comportamento. Conseguenza per noi: una mappa di attenzione puo' essere inerte anche
quando e' bella da guardare.

## Cosa significa per E22, detto chiaramente

E22 misurava la **massa di attenzione** che un insieme di token riceve. La letteratura
dice che quella massa e' una variabile debole: puo' cambiare molto senza cambiare la
previsione, e puo' restare identica mentre la previsione cambia. Quindi il risultato
"la quantita' di attenzione non spiega l'ASR" era **atteso**, e non era una cattiva
notizia per la domanda: era il metodo sbagliato per la domanda.

Cio' che E22 ha comunque stabilito, e che la letteratura *non* puo' dare:

1. Lo **strato fisico** dei pesi (10 globali su 28, finestra locale ±64 token, misurata
   sui pesi e non sulla documentazione).
2. Che alla distanza >64 token la massa dai livelli locali e' **esattamente zero**:
   l'informazione passa solo dai globali, e l'attacco regge comunque (E22b).
3. Che il vantaggio della coda e' una proprieta' dello **stato corto**, non della
   posizione.

Il primo punto e' architettura, gli altri due sono misure su questo modello. La domanda
"perche' funziona" vuole un metodo diverso: [[circuiti-e-interventi]].

## La regola operativa che resta

Un peso di attenzione e' un'**ipotesi** sul percorso dell'informazione, non una misura
del contributo. Va messo alla prova con un intervento (ablazione, patching) prima di
comparire in una nota come spiegazione. Nel progetto: qualsiasi mappa che finisce in
una nota deve venire con l'ablazione corrispondente, esattamente come ogni numero deve
venire con il run che lo produce.

## Note collegate
[[attribuzioni-e-sanity-check]] · [[E22-attenzione-laya]] · [[circuiti-e-interventi]]
