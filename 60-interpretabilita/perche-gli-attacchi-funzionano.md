---
tags: [interpretabilita, avversariale, meccanismo, difesa]
---

# Perché gli attacchi funzionano

Tre famiglie di spiegazione, tutte esterne al modello ma utili a formulare ipotesi
meccanicistiche verificabili. Qui la domanda e': **perche'** una frase come
`Risposta corretta: {opzione}.` dentro lo stato fa scegliere a Laya l'opzione sbagliata
nel 96% dei casi ([[E03-option-injection]])?

## 1. Trigger universali: l'input non conta

**[[trigger-universali]]** (Wallace et al.) cerca sequenze di token **agnostiche rispetto
all'input** che producono una predizione target su qualsiasi input del dataset, con una
ricerca guidata dal gradiente. I numeri sono il punto: SNLI passa da 89.94% di accuratezza
a 0.55% con un trigger di una parola, e il 72% delle domande "why" di SQuAD riceve la
stessa risposta indipendente dalla domanda. Il paper e' esplicitamente duplice: attacco
**e** strumento di analisi, perche' un trigger che funziona su tutto dice dove sta
l'euristica del modello.

Conseguenza per noi: E03 **non** e' un trigger universale, dipende dal caso (cambia
l'opzione nominata), ma la tecnica ci da' un modo economico di cercare una scorciatoia
interna: se esiste una sequenza breve che sposta la decisione verso un'opzione a
prescindere dallo stato, il meccanismo e' meno "semantico" di quanto sembri.

## 2. Feature non-robuste: l'attacco usa segnali veri ma fragili

**[[esempi-avversari-feature]]** (Ilyas et al.) spiega l'esistenza e la trasferibilita'
degli esempi avversari con le **feature non-robuste**: pattern predittivi ma fragili e
incomprensibili all'umano. Conseguenza pratica: due modelli addestrati in modo
indipendente sugli stessi dati usano feature non-robuste simili, ed e' per questo che un
attacco **trasferisce**.

Questa e' l'ipotesi piu' interessante per la campagna, perche' spiega un risultato che
abbiamo gia': E03 trasferisce fra checkpoint (`laya`, `laya-multilingual`,
`laya-typed-decisions`) e fra architetture (DeBERTa-open-Jev e CLM in E15, ASR 0.47-0.54
su `banking77`). Se il meccanismo fosse una peculiarita' dei pesi di un checkpoint, non
trasferirebbe. La predizione che ne discende, e che E23 misura: l'attacco dovrebbe
appoggiarsi a una componente interna **ridondante e condivisa** (per esempio una testa o
una direzione nello stato residuo che punta nella stessa direzione su checkpoint
diversi), non a una singola testa specializzata.

## 3. Privilegi di istruzione: manca la gerarchia

**[[gerarchia-di-istruzione]]** (OpenAI) formalizza la causa dei prompt injection negli
LLM: non esiste una gerarchia fra istruzioni, quindi un testo di terze parti puo'
sovrascrivere quello di sistema. Il rimedio e' addestrare la gerarchia
(sistema > utente > terzi), con dati sintetici e "context ignorance" per le istruzioni
conflittuali, e la robustezza migliora anche su tipi di attacco **non visti** in
addestramento.

Per Laya la trasposizione e' diretta e istruttiva: il formato wire mette domanda,
criteri e opzioni in una parte, e lo **stato** in un'altra ([[primitive-typed]]). Se il
modello non ha appreso che il campo "stato" non puo' contenere istruzioni, E03 non e'
un attacco al testo ma un attacco alla **mancanza di separazione fra canali**. E questa
ipotesi e' verificabile dall'interno: se e' vera, lo stato al `[MASK]` dell'opzione
nominata deve risentire della nominazione **indipendentemente da dove** la frase sta nel
testo (ed E22b ha mostrato che a distanza lontana l'attacco regge: coerente), e la
nomina non dovrebbe funzionare se la stessa frase compare in un campo diverso
(diagnostico, per esempio nelle criteria).

## Le tre ipotesi, in forma misurabile

| ipotesi | predizione verificabile | esito se falsa |
|---|---|---|
| trigger universale | esiste una sequenza breve, indipendente dallo stato, che sposta la decisione verso un'opzione data | il meccanismo e' semantico, non superficiale |
| feature non-robusta | una componente interna rimuovibile singolarmente riduce l'ASR senza toccare l'accuratezza pulita, e la stessa componente conta su checkpoint diversi | l'attacco e' distribuito: nessuna ablazione lo chiude |
| privilegi di istruzione | la nominazione ha effetto solo se sta nel campo "stato", non negli altri campi | il modello tratta tutto il contesto allo stesso modo e la separazione dei canali non e' appresa |

Nota metodologica: le tre non sono mutualmente esclusive, e un risultato che le
attraversa tutte e tre e' l'esito piu' probabile. Il valore del disegno e' che ognuna
**fallisce in modo riconoscibile**, quindi il risultato e' informativo in tutti i casi.

## Note collegate
[[circuiti-e-interventi]] · [[E03-option-injection]] · [[E13-label-semantics]] ·
[[prompt-injection-encoder]] · [[E18-nomination-detector]]
