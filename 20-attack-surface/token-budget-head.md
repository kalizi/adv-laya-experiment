---
tags: [attacco, strutturale, token-budget]
---

# Token budget delle opzioni

**Leva #1.** Le opzioni di un `choice` condividono un budget fisso: affamarle è un attacco.

## Meccanismo
`head_max_len` copre **question + criteria + tutte le opzioni**:

| checkpoint | head_max_len | note |
|---|---|---|
| `laya` (English) | 192 | |
| `laya-multilingual` | 256 | |
| `laya-typed-decisions` | 256 | ereditato |

Con K opzioni: circa `(head_max_len − overhead) / K` token **per opzione**.

### Dato misurato (Banking77)
77 opzioni → **~3–4 token per label** → accuracy **0.425** vs Jev **0.870** (che supporta 255 opzioni out-of-the-box).
Consiglio ufficiale di Convai: **mantenere `choice` sotto ~20 opzioni**.

## Muro misurato: la troncatura silenziosa (2026-09-29)
Superato `head_max_len`, **laya non segnala errore**: mantiene i marker delle opzioni ma
tronca il loro testo. Quindi l'accuracy crolla per un motivo che non è il modello, e il
risultato sembra un attacco riuscito. Verificato decodificando la sequenza tokenizzata e
contando i nomi effettivamente presenti (MASSIVE it-IT, descrizioni vuote, caso peggiore
= intent piu' lunghi):

| K | head token | opzioni visibili | gold visibile |
|---|---|---|---|
| 40 | 240 | 500/500 | 500/500 |
| 42 | 255 | 500/500 | 500/500 |
| 44 | 255 | 196/200 | 196/200 |
| 46 | 255 | 0/200 | 140/200 |
| 60 | 255 | 0/200 | 0/200 |

Da K=46 in su scompare **anche il gold**: l'accuracy misurata non dice piu' niente sul
modello. Nessuna curva acc-vs-K è pubblicabile sopra K=42 su it-IT.

**Contromisura nel codice**: `layaadv.data.loaders.check_head_budget` verifica su un
campione che tutte le opzioni e il gold sopravvivano alla tokenizzazione e **aborta con
errore esplicito** altrimenti. Serve perche' un loader sbagliato non fallisce: restituisce
metriche false.

## Due modi per sprecare budget, misurati
1. **Descrizione uguale al label**: laya rende l'opzione come `"label: descrizione"`, quindi
   una descrizione identica al label paga il testo due volte e dimezza le opzioni che
   entrano (limite da K=20 a K=42 solo correggendo questo).
2. **Etichette lunghe**: i token per opzione variano da 3 a 8 sugli intent di MASSIVE.

## Il testo delle opzioni pesa, ma non per la sua lunghezza
Rinominare le opzioni da `calendar_query` a `I7` costa **25-28pp** ([[E13-label-semantics]]),
ma *aggiungere* il nome lungo accanto al corto (`I7: calendar_query`, che paga budget) non
costa nulla: indistinguibile dalle etichette reali entro il rumore.

Quindi il budget conta due volte, e solo una è sfruttabile: le opzioni lunghe costano
tokens, ma i tokens sono sprecati **solo** se il testo non porta informazione che il modello
usa. Accorciare un'etichetta non è un attacco (cambia il backend), è una misura: serve a
sapere che il canale delle opzioni è informativo, non cosmetico.

## Due direzioni d'attacco

### A. Affamare le opzioni corrette (black box)
L'attaccante controlla lo **stato**, non i criteria. Ma se il deployment ha criteri lunghi e descrittivi (o un payload JSON ampio), il budget per lo stato e per le opzioni è già saturo:
1. **Stato gonfiato**: riempire lo stato fino al budget → truncation + embedding in coda degrada il match con le opzioni → [[truncation-stato]]
2. Opzioni numerose per design (es. routing tra 50 dipartimenti): solo agire sullo stato per introdurre rumore che colpisca le opzioni affamate

### B. Definire un workload bersaglio (gray box / dataset)
Per la study: scegliere/creando un dataset con **K ≥ 30 opzioni** (intent routing, Banking77-like) è lecito? Sì se il dataset è dichiarato come parte del setup → [[protocollo-dataset-valido]].
- Baseline da misurare prima: [[E01-baseline]] con K variabile
- Curve attese: acc vs K (ripreso dal benchmark: Banking77 0.425 a 77 label)

### C. Variante runtime (solo gray box)
`agent.cfg["head_max_len"] = 512` alza il budget: serve per **misurare** la curva, non per attaccare.

## Metriche per questo attacco
- `acc(K)` per K = 4, 8, 16, 32, 64, 77
- `overconf(K)`: la confidence cala o resta alta? (se resta alta → molto più grave, vedi )
- Token effettivi per opzione (contare con il tokenizer del checkpoint)

## Da misurare
- [ ] Curva acc vs K sul proprio hardware
- [ ] Curva confidence vs K
- [ ] Effetto di `head_max_len` (192/256/512) sulla stessa curva
- [ ] Cross: K fisso alto + stato gonfiato → interazione con [[truncation-stato]]

## Note collegate
[[architettura-laya]] · [[truncation-stato]] · [[token-budget-head]] · [[esperimenti]]
