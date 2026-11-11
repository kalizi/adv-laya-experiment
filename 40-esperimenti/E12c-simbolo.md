---
tags: [esperimento, superficie, budget, rumore, negativo]
stato: fatto
data: 2026-09-30
study_valid: true
---

# E12c — aggiungere un simbolo invece di sostituire una parola

## Ipotesi
Dopo che [[E12b-sinonimi-rari]] ha mostrato che sostituire parole non scala, la perturbazione più
economica rimasta è **un carattere che non cambia significato**. Se il modello è sensibile a
qualcosa di più grossolano del significato, deve esserlo anche a questo. E non costa un
vocabolario.

## Setup
- checkpoint `convaiinnovations/laya-typed-decisions`, N=30, baseline 0.7533, 113 decisioni
  corrette al pulito
- iniezione di caratteri in posizioni casuali **dentro il corpo** dello stato, 4 livelli
  (1, 3, 10, 30) e due alfabeti: simboli `# * ~ | @ + = ^ % $` e spazi
- controlli: singolo spazio, spazio in due punti, e JSON rotto (rimossa una virgola)
- tracciato il costo in token con `engine.count_tokens`, per separare l'effetto del
  carattere dall'effetto budget di [[E04-truncation]]

## Il controllo che decide tutto
Prima di interpretare: **lo stato identico allo stato pulito sposta le previsioni?**

| confronto | label diverse su 150 | Δ confidenza |
|---|---|---|
| clean contro clean | **0** | **0.000000** |
| clean chunk 32 contro chunk 1 | **0** | — |

Il modello è **perfettamente deterministico** su questo setup: nemmeno una decisione su 150
si muove, e la confidenza è identica bit per bit. Quindi ogni differenza che segue è un
effetto reale della perturbazione, non rumore numerico. Senza questo controllo i numeri
sarebbero non interpretabili.

## Risultati

### Simbolo contro spazio, stessa procedura
| livello | Δtoken | label cambiate | flip | ASR | confidenza sul torto | OER |
|---|---|---|---|---|---|---|
| simbolo ×1 | 1.8 | 1/90 | 0 | 0.0000 | — | 0 |
| simbolo ×3 | 5.2 | 5/90 | 2 | 0.0177 | 0.271 | 0 |
| simbolo ×10 | 16.5 | 10/90 | 5 | 0.0442 | 0.305 | 0 |
| simbolo ×30 | 48.9 | 30/90 | 19 | **0.1681** | 0.221 | 0 |
| spazio ×1 | 1.4 | 3/90 | 1 | 0.0088 | 0.286 | 0 |
| spazio ×3 | 3.6 | 4/90 | 2 | 0.0177 | 0.277 | 0 |
| spazio ×10 | 11.3 | 10/90 | 4 | 0.0354 | 0.288 | 0 |
| spazio ×30 | 33.2 | 32/90 | 22 | **0.1947** | 0.182 | 0 |

### Il simbolo è un caso particolare del budget
L'efficienza per token speso, cioè quanto ASR si compra con ogni token di budget:

| livello | simbolo ASR/token | spazio ASR/token |
|---|---|---|
| ×1 | 0.00000 | 0.00629 |
| ×3 | 0.00340 | 0.00492 |
| ×10 | 0.00268 | 0.00313 |
| ×30 | 0.00344 | 0.00586 |
| **media** | **0.00238** | **0.00505** |

L'ASR è una funzione quasi lineare dei **token spesi**, e lo slope non dipende dal carattere.
Uno spazio è **2.1×** più efficace per token consumato. Che cosa sia il carattere non conta:
conta quanto budget mangia.

## Il caso più istruttivo: rompere il JSON non fa niente
Rimuovendo una virgola lo stato smette di essere JSON valido in **30 casi su 30**, e le
previsioni non cambiano **nemmeno una volta su 150**.

| braccio | JSON valido | label cambiate | flip |
|---|---|---|---|
| simbolo dentro un valore | 30/30 | 4/150 | 1 |
| **JSON rotto** | **0/30** | **0/150** | **0** |
| spazio singolo (controllo) | 30/30 | 2/150 | 1 |
| due spazi (controllo) | 30/30 | 3/150 | 0 |

Il modello **non usa la struttura**: si può cancellare una virgola dal JSON e la decisione è
identica. Non c'è un attacco "di formato" da fare, perché il formato non viene letto.

## Riallineato allo store (2026-10-01)

L'esperimento è ora dentro `adv e12-surface`, con archivio, e la dose dichiarata è **50% delle
posizioni di contenuto** (spaCy POS), non "30 simboli". A quella dose l'ASR è **0.0177**, con
controllo a spazio 0.0088: il simbolo non aggiunge niente allo spazio che occupa lo stesso posto.
I numeri storici in tabella (0.1681 a 30 simboli) vengono da una dose diversa e non hanno uno
store: sono la stessa osservazione, non una contraddizione, ma la riga citabile è 0.0177.

## Verdetto
- **Il simbolo funziona, ma non è una superficie nuova**: è [[E04-truncation]] con un filler
  peggiore. Stesso meccanismo (budget consumato), risultato inferiore: 30 spazi danno ASR
  0.1947 contro 0.1681 dei 30 simboli, e costano 33 token invece di 49.
- **Nessun vocabolario serve**: se si volesse degradare il segnale, la carattere più economico
  è lo spazio. È un risultato scomodo dal punto di vista del gioco, ma è la misura.
- **La confidenza non sale mai**: OER 0 a ogni livello, confidenza sul torto 0.18-0.30, in
  tendenza *decrescente* con la dose (0.305 a ×10, 0.221 a ×30). Non sono errori convinti:
  è un modello che peggiora con il rumore, non che viene ingannato. Nessuna configurazione
  soddisfa .
- **Il modello è deterministico** su questo setup (0/150, Δconf 0.000000): ogni misura di
  questo progetto è priva di rumore numerico, il che rende i confronti esatti.

## Cosa resta
- Se l'obiettivo è degradare il segnale, il perimetro da esplorare è la **dose** di budget e
  non il carattere: è già la leva di E04, e lì c'è già il numero (ASR 0.553 a N=400).
- Se l'obiettivo è un attacco **convinto**, nessuna quantità di rumore lo dà. Serve un
  segnale, e l'unico che funziona resta nominare un'opzione ([[E03-option-injection]]).

## Note collegate
[[E04-truncation]] · [[E03-option-injection]] · [[E12b-sinonimi-rari]] ·  · [[token-budget-head]] · [[esperimenti]]

## Output
Nessun file in `results/`: la sonda è uno script one-off, non un esperimento con protocollo
(nessun ciclo, nessuna selezione). I numeri sono nelle tabelle sopra e si riproducono:
```bash
uv run python tools/noise_floor.py          # il determinismo: 0/150, Δconf 0
uv run python tools/e12c_symbol_probe.py    # 4 braccia, incluso JSON rotto
uv run python tools/e12c_symbol_scale.py    # curva cumulative 1/3/10/30
```
