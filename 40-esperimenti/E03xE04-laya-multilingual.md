---
tags: [esperimento, composizione, e03, e04, filler, cross-checkpoint]
stato: fatto
data: 2026-09-30
study_valid: true
---

# E03xE04 su laya-multilingual — il filler non ripara il guardrail

## Perché
Su `typed-decisions` la composizione dava un risultato che sembrava una **mitigazione**: il
filler (E04) abbassava l'OER sotto attacco da 0.90 a 0.00 quando l'iniezione in coda veniva
mangiata. Su `laya-multilingual` l'attacco è più forte (E03 da solo: OER 0.995), quindi la
domanda è se la difesa regge. È un esperimento di difesa, non di offesa: il punto è capire se
un difensore può usare il filler per spegnere l'errore convinto.

## Setup
`laya-multilingual`, `massive-it`, iniezione in coda (`append`), N=1000 per riga, filler
(lorem ipsum neutro) 0 / 400 / 600 token davanti allo stato. `questions`, `criteria` e gold
byte-identici: solo lo stato cambia. Stesso protocollo di [[E03xE04-composizione]].

## Risultati

| K | filler | acc pulito | acc attaccato | ASR | **OER@0.75** | conf sul torto |
|---|---|---|---|---|---|---|
| 4 | 0 | 0.772 | 0.014 | 0.9819 | **0.907** | 0.9432 |
| 16 | 0 | 0.538 | 0.001 | 0.9981 | **0.995** | 0.9928 |
| 4 | 400 | 0.772 | 0.007 | 0.9922 | **0.942** | 0.9561 |
| 16 | 400 | 0.538 | 0.004 | 0.9963 | **0.943** | 0.9574 |
| 4 | 600 | 0.772 | 0.013 | 0.9870 | 0.875 | 0.9160 |
| 16 | 600 | 0.538 | 0.140 | 0.8327 | **0.050** | 0.2233 |

## Il finding, e perché è diverso da typed-decisions

**Il filler NON è una mitigazione su questo checkpoint, almeno fino a 400 token.**
L'OER resta altissimo (0.907 → 0.942 → 0.943): il guardrail a 0.75 continua a non intercettare
l'errore convinto, e la confidenza sul torto resta sopra 0.94. Anzi, a 400 token l'attacco è
**più forte che a 0** (ASR 0.9922 e 0.9963 contro 0.9819 e 0.9981, OER 0.942 contro 0.907 a K=4).

Confronto diretto con lo stesso esperimento su `typed-decisions` (K=16, OER@0.75):

| filler | typed-decisions | laya-multilingual |
|---|---|---|
| 0 | 0.902 | **0.995** |
| 400 | 0.545 | **0.943** |
| 600 | 0.000 | 0.050 |

Su `typed-decisions` 400 token di filler dimezzavano già l'OER (0.90 → 0.55) e 600 lo
azzeravano. Su `laya-multilingual` 400 non fa quasi nulla (0.995 → 0.943) e serve il filler
massimo, a 600, che però **mangia anche il gold**: a K=16 l'accuracy attaccata risale a 0.140
(il filler ha reso illeggibile anche la risposta vera) e l'ASR scende a 0.8327.

## Interpretazione
Ci sono due regimi, e il confine è il **budget della coda**:
- **Finché il filler non mangia il gold** (0 e 400 token): l'iniezione resta leggibile, l'ASR
  resta 0.99, l'OER resta 0.94-0.99. Il difensore ha aggiunto rumore e **non ha ottenuto
  niente**: l'errore convinto resta convinto.
- **Quando il filler mangia il gold** (600 a K=16): l'accuracy risale, l'ASR scende a 0.83 e
  l'OER crolla a 0.05. Ma a quel punto **anche la difesa ha distrutto il compito**, non lo ha
  protetto: il modello sbaglia per incapacità di leggere, non perché sia stato contrastato.

Questo è lo stesso meccanismo di [[E03xE04-composizione]], ma con la soglia spostata: su
`laya-multilingual` il budget di testa e la brevità dello stato rendono il filler da 400
**non sufficiente a mangiare l'iniezione in coda**, quindi non c'è la finestra in cui il filler
protegge senza distruggere. Su `typed-decisions` quella finestra esisteva e il filler agiva
come guardia.

## Verdetto per la difesa
- **Non c'è una dose di filler che spegne l'errore convinto senza anche rompere il compito** su
  questo checkpoint. O il rumore è troppo poco (OER resta 0.94) o è troppo (il gold viene
  mangiato, l'ASR e l'accuracy crollano insieme).
- Questo innalza il costo della difesa rispetto a `typed-decisions`: lì il filler era
  un'arma a doppio taglio che almeno al 600 spegneva l'attacco. Qui al 600 spegne anche il
  target, quindi un difensore che gonfiasse lo stato per proteggersi romperebbe il proprio
  servizio.
- Il risultato **non è "E03 è più forte su multilingual"**, che era già noto. È che **la
  mitigazione osservata sull'altro checkpoint non si trasferisce**, il che la rende molto meno
  promettente come contromisura.

## Dove mettere il filler: il margine non esiste
Il `--pad` di E03 usa `push_out_state`, che **antepone** il filler, quindi l'esperimento qui
sopra ha gia' misurato il filler in **testa**. Ho verificato con `probe_sequence` se il filler
**in coda** offrisse al difensore una leva piu' economica (spostare il marcatore fuori budget
senza consumare la testa). **Non esiste.**

| dose | marcatore visibile, filler in **testa** | filler in **coda** |
|---|---|---|
| 0 | 100% | 100% |
| 200 | 100% | 100% |
| 400 | 98.7% | 100% |
| 600 | 55.1% | 100% |
| 800 | **0%** | **100%** |

Il filler in coda non spegne mai il marcatore, a nessuna dose. Il motivo è geometrico e non
è una questione di budget: `build_sequence` tronca la **coda destra** (`truncate_left=False`),
quindi il filler appeso in coda è proprio lui la parte che viene tagliata, e il marcatore,
che sta **prima** del filler, resta sempre dentro. Non esiste una dose che lo spenga perché
non è raggiungibile dal taglio.

Il filler-in-testa è quindi gia' l'unica posizione che funziona, ed è quella dei due
regimi sopra: non basta a 400 e a 600 mangia il gold. **Non c'è un margine di difesa
inesplorato in questa direzione.** (Il mio errore iniziale: avevo proposto il filler in coda
ragionando che spostasse il marcatore fuori budget; è l'esatto contrario, perche' il taglio
punta a destra.)

## Cosa resta
- Nessun margine sul lato filler: posizione e dose sono chiuse, e il confine è il budget della
  coda. L'unica direzione aperta è fuori dal filler, cioe' **filtri a monte** (normalizzazione,
  rilevamento di istruzioni nello stato), che non sono un attacco ma una difesa di pipeline.
- Se il difensore non puo' agire sullo stato, resta il **backend**: le opzioni sono un canale di
  informazione ([[E13-label-semantics]]), ma sono backend e non controllabili dall'attaccante.

## Note collegate
[[E03-option-injection]] · [[E03xE04-composizione]] · [[E04-truncation]] · [[E01-baseline]] ·
 · [[token-budget-head]] · [[metriche-attacco]] · [[superficie-attacco]] · [[esperimenti]]

## Output
- `results/E03xE04__laya-multilingual__pad400.json` (scope durevole della riga filler 400)
- le righe 0 e 600 si riottengono con:
  ```bash
  uv run adv e03 --dataset massive-it --k 4 --k 16 --template append --limit 1000 \
      --pad {0,600} --checkpoint convaiinnovations/laya-multilingual --force-new
  ```
  (il file per scope viene sovrascritto a ogni run: va salvato dopo ciascuno)
