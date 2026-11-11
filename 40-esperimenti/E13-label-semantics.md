---
tags: [esperimento, etichettatura, diagnostico, non-submission]
stato: fatto
data: 2026-09-30
study_valid: false
---

# E13 — Le opzioni sono un canale di informazione?

## Ipotesi

Se rinominare le opzioni da `calendar_query` a `I7` costa accuracy, allora il modello non
classifica fra concetti: **legge una stringa**. Il canale delle opzioni sarebbe il punto
d'ingresso più economico in assoluto.

Ma le opzioni sono **backend**: toccarle sposta il goalpost, quindi questo non è una submission.
È una misura diagnostica per stabilire se il canale è informativo (→ [[superficie-attacco]],
"cosa NON è attaccabile").

## Setup

- checkpoint: `convaiinnovations/laya-typed-decisions` (`head_max_len=256`)
- dataset: MASSIVE 1.0 `it-IT`, split `test`, **N=1000**, i primi 1000 dei 2974
- K = 4, 8, 16: qui l'effetto è di decine di pp, quindi N=1000 basta per il claim. La replica
  manca comunque (vedi avvertenza 2).
- descrizioni **vuote** in tutti i braccia: `build_choices` scrive `criteria = {opzione: ""}`
- `seed=0`, `batch_size=16`, `chunk=32`, `subset_hash=5cabb1674460…` (identico ai 3 K: il subset
  non cambia con K, cambia solo il mix di opzioni)
- E13 non tocca lo stato, quindi `delta_tokens = 0` per costruzione: le opzioni rinominate sono
  l'unica variabile e l'accuracy si legge come canale, non come rumore
- comando: `uv run adv e13`
- output: `results/E13_label_semantics.json`

## Metodo

Tre braccia, **stesso stato, stesso K, stesso ordine delle opzioni**: cambia solo il testo.
La rinomina segue `sorted(labels)`, quindi `calendar_query` (indice 7 delle 59 label it-IT)
diventa `I7`.

| braccio | opzione | cosa isola |
|---|---|---|
| `real` | `calendar_query` | baseline |
| `short` | `I7` | il formato conta, o il significato conta? |
| `prefix` | `I7: calendar_query` | lunghezza del testo vs significato |

`prefix` è il braccio decisivo: se recupera quasi tutto di `real`, il modello legge la stringa
lunga e l'abbreviazione non aggiunge informazione. Se `short` regge, il formato non importa e il
canale è puramente posizionale.

Il **gold segue la rinomina**: altrimenti la domanda non avrebbe una risposta valida e l'accuracy
misurerebbe un dataset rotto. Ecco perché l'esperimento non è confrontabile con E03: la domanda
è diversa.

## Risultati

N=1000 per braccio, `n_decisions=1000`, una decisione per caso.

| K | 1/K (`chance`) | `real` | `short` (I7) | `prefix` (I7: calendar_query) |
|---|---|---|---|---|
| 4 | 0.25 | 0.4920 [0.461, 0.523] | **0.2270** [0.202, 0.254] | 0.5010 [0.470, 0.532] |
| 8 | 0.125 | 0.3750 [0.346, 0.405] | **0.0990** [0.082, 0.119] | 0.4010 [0.371, 0.432] |
| 16 | 0.0625 | 0.2940 [0.267, 0.323] | **0.0390** [0.029, 0.053] | 0.2780 [0.251, 0.307] |

I limiti storici della riga `real`, K=8, non erano campi dello store. L'audit
`results/provenance/historical_ci_audit.json` li ha ricostruiti dai 1.000 record del run
storico `2fd9b9f346f4daed`: 375 corretti, Wilson 95% [0.3455, 0.4054], quindi
[0.346, 0.405] dopo arrotondamento. Non sono intervalli della run corrente.

**Interi numeri di questa tabella sono storici**: vedi la sezione "Ricalcolata il 2026-10-05" in
coda al file. I punti stimati di questa tabella non combaciano con l'archivio corrente.

| K | drop `short` vs `real` | drop `prefix` vs `real` |
|---|---|---|
| 4 | **26.5 pp** | −0.9 pp |
| 8 | **27.6 pp** | −2.6 pp |
| 16 | **25.5 pp** | +1.6 pp |

 Formule (tutte ricalcolabili, nessuna riportata così nell'archivio):

- `drop_vs_real_pp = 100 × (accuracy_real − accuracy_braccio)`, con `round(..., 2)`, calcolato dal
  codice e riportato nell'archivio. 100 × (0.4920 − 0.2270) = 26.5; 100 × (0.3750 − 0.0990) = 27.6;
  100 × (0.2940 − 0.0390) = 25.5; 100 × (0.4920 − 0.5010) = −0.9; 100 × (0.3750 − 0.4010) = −2.6;
  100 × (0.2940 − 0.2780) = +1.6.
- **Segno**: `drop > 0` = il braccio sta peggio di `real`. Quindi `prefix` a K=4 e K=8 è *sopra*
  `real` (drop negativo) e a K=16 è 1.6 pp *sotto*.
- **I limiti di CI non sono nell'archivio**: Wilson 95% su N=1000, cioè
  `(p ± z²/2n ∓ z√(p(1−p)/n + z²/4n²)) / (1 + z²/n)` con `z=1.96`. Ricalcolati qui da `accuracy`
  e `n_decisions`: per p=0.492 dà [0.461, 0.523], per p=0.039 dà [0.029, 0.053], e così via per
  le altre sette celle. È la ragione per cui `tools/check_note_numbers.py` li segnala.
- `1/K` = campo `chance` dell'archivio, `round(1/k, 4)`.

Provenienza: 9 run separati (3 braccia × 3 K) in `results/E13_label_semantics/runs/<run_id>/`,
`config.json` + `preds.jsonl` (1000 righe) per ognuno.

## I due risultati

> **Due avvertenze prima di confrontare questi numeri con quelli di E03.**
>
> 1. **N diverso.** Qui N=1000 (i primi 1000 casi), in E03 N=2974. Lo stesso K=4 sullo stesso
>    checkpoint dà qui `real` = **0.4920** e in E03 **0.6513** (`accuracy_clean`, K=4, in
>    `results/E03_option_injection__laya__massive-it.json`, N=2974: numero di E03, non di questo
>    archivio). Il sottoinsieme è più difficile. I delta *interni* a questa tabella valgono, i
>    valori assoluti non sono interscambiabili con le tabelle di E03.
> 2. **Un solo seed** (`SEED=0`), come tutta la campagna. Con il position bias che la riga `real`
>    mostra sotto, la varianza da disegno delle opzioni non è trascurabile e non è mai stata
>    stimata. Da ora E03 ed E13 accettano `--seed`; la replica va girata.

### 1. Il canale è informativo, e conta ~26pp

Rinominare le opzioni costa **25-28 pp a ogni K**, costante. Il claim precedente in `todo-list.md`
e [[superficie-attacco]] diceva "~22pp a K=4 da `calendar_query` a `I7`": nessuna run lo
sosteneva, il numero misurato è **26.5 pp** e la tabella qui sopra lo sostiene ora.

Il formato non c'entra: `prefix` è 4 caratteri più lungo di `calendar_query` (18 contro 14), e
`short` ne ha 2. Eppure `prefix` è indistinguibile da `real` (drop −0.9, −2.6, +1.6 pp, tutti
entro il rumore). Quindi non è che il modello preferisca nomi brevi o lunghi: **legge il testo e
lo usa**, e accorciarlo costa 25-28 pp.

### 2. Con etichette anonime il modello è peggio che casuale

Il dato più forte, e non era annotato da nessuna parte. A K=8 e K=16 l'accuracy di `short` è
**sotto il caso** con il CI che non lo tocca:

| K | `short` | 1/K | z | verdetto |
|---|---|---|---|---|
| 8 | 0.0990 [0.082, 0.119] | 0.125 | −2.49 | sotto il caso |
| 16 | 0.0390 [0.029, 0.053] | 0.0625 | −3.07 | sotto il caso |
| 4 | 0.2270 [0.202, 0.254] | 0.25 | −1.68 | sul caso |

`z = (accuracy − 1/K) / √((1/K)(1 − 1/K)/N)`, con N=1000: a K=16 vale
(0.0390 − 0.0625) / √((1/16)(1 − 1/16)/1000) = **−3.07**, cioè 3 sigma sotto il caso. A K=4 il
CI contiene 1/K, e z = −1.68 non lo distingue: **sul caso**, non sotto.

**Il null è giusto, il meccanismo no.** `1/K` è il null corretto e la verifica si fa:
`build_choices` mescola le opzioni (`rng.shuffle`), quindi il gold è uniforme sulle K opzioni per
costruzione, e un modello indipendente dal gold azzerecherebbe proprio `1/K = 0.0625`. Ma
"informazione spuria" descrive un meccanismo che **non è quello osservato**.

Il meccanismo vero è un **collasso sulla posizione finale**. Ricostruendo l'ordine delle opzioni
dai run salvati (`preds.jsonl`, l'ordine delle chiavi di `probs` è l'ordine delle opzioni), a K=16
il braccio `short` mette **235 delle 1000 predizioni sull'ultima opzione** (indice 15), contro 93
sulla prima. L'accuracy per posizione del gold:

| posizione del gold | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `real` | .29 | .30 | .23 | .38 | .39 | .22 | .35 | .33 | .34 | .29 | .35 | .28 | .33 | .23 | .24 | .19 |
| `short` | .05 | .13 | .05 | .04 | **.00** | .03 | .01 | .02 | **.00** | **.00** | .03 | .02 | **.00** | .02 | **.00** | **.19** |
| casi (denominatore) | 58 | 61 | 74 | 72 | 57 | 64 | 69 | 58 | 61 | 56 | 65 | 60 | 55 | 66 | 55 | 69 |

Il denominatore per posizione è il numero di casi con il gold lì, 55-74: ogni cella è quindi
pochi casi, quindi va letta come indicatore di dove collassa il modello, non come stima puntuale.
La tabella è ricalcolata dai `preds.jsonl` dei due run `03553d7fa78fc8b8` (`short`) e
`c2457df3e6b8620d` (`real`), quindi **non è nell'archivio dichiarato** e `check_note_numbers.py`
non la verifica.

Il modello non è "tratto in inganno" dalle opzioni anonime: quando non sa più distinguerle
**collassa sull'ultima**, che è il picco di segnale residuo (indice 15: 13/69 = 0.19, **3.0× il
caso**). Le uniche due posizioni sopra il caso sono la 1 (8/61 = 0.13, 2.1× il caso) e l'ultima.
Sotto il caso, quindi, è un **fallback posizionale dell'architettura**, non informazione spuria.
La differenza non è accademica: se fosse informazione spuria un attaccante potrebbe sfruttarla,
un fallback di posizione no.

Nota l'effetto crescente in K: più opzioni anonime, più il collasso. `short` sta a 0.91× il caso
a K=4 (0.2270 / 0.25), 0.79× a K=8 (0.0990 / 0.125) e 0.62× a K=16 (0.0390 / 0.0625). E il
braccio `real` mostra lo stesso effetto in forma lieve ma **attribuibile a position bias reale**:
a K=16 l'accuracy va da **0.19** (indice 15, 13/69) a **0.39** (indice 4, 22/57), cioè da 3.0× il
caso a 6.2× il caso.

## Verdetto

- **Le etichette sono un canale di informazione reale**, non un'etichetta cosmetica: 25-28 pp di
  accuracy dipendono dal testo delle opzioni (26.5, 27.6, 25.5 pp a K=4, 8, 16).
- **Ma non è attaccabile.** Le opzioni sono backend, come gold e `instructions`: cambiarle sposta
  il goalpost e invalida il confronto con E01. Il valore che questa misura produce è capire *perché*
  E03 funziona, non trovare un attacco.
- Il canale che l'attaccante controlla resta **lo stato** ([[E03-option-injection]]), ed è l'unico
  su cui si può agire senza toccare il backend. Questo esperimento spiega perché l'injection
  funziona: se il modello pesa così tanto il testo delle opzioni, allora nominarne una nello stato
  è l'operazione più economica per spostarlo, e non serve nemmeno toccare le opzioni.

## Cosa resta

- Il braccio `prefix` a K=16 è l'unico con drop positivo (+1.6 pp = 0.2940 − 0.2780, cioè 1.6 pp
  sotto `real`). Con N=1000 il rumore è ~±3 pp: la semi-ampiezza del CI 95% di una accuracy è
  1.96 × √(p(1−p)/N) = 0.028 a p=0.294, N=1000. Quindi non è distinguibile, e il caso non va
  chiuso su questo numero.
- Manca la replica: `--seed` esiste e funziona, la curva è single seed.

## Note collegate

[[superficie-attacco]] · [[E03-option-injection]] · [[token-budget-head]] · [[metriche-attacco]] ·  · [[protocollo-dataset-valido]]

## Output

- `results/E13_label_semantics.json`
- `results/E13_label_semantics/runs/<run_id>/` (3 braccia × 3 K = 9 run separati)
## Ricalcolata il 2026-10-05: i numeri pubblicati erano di un run superato

Rilanciando `adv e13` a N=1000 con il codice di oggi i numeri **cambiano**, e non di poco:
`max |Δp|` fra il run vecchio (`c2457df3e6b8620d`, `real`) e quello nuovo
(`8717a503bdb0d2b2`, stesso `subset_hash`, stesso `chunk`, stesso `batch_size`) è **0.5826**, e
**109 label su 1000** sono diverse. Non è rumore bf16: è un cambiamento di comportamento del
codice fra le due esecuzioni.

| K | chance | real | short | prefix | drop short |
|---|---|---|---|---|---|
| 4 | 0.2500 | 0.4950 | 0.2200 | 0.5190 | 27.5pp |
| 8 | 0.1250 | 0.3860 | 0.0950 | 0.4110 | 29.1pp |
| 16 | 0.0625 | 0.3060 | 0.0380 | 0.2880 | 26.8pp |

Quindi le tabelle qui sopra (0.0990 a K=8, 0.0390 a K=16, drop 25-28pp) descrivono il run
vecchio e vanno lette come **storiche**. La conclusione non cambia: le etichette restano un
canale di informazione reale e l'accuracy anonima resta **sotto il caso** a ogni K. Cambiano le
cifre, e il paper porta adesso quelle nuove.

La causa del buco è `run_identity()`: quando è stato aggiunto al config, il `run_id` è cambiato
perché è lo sha del config, quindi i run vecchi **non vengono più ripresi**. Ogni rilancio è un
run nuovo, e i risultati pubblicati restano invecchiati finché nessuno rilancia. È la stessa
lezione del 2026-10-01 su E01, ma qui la conseguenza è che i numeri erano davvero diversi e
nessuno se n'era accorto.
