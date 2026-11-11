---
tags: [esperimento, superficie, character-level, multicarattere, bidi, negativo]
stato: fatto
data: 2026-09-30
study_valid: true
---

# E12n — multicarattere e bidi override

Due domande diverse, chiuse insieme perché entrambe chiudono un buco aperto da
[[E12f-omoglifi-livelli]].

## 1. I multicarattere allargano la copertura?

La domanda era aggirare il limite di copertura di UTS #39, cioè raggiungere le lettere senza
confusibile. **La risposta è nell'utenza del file, e va detta prima di misurare:**

- UTS #39 **non contiene nessuna sequenza multi-carattere verso una lettera latina**. Zero
  righe. L'unica voce inversa è `m` → `rn`, che è una scomposizione latino-latina e non serve
  al cross-script.
- `f`, `g`, `k`, `m` **non hanno confusibile in nessuno script**. Solo `n`, `t`, `z` ne hanno
  uno, ma è same-script (fullwidth o matematico), quindi costa 3 byte.

Il vincolo non è l'arità della sostituzione, è che **il carattere non esiste**. I
multicarattere non possono raggiungere quelle quattro lettere.

### Cosa possono fare, e cosa misura
Una sostituzione multi-carattere 2→1 è una **contrazione**: toglie token invece di
aggiungerne. E la legge di tutta la linea è che funziona solo ciò che aggiunge token.

| tabella dichiarata (non normativa) | Δtoken | ASR | drop | collasso |
|---|---|---|---|---|
| `rn→м`, `cl→d`, altre 3 | **+2.3** | **0.0000** | −2.0pp | 12 → 12 |

Un flip su trenta casi (ASR 0.035, N=30): inerte, ma non letteralmente zero. Dei 5 digrammi dichiarati, solo 2 compaiono davvero negli stati (`rn` 30 volte,
`cl` 11; `vv`, `nn`, `ii` zero volte), quindi la copertura reale era minima comunque.

## 2. Il bidi override

U+202E (RLO) e U+202D (LRO) forzano la direzione di **rendering** dei caratteri seguenti:
l'umano che guarda lo schermo li vede in ordine inverso, i byte sul disco no. È l'attacco
classico del nome file, e in letteratura dà 52.8% (arXiv 2508.14070).

**La previsione era che non avrebbe funzionato**, e il motivo è noto: l'inversione sta nel
rendering, e un encoder non disegna niente. Il codepoint letto è esattamente quello che c'è. Su
un encoder l'attacco aggiunge solo caratteri invisibili, come i zero-width.

| braccio | Δtoken | drop | ASR | collasso |
|---|---|---|---|---|
| RLO 100% | +79.5 | −1.3pp | 0.0885 | 12 → 12 |
| LRO 100% | +79.5 | −2.7pp | 0.0708 | 12 → 12 |
| LRI 100% | +79.5 | 0.0pp | 0.0885 | 12 → 12 |

Letteratura 52.8%, da noi 0.09 con drop negativo. L'attacco è pensato per ingannare un
**moderatore umano** che approva un commento, non un modello. Su un encoder non ha bersaglio.

## 3. Il risultato positivo, che non era dove cercavo

Estendendo la tabella **a singolo carattere** con 3 sostituti *dichiarati* per le lettere
scoperte (`m→т` cirillico, `n→п`, `k→ķ`):

| tabella | lettere | Δtoken | ASR | drop | ASR/token |
|---|---|---|---|---|---|
| UTS #39 sola | 19 | +175.6 | 0.5841 | 38.0pp | 0.00333 |
| **+ m, n, k** | **22** | +189.6 | **0.6726** | **46.7pp** | **0.00355** |

Nove punti di ASR in più a costo quasi uguale. Quindi la copertura conta, ma **solo delle
lettere frequenti nel testo**: `m` e `n` lo sono, `k` no. In E12f, 9 lettere eguagliavano 19
perché le 9 includevano `a c i o s`. La regola è la frequenza nel corpus, non il numero.

I sostituti sono affermazioni tipografiche, non normative, quindi valgono come la prova B di
E12f: funzionano perché il **cambio di script** basta, non perché il glifo sia davvero simile.

## Verdetto
- **Multicarattere e bidi: entrambi negativi.** Uno per costruzione (contrazione, toglie
  token), l'altro perché colpisce un rendering che un encoder non ha.
- La **previsione sul bidi era giusta** ed è ora misurata: chiudere il buco non per averlo
  provato, ma perché l'avevamo previsto male solo nella direzione (sospettavamo che l'ordine
  dei caratteri contasse, e [[E12m-swap-tasti]] aveva già mostrato il contrario).
- La copertura delle lettere frequenti migliora l'attacco, ma di 9 punti, restando sotto la
  soglia di confidenza come tutti gli altri.

## Note collegate
[[E12f-omoglifi-livelli]] · [[E12l-zerowidth]] · [[E12m-swap-tasti]] · [[E12e-omoglifi-uts39]] · [[E12-sintesi-superficie]] ·  · [[attacchi-pubblicati]] · [[superficie-attacco]] · [[esperimenti]]

## Output
Nessun file in `results/`: sonda one-off.
```bash
uv run python tools/e12n_multichar_bidi.py
```
