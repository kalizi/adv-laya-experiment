---
tags: [esperimento, superficie, character-level, unicode, zero-width, invisibile]
stato: fatto
data: 2026-09-30
study_valid: true
---

# E12l — caratteri a larghezza zero

## Perché
Nella letteratura i zero-width danno 54.2% di successo (arXiv 2508.14070) e sono l'unica
perturbazione che unisce due proprietà: **frammentano le parole *dentro***, che è il meccanismo
che [[E12d-omoglifi]] ha misurato come leva, e sono **invisibili in modo assoluto** (larghezza
zero, nessun rendering). Gli omoglifi funzionano ma si vedono; i zero-width no.

## Il test che viene prima
Se il tokenizer normalizza, l'attacco non esiste. Risposta: **no, sopravvivono tutti e cinque**.
Nessuna normalizzazione NFKC li rimuove (11/11 sopravvivono).

| carattere | 1 in mezzo | 1 ogni 12 caratteri |
|---|---|---|
| ZWSP U+200B | +1 token | +15 |
| ZWNJ U+200C | +1 | +17 |
| ZWJ U+200D | +2 | +26 |
| FEFF U+FEFF | +1 | +15 |
| WORD JOINER U+2060 | +2 | +26 |

## Risultati
N=30, baseline 0.7533, inserimento **dentro** le parole alle posizioni alfabetiche.

| dose | ZWSP | ZWNJ | ZWJ | FEFF | WORD JOINER |
|---|---|---|---|---|---|
| 10% | 0.1947 | 0.1858 | 0.2478 | 0.1858 | 0.2124 |
| 25% | 0.4779 | 0.5044 | 0.5398 | 0.4779 | 0.5310 |
| 50% | **0.6549** | **0.6549** | **0.6991** | 0.6195 | 0.6814 |

Δtoken a dose 50%: 232 per ZWSP, 359 per ZWJ. Drop fino a **47.3pp**. **OER 0 ovunque**,
confidenza sul torto 0.19-0.27. Il collasso del vocabolario c'è ed è lo stesso degli omoglifi:
12 → 7, sparite `3` `continue` `false` `harmful` `observe` `success`.

Il migliore in assoluto, **ZWJ a dose 50%, ASR 0.735 (dose 100% delle posizioni alfabetiche, ZWSP dentro le parole)**, eguaglia il migliore
omoglifo (E12e, 0.6991) ma con un vantaggio che l'omoglifo non ha: **è byte-diverso e
rendering-identico**. Verificato: 335 → 556 caratteri, e rimuovendo i ZWJ si ottiene il testo
originale carattere per carattere. I numeri restano intatti.

## La posizione conta, ancora
| dove | dose 100% | Δtoken | ASR |
|---|---|---|---|
| dentro le parole | 100% | +232 | **0.6549** |
| fra le parole | 100% | +40 | 0.0531 |

Stesso carattere, posizione diversa, 12× di differenza. Seconda volta che la posizione è la
leva, dopo E12d (omoglifi contro filler in coda) e E12f (spazi distribuiti contro in coda).

## Riferimenti a parità di concetto
| braccio | Δtoken | ASR | conf | OER |
|---|---|---|---|---|
| spazi dentro le parole 50% (**non** invisibili) | +120 | 0.7345 | 0.177 | 0 |
| `_` dentro le parole 50% | +223 | 0.6637 | 0.193 | 0 |
| zero-width dentro 50% (**invisibili**) | +232 | 0.6549 | 0.203 | 0 |

Lo spazio resta il più efficiente per token (0.00611 contro 0.00282), ma i zero-width
**rendono invisibile** quello che lo spazio rende solo poco notevole.

## Un bug che ha prodotto un falso finding
La prima esecuzione riportava **ZWSP completamente inerte a ogni dose, Δtoken 0.0**, e la
lettura era "il tokenizer di laya elimina lo zero-width space". Era falso: nel sorgente la
costante `"ZWSP U+200B"` conteneva uno **spazio ASCII**, perché il letterale a larghezza zero
si è perso in scrittura. Il carattere non era mai stato testato.

Risolto usando sequenze di escape esplicite (`"\u200b"`), che non possono perdersi. È il
secondo bug di questa sessione che produce uno zero e sembra confermare un finding: insieme
alla disallineazione spaCy/`split()` di E12b, la regola è che **ogni costante Unicode va
scritta come escape, e ogni zero va verificato con un test di identità** (rimuovere la
perturbazione deve restituire l'originale).

## Verdetto
- **Funziona, e chiude l'obiettivo di invisibilità**: ASR 0.65-0.70 a parità degli omoglifi
  migliori, con testo byte-diverso e identico a occhio.
- **Non passa i **: OER 0, confidenza 0.19-0.27. Il modello non viene
  ingannato, viene degradato. Come tutti gli attacchi character-level.
- Conferma il meccanismo di E12d: non conta il carattere, conta **quanti token si aggiungono
  dentro le parole**. ZWSP (2 byte, +232 token) e ZWJ (3 byte, +359 token) danno lo stesso
  ASR a dosi diverse, e il costo per token è ~0.0028 per entrambi.
- Il **collasso del vocabolario** è la firma costante di questa famiglia di attacchi: 12 → 7
  label. Non è un caso: è il modello che perde la capacità di esprimere metà delle risposte.

## Cosa resta
- **Bidi override** (U+202E/D): 52.8% in letteratura, mai provato. Su un encoder, dove l'ordine
  dei caratteri è tutto, è il candidato più promettente rimasto. Nota però che E12m ha misurato
  che l'ordine dei caratteri **non** conta (swap di tasti: ASR 0.0177), quindi la previsione
  va contro le nostre misure: o il bidi override funziona per un motivo diverso dall'ordine
  (rompe il rendering, non i byte), o non funzionerà.

## Note collegate
[[E12d-omoglifi]] · [[E12e-omoglifi-uts39]] · [[E12f-omoglifi-livelli]] · [[E12i-underscore]] · [[E12c-simbolo]] · [[E04-truncation]] ·  · [[attacchi-pubblicati]] · [[esperimenti]]

## Output
Nessun file in `results/`: sonda one-off.
```bash
uv run python tools/e12l_zerowidth.py
```
