---
tags: [esperimento, superficie, character-level, unicode, confusables, stealth]
stato: fatto
data: 2026-09-30
study_valid: true
---

# E12e — omoglifi con la tabella normativa UTS #39

## Perché rifare E12d
[[E12d-omoglifi]] misurò ASR 0.62-0.76 con una tabella di sostituzioni **scritta a mano**. Il
problema è che la tesi dell'attacco (indistinguibile per un umano) poggiava su quella tabella,
cioè su quello che avevo scritto io. Sostituire la tabella con il dato normativo **UTS #39**
(`confusables.txt`, lo standard che usano browser e IDN) rende la pretesa verificabile.

Il metodo è deliberatamente **solo testo**: nessun font, nessuna rasterizzazione. La forma del
glifo la decide il font installato, quindi misurare i pixel avrebbe misurato Noto Sans, non la
proprietà dei caratteri. UTS #39 lavora su *skeletri* di carattere, che è la definizione
normativa di "confondibile".

## La tabella a mano era sbagliata in 9 casi su 20
Confronto lettera per lettera della mia tabella di E12d contro UTS #39:

| mia scelta | esiste in UTS #39? |
|---|---|
| `b→β`, `e→ε`, `k→κ`, `l→λ`, `m→μ`, `n→η`, `t→τ`, `x→χ`, `z→ζ` | **no, non esistono** |
| `a→α`, `c→ϲ`, `d→ԁ`, `i→ι`, `o→ο`, `p→ρ`, `r→г`, `s→ѕ`, `u→υ`, `v→ν` | sì |

Nove sostituzioni erano plausibili a occhio ma **non confusabili secondo lo standard**: `λ` per
`l`, `μ` per `m`, `η` per `n` non sono considerati confondibili con le rispettive lettere
latine. Un attaccante che si fosse fidato della mia tabella avrebbe usato sostituti che un
umano *vedrebbe* essere diversi.

## Risultati con la tabella normativa
N=30, baseline 0.7533, determinismo verificato 0/150.

### Curva di dose, cross-script (greco/cirillico)
| dose | Δtoken | label cambiate | flip | ASR | confidenza | OER |
|---|---|---|---|---|---|---|
| 10% | 42 | 28/150 | 14 | 0.1239 | 0.187 | 0 |
| 25% | 100 | 64/150 | 45 | **0.3982** | 0.241 | 0 |
| 50% | 176 | 88/150 | 66 | 0.5841 | 0.212 | 0 |
| 100% | 286 | 100/150 | 79 | **0.6991** | 0.211 | 0 |

**La tabella normativa funziona meglio di quella a mano: ASR 0.6991 contro 0.6195.** I numeri di
E12d erano prudenti. Il motivo: avevo scelto i greci "più belli" (`λ`, `μ`, `η`) che sono anche i
peggi, mentre quelli normativi (`а`, `ԁ`, `і`, `ј`, `ѕ`, `ϱ`, `ѡ`, `Ӏ`) sono in parte cirillici e
costano più token ma rompono di più.

### Cross-script contro same-script
| tabella | Δtoken | ASR | ASR/token |
|---|---|---|---|
| cross (greco/cirillico) | 286 | 0.6991 | **0.00244** |
| same (matematico/fullwidth) | 934 | 0.7434 | 0.00080 |

Il same-script arrappa più in assoluto ma costa 3× i token e per token è **3× meno
efficiente**. Inoltre la pretesa di invisibilità è più debole: il fullwidth ha larghezza
diversa, il matematico è un altro stile. **Per un attacco furtivo la scelta giusta è
cross-script.**

### Il punto di efficienza è il 25%, non il 100%
| braccio | Δtoken | ASR | ASR/token |
|---|---|---|---|
| cross dose 25% | 100 | 0.3982 | **0.00398** |
| cross dose 50% | 176 | 0.5841 | 0.00332 |
| cross dose 100% | 286 | 0.6991 | 0.00244 |
| same-script 100% | 934 | 0.7434 | 0.00080 |
| spazi distribuiti (controllo) | −25 | 0.6726 | — |

L'efficienza massima è a dose 25%: **ASR 0.40 spendendo 100 token**. Oltre si paga sempre di
più per meno. Questo conferma quanto trovato in [[E12c-simbolo]]: l'ASR è una funzione dei
token spesi, e tutte le famiglie di caratteri stanno sulla stessa curva.

### Il meccanismo è ancora la corruzione in place
Il controllo di E12d tiene senza cambiare: spazi distribuiti **0.6726** contro omoglifi
**0.6991**, con Δtoken negativo. Non è lo script a fare il danno, è il fatto di sostituire
caratteri dentro le parole.

### Il collasso del vocabolario è lo stesso
12 label distinte → 7, e sparite le stesse sei di E12d: `3`, `continue`, `false`, `harmful`,
`observe`, `success`. Accuracy 0.753 → ~0.28 contro 0.083 di caso.

## Limite di copertura: 7 lettere non sono attaccabili
UTS #39 non dà omoglifi cross-script per **f, g, k, m, n, t, z**, e per **`m` non esiste
nessun confusabile in assoluto** (nemmeno same-script: zero).

| lettera | omoglifi cross | omoglifi same |
|---|---|---|
| a | 2 | 18 |
| l, o, p | 4 | 31, 24, 23 |
| c, i, w, y | 3 | 12-19 |
| **f, g, k, m, n, t, z** | **0** | 12-13 (**m: 0**) |

È un limite reale e non aggirabile con questa tabella: `m` non si può toccare. Su
typed-decisions le parole che portano il gold contengono proprio queste lettere
(`duration_s`, `minutes`, `constraint_violations`), quindi l'attacco colpisce il testo
descrittivo ma non i nomi di campo. Chi volesse colpire i numeri dovrebbe usare
[[E03-option-injection]].

## Verdetto
- **La tesi regge, e ora su dato normativo**: con i confusabili che lo standard dichiara
  confondibili, l'attacco dà **ASR 0.6991**. Non è più "quello che pensavo che si vedessero
  uguale", è "quello che Unicode dice che un umano confonde".
- **E12d era conservativo**, non sovrastimato. Il confronto ha corretto la tabella in meglio,
  non in peggio.
- **Il limite vero resta la confidenza**: **OER 0 a ogni dose**, confidenza sul torto
  0.19-0.24. Il modello non viene ingannato, viene degradato: il vocabolario delle risposte
  si riduce da 12 a 7 label e metà sparisce. Nessun errore convinto in nessuna
  configurazione, quindi nessuna di queste misure soddisfa i . L'unico
  attacco che li produce resta nominare un'opzione ([[E03-option-injection]]).
- **Il meccanismo non è lo script**: spazi distribuiti ottengono lo stesso ASR. Gli omoglifi
  servono perché **un revisore umano non li distingue**, non perché ingannino il modello più
  di quanto faccia uno spazio.
- **Il punto dolente pratico è il costo**: 4.8× di token è il prezzo dell'invisibilità. Il
  modello non normalizza Unicode, e quello è il difetto di robustezza da segnalare
  indipendentemente dall'attacco.

## Cosa resta
- **Verificare l'invisibilità su un umano vero**: UTS #39 dice "confundibile secondo lo
  scheletro del carattere", che è una proxy, non una verifica. Con un campione di revisori
  si chiuderebbe il cerchio, e non è misurabile qui.
- Colpire le lettere non coperte (`m`, `n`, `t`, `k`) con una strategia diversa: sostituzione
  multi-carattere (`mn` → `mн`) aggira il vincolo di UTS #39.
- **Estendere a livelli**: la stessa procedura su caratteri *visibili* (β per `b`) dice se la
  confusabilità predice l'efficacia → **fatto in [[E12f-omoglifi-livelli]]**, risposta no.

## Note collegate
[[E12d-omoglifi]] · [[E12c-simbolo]] · [[E03-option-injection]] · [[E04-truncation]] ·  · [[superficie-attacco]] · [[token-budget-head]] · [[esperimenti]]

## Output
Nessun file in `results/`: sonda one-off. I numeri si riproducono con:
```bash
uv run python tools/e12e_confusables.py      # UTS #39 e verifica della tabella a mano
uv run python tools/e12e_homoglyph_uts.py    # curve, meccanismo, collasso
```
Il file `confusables.txt` si scarica da `https://www.unicode.org/Public/security/latest/confusables.txt`