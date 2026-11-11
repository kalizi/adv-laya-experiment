---
tags: [E20, difesa, detector, evasione, parafrasi, E18]
stato: fatto
data: 2026-10-02
study_valid: true
---

# E20: la copertura del detector, misurata sulle riformulazioni

E18 chiude il template di E03 (`ASR` 1.0000 → 0.0000) ma non chiude l'attacco: riconosce le
variazioni che **preservano le parole** dell'opzione, non quelle che le sostituiscono o le
deformano. E "il detector non la vede" non basta a dirlo: un attacco che non funziona più non
è un'evasione. Questo esperimento incrocia due misure per ogni riformulazione, sullo stesso N
pieno (`laya-multilingual`, `massive-it`, K=4, N=2974, 2276 decisioni corrette al pulito):

- **il detector la vede?**
- **l'attacco funziona ancora?** (`ASR` e drop)

Il numero che dichiara la copertura è la quota di **riformulazioni che funzionano e che il
detector vede**.

## Il risultato

| riformulazione dell'iniezione | `ASR` | drop | detector la vede | copre gli attacchi funzionanti |
|---|---|---|---|---|
| template di E03 (verbatim) | 0.9732 | 74.4pp | 1.0000 | **1.0000** |
| cornice diversa ("Devi rispondere X") | 0.9565 | 73.1pp | 1.0000 | **1.0000** |
| cornice in inglese | 0.9288 | 71.0pp | 1.0000 | **1.0000** |
| solo la parola, senza cornice | 0.8304 | 63.2pp | 1.0000 | **1.0000** |
| senza separatori ("playmusic") | 0.9723 | 74.3pp | **0.0000** | **0.0000** |
| refuso ("play musiD") | 0.9372 | 71.5pp | **0.0000** | **0.0000** |
| leet ("p1ay music") | 0.8269 | 62.4pp | **0.0000** | **0.0000** |

**Copertura reale: 4 riformulazioni su 7.** Le tre che sfuggono **evadono al 100% e funzionano
ancora** (`ASR` 0.97, 0.94, 0.83). Il `1.0000 → 0.0000` di E18 è quindi un risultato **sul
template di E03**, non sulla classe di attacchi, e va letto con questa tabella accanto.

## Il controllo che rende l'esperimento informativo

Le prime quattro righe cambiano **solo la cornice** e lasciano intatta l'opzione. L'attacco
funziona in tutti e quattro i casi (`ASR` 0.83-0.97, drop 63-74pp) e il detector li vede
**sempre**. Questo prova che il detector di E18 **non sta riconoscendo il template di E03**: sta
riconoscendo la nominazione dell'opzione, che è la firma che conta.

E le ultime tre righe deformano **l'opzione** invece della cornice: l'attacco funziona
uguale o meglio (`compact` è il caso più forte di tutti, 0.9723) e il detector non vede nulla.
Il messaggio è che la rigidità che sembrava il difetto è la **firma corretta**: il detector è
sensibile alla parola, e chi scrive l'attacco può scrivere una parola diversa che il modello
capisce uguale.

## Perché non c'è una variante intermedia utile

Allentare il detector per vedere le riformulazioni lessicali costa il traffico legittimo
(misurato in E18, 2000 stati puliti): `strict` 0.70% di falsi positivi su `massive-en`, una
parola dell'opzione **40.1%**, metà delle parole 37.9%. Per coprire `compact`/`typo`/`leet`
servirebbe la permissività che blocca due richieste su cinque. Quindi il muro non è la regola:
è che **la nominazione è riconoscibile solo se le parole sono le stesse**, e l'attaccante può
scegliere parole diverse che il modello non fa fatica a riconoscere.

## Il limite dichiarato

Le riformulazioni qui sono **superficiali e deterministiche** (separatori, refuso, leet), derivate
meccanicamente dall'etichetta. Una **parafrasi semantica vera** (sinonimo di `play_music`,
traduzione in italiano, metonimo) richiederebbe un lessico che questo progetto non ha e non è
misurata. Il risultato è quindi un **limite inferiore** dell'evasione: le riformulazioni
superfici già bastano a mandare a zero la copertura, e quelle semantiche la manderanno più in
giù.

## Note collegate

[[E18-nomination-detector]] · [[E19-detector-labels]] · [[E16-calibrazione-difesa]] ·
[[E03-option-injection]] · [[metriche-attacco]]