---
tags: [esperimento, superficie, character-level, unicode, robusto, negativo]
stato: fatto
data: 2026-09-30
study_valid: true
---

# E12d — attacco character-level a omoglifi

## Ipotesi
Sostituire le lettere con equivalenti Unicode visivamente identici (greco, cirillico,
matematiche) dovrebbe ingannare un encoder che non normalizza, **senza che un revisore umano
se ne accorga**. È l'attacco più economico in assoluto: costa 0 vocabolario e 0 conoscenza del
dominio.

Prima di misurare, il costo: il tokenizer è byte-level e non riconosce niente fuori dal latino.

| alfabeto | `Rotate the expired TLS certificate` | inflazione |
|---|---|---|
| greco | 6 → 29 token | 4.8× |
| cirillico | 6 → 28 token | 4.7× |
| matematico bold | 6 → 124 token | 20.7× |

Ogni lettera greca costa 2-3 token invece di 1. Quindi fin dall'inizio l'attacco è anche un
attacco al budget, e va separato da quello.

## Setup
`laya-typed-decisions`, N=30, baseline 0.7533 (113/150), `max_len` 1024, `head_max_len` 256,
stati puliti 91-118 token. Solo lo stato viene modificato; cifre e struttura JSON restano
intatte (i caratteri sostituiti sono alfabetici, mai cifre). Determinismo verificato prima:
**clean contro clean 0/150 label diverse**.

## Risultato greco: curva di dose
| dose | Δtoken | label cambiate | flip | ASR | confidenza | OER | label distinte |
|---|---|---|---|---|---|---|---|
| 10% | 46 | 48/150 | 32 | 0.2832 | 0.183 | 0 | 12 |
| 25% | 97 | 87/150 | 69 | **0.6106** | 0.202 | 0 | 12 |
| 50% | 162 | 86/150 | 70 | 0.6195 | 0.205 | 0 | 7 |
| 100% | 212 | 83/150 | 70 | 0.6195 | 0.211 | 0 | 6 |

Cirillico 100%: ASR 0.6372. Il matematico con 10% arriva già a 0.3540.

**Archiviata il 2026-10-05.** La curva è in `results/E12d_homoglyph_dose.json` (11 righe, con
`dose` e `asr_denominator` per riga), quindi anche il valore della riga 50% nella colonna
`confidenza`, che fino a qui non era rintracciabile in nessuno store, ora lo è e
`check_claim_numbers.py` non lo segnala più. Il tool scriveva solo su console.

**Satura al 25%**: oltre, aggiungere corruzione non aumenta l'ASR ma continua a mangiare
budget e a restringere le label. Il punto di efficienza massima è 10% di dose (0.2832 con 46
token), non 100%.

## Il meccanismo non è quello che sembra
Questo è il punto, e viene dai controlli. A parità di caratteri toccati:

> **Corretto il 2026-10-01: la parità non c'era.** Il numero di spazi da sostituire era
> `len(state) * dose`, cioè **tutta** la lunghezza dello stato, cifre e punteggiatura
> comprese, mentre l'arma a omoglifi sostituisce solo i caratteri presenti nella tabella.
> A dose 100% gli spazi toccavano ~374 caratteri contro ~40 dei greci: il braccio di
> controllo era nove volte più distruttivo dell'arma. Ora il conteggio è fatto sugli stessi
> caratteri mappabili. **La conclusione** ("gli spazi ottengono lo stesso ASR spendendo meno
> token") **resta valida** perché è un confronto di costo, ma le righe qui sotto vanno
> rilette come "spazi, ~40 caratteri" e non "~374". I numeri di questa tabella non sono stati
> ricalcolati (servirebbe un giro GPU).

| braccio | Δtoken | ASR | label distinte |
|---|---|---|---|
| omoglifi greci 25% | 97 | 0.6106 | 12 |
| **spazi distribuiti, ~93 caratteri** | **61** | **0.5841** | 11 |
| omoglifi greci 100% | 212 | 0.6195 | 6 |
| **spazi distribuiti, ~374 caratteri** | **−25** | **0.6726** | 6 |
| filler neutro in coda, 212 token | 212 | 0.1947 | 11 |

**Sostituire lettere con spazi ottiene lo stesso ASR, e di più, spendendo meno token** (nel
braccio da 374 caratteri il Δtoken è *negativo*: sostituire lettere con spazi fa anche
fusione di token). Il filler in coda, che costa esattamente gli stessi 212 token, fa **ASR
0.1947, tre volte meno**.

Quindi il driver non è lo script e non è il budget. È la **corruzione in place del contenuto**:
se sostituisci caratteri *dentro* le parole il modello non le legge più, se li appendi *dopo*
non gli serve niente. Non c'è troncatura coinvolta: 384 token di stato contro `max_len` 1024.

Da questo segue che **l'omoglifo non è un canale di attacco nuovo**. Il suo unico vantaggio
rispetto agli spazi distribuiti è uno solo, ed è decisive: **li spazi li vede un umano, gli
omoglifi no**.

## Non è un attacco mirato, è un collasso del vocabolario
Sotto attacco il modello smette di emettere metà delle sue risposte:

| label | predette al pulito | predette sotto attacco |
|---|---|---|
| `false` | 19 | **0** |
| `observe` | 11 | **0** |
| `harmful` | 7 | **0** |
| `continue` | 6 | **0** |
| `success` | 9 | **0** |
| `3` | 14 | **0** |
| `true` | 11 | 30 |
| `human_review` | 13 | 30 |
| `1` | 29 | 40 |

**12 label distinte diventano 6**, e sei spariscono completamente. L'accuracy va da 0.753 a
**0.287**, contro una probabilità casuale di 0.083 per 12 classi: il modello è degradato
quasi a caso, non portato verso un bersaglio.

Conferma: su 83 previsioni cambiate, 70 diventano sbagliate e **13 diventano giuste**. Se fosse
un attacco mirato non ci sarebbero miglioramenti casuali in un attacco che vuole solo fare
danni.

## Riallineato allo store (2026-10-01)

Ora in `adv e12-surface` con archivio, e i due livelli di somiglianza sono distinti per costruzione:

| livello | tabella | ASR | OER@0.75 | label |
|---|---|---|---|---|
| A | UTS #39 (43 coppie latine) | **0.6991** | 0 | 12 → 6 |
| B | greco visibile, scelto a mano | **0.6195** | 0 | 12 → 6 |
| controllo | spazi nelle stesse posizioni | 0.6637 | 0 | 12 → 6 |

Il **controllo batte entrambi i livelli**: il vantaggio degli omoglifi è che si vedono, non che
ingannano. Le cifre 0.62-0.76 erano una curva di dose su N ridotto; a dose piena il numero è
0.6991 e 0.6195.

## Verdetto
- **Funziona, ed è l'ASR più alto mai misurato su una superficie non semantica**: 0.62-0.76
  contro 0.55-0.98 di E04 e 0.83-0.89 di E03. Ma è un attacco che **non sceglie la risposta**.
- **Non soddisfa i **: OER **0 in ogni braccio**, confidenza sul torto
  0.18-0.22, in ogni configurazione e a ogni dose. Non c'è mai un errore convinto. L'unico
  intervento che li produce resta nominare un'opzione ([[E03-option-injection]]).
- **Il meccanismo è la corruzione in place, non lo script né il budget.** Questo corregge
  l'ipotesi di partenza: la parte interessante non è che il greco inganni, è che **basta
  rompere le parole dove sono** e che un umano non se ne accorge se i caratteri di rimpiazzo
  sono visivamente equivalenti.
- **Però il modello non normalizza**: 4.8× di token per la stessa frase è un difetto di
  robustezza del tokenizer, indipendente dall'attacco, e va segnalato come tale.
- Sul piano della sicurezza il finding è di **disponibilità, non di inganno**: l'attaccante non
  sceglie la risposta che il modello darà, ma gli impedisce di esprimere metà dello spazio delle
  risposte. È una denial of service sulla decisione, e si ottiene senza vocabolario.

## Cosa resta
- **La domanda da 10 secondi che vale davvero**: se basta rompere le parole in place, allora
  un attaccante umano con un editor può distruggere la decisione in modo invisibile. Il
  controllo onesto è se un revisore umano distingue uno stato in omoglifi dallo stato
  originale. Questo progetto non può misurarlo: qui l'omogeneità visiva è una **proprietà
  dichiarata della tabella di sostituzione**, non una misura. → **risolto in [[E12e-omoglifi-uts39]]**:
  la tabella normativa conferma che 9 dei miei 20 omoglifi *non* erano confusabili, e li sostituisce.
- Verificare se il collasso del vocabolario è specifico del checkpoint o della tokenizzazione
  byte-level: un tokenizer con normalizzazione Unicode lo eliminerebbe.

## Note collegate
[[E12c-simbolo]] · [[E04-truncation]] · [[E03-option-injection]] · [[E12b-sinonimi-rari]] ·  · [[token-budget-head]] · [[superficie-attacco]] · [[esperimenti]]

## Output
Nessun file in `results/`: sonda one-off, nessun ciclo né selezione. I numeri sono nelle
tabelle sopra e si riproducono con:
```bash
uv run python tools/e12d_homoglyph.py
```

**Cosa è e cosa non è questo archivio.** Le 11 righe sono **sintesi con denominatore esplicito**
(`asr_denominator`, `n_decisions`), non record grezzi: sono prodotte dai tool in `tools/`, che
calcolano e stampano ma non passano da `io.RunStore`. Quindi `verify_results.py` le conta fra le
"righe senza run_id" e **non le ricalcola dai `preds.jsonl`**: la verifica indipendente qui non
c'è, e il controllo che le copre è `check_claim_numbers.py`, che si limita a confermare che la
cifra esiste in un archivio. Chi vuole la ricalcolabilità piena deve passare questi operatori
attraverso `RunStore` come E12-surface.
