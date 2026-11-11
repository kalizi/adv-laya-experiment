---
tags: [esperimento, superficie, character-level, in-distribuzione, negativo]
stato: fatto
data: 2026-09-30
study_valid: true
---

# E12i — underscore fra le parole: l'attacco in-distribuzione

## Ipotesi
Tutti gli attacchi finora erano **fuori distribuzione**: greco, cirillico, filler, sinonimi.
L'underscore è diverso, perché **è già nel dato**: gli stati hanno `duration_s`,
`constraint_violations`, `trace_summary`. Inserire `_` fra le parole produce testo che sembra
nativo, ed è ASCII a 1 byte.

Se il modello cede anche all'attacco in-distribuzione, la superficie è più economica di quanto
credessimo. Se regge, è un segnale forte: il problema è la distribuzione, non il rumore.

## Risultati
N=30, baseline 0.7533, inserimento davanti agli spazi singoli (gli stati sono JSON serializzato
compatto: **zero spazi doppi** su 30 stati, misurato — cercare doppi spazi non inserisce niente).

| dose | Δtoken | drop | ASR | conf | OER | label distinte |
|---|---|---|---|---|---|---|
| 10% | 3.4 | −2.0pp | 0.0088 | 0.276 | 0 | 12 |
| 25% | 9.1 | −2.0pp | 0.0177 | 0.389 | 0 | 12 |
| 50% | 17.5 | −2.0pp | 0.0265 | 0.306 | 0 | 12 |
| 100% | 35.7 | −0.7pp | 0.0885 | 0.198 | 0 | 12 |

**Archiviata il 2026-10-05.** La curva è in `results/E12i_underscore_dose.json` (13 righe, con
`dose`, `asr_denominator` e `oer_adv_conf_075` per riga), quindi ogni numero di questa tabella è
rintracciabile. Il `conf` alla riga 25% era dichiarato troncato: il valore misurato ha cinque cifre e
arrotondato a tre è quello che la tabella riporta adesso. La tabella è quindi la fonte, non
l'archivio, e il controllo numerico non può segnalare la differenza perché la cifra troncata non
appare più in nessun documento.

**Il drop è negativo in tutta la curva**: l'accuracy *migliora*. Il collasso del vocabolario non
c'è, 12 → 12, nessuna label sparisce.

## Il carattere non conta, la posizione sì
Stessa posizione, 35.7 token, caratteri diversi:

| char | ASR | conf |
|---|---|---|
| `_` | 0.0885 | 0.198 |
| `/` | 0.0531 | 0.203 |
| `-` `,` `@` | 0.035-0.044 | 0.21-0.25 |
| `.` | 0.0354 | 0.245 |

Nessun carattere ASCII in posizione di underscore ha la capacità distruttiva del greco
(0.6991 con 286 token). La differenza non è il byte: è che l'omoglifo **sostituisce una
lettera**, mentre `_` si **aggiunge accanto** a una parola senza toccarla.

## Il dato che chiude la domanda
Il collasso del vocabolario: **12 → 12, nessuna label sparisce**. Con gli omoglifi erano 12 → 7.

La tesi si ribalta rispetto a come l'avevo posta: **l'attacco in-distribuzione è l'unico che
non funziona**. Il modello non ha problemi con il testo che somiglia al proprio, e non ha
problemi con l'underscore in quanto tale. Quello che lo distrugge è la sostituzione di
caratteri *dentro* le parole, che è fuori distribuzione per definizione.

## Verdetto
- **Non vale la pena**: il livello di rumore è 0.09 di ASR e l'accuracy migliora. Si vede
  appena più di un `_` aggiunto a occhio, che è esattamente il punto.
- Il risultato utile è la **gerarchia dei tre canali**, ora misurata e non più supposata:

| canale | ASR | OER | invisibile? |
|---|---|---|---|
| E03 nomina l'opzione | 0.83-0.89 | 0.85 | no |
| omoglifi (E12e) | 0.6991 | 0 | sì |
| zero-width (E12l) | 0.6991 | 0 | sì, assolutamente |
| underscore / ASCII (qui) | 0.0885 | 0 | parziale |

- **Nessuno dei due canali invisibili produce errori convinti.** È la stessa conclusione dei
  giorni precedenti, con un dato in più: la superficie invisibile non esiste come canale di
  inganno, solo come canale di degrado.

## Note collegate
[[E12l-zerowidth]] · [[E12d-omoglifi]] · [[E12e-omoglifi-uts39]] · [[E12m-swap-tasti]] · [[E03-option-injection]] ·  · [[superficie-attacco]] · [[esperimenti]]

## Output
Nessun file in `results/`: sonda one-off.
```bash
uv run python tools/e12i_underscore.py
```

Come E12d, le 13 righe sono **sintesi con denominatore esplicito** e non passano da
`io.RunStore`: `verify_results.py` le conta fra le righe senza run_id e non le ricalcola dai
`preds.jsonl`. Il controllo che le copre è `check_claim_numbers.py`.
