---
tags: [metodologia, metriche]
---

# Metriche d'attacco

Definizioni operative. Usare **queste** in tutti gli esperimenti.

## Notazione
Su un sottoinsieme `S` di esempi con gold `y`, predizione pulita `ŷ0`, predizione avversaria `ŷa`, confidence `c`.

## Metriche core

### 1. Drop di accuratezza
```
drop = acc(ŷ0) − acc(ŷa)
```
Sempre **relativo allo stesso S**, non al model card. Soglia 🥉: **≥ 20pp**.

### 2. ASR — Attack Success Rate
Solo sugli esempi **giusti** al pulito (evita che l'attacco "vinca" su errori preesistenti):
```
S_ok = { i : ŷ0(i) == y(i) }
ASR  = |{ i ∈ S_ok : ŷa(i) != y(i) }| / |S_ok|
```
Soglia 🥉: **≥ 50%**.

### 3. OER — Overconfident Error Rate
La firma del "fischi per fiaschi":
```
OER(τ) = |{ i : ŷa(i) != y(i) AND c(i) >= τ }| / |S|
```
τ = **0.75** (primario), τ = 0.85 (secondario). OER va sempre riportato anche **in baseline** (il modello sbaglia già confidente).

#### 3b. `c(i)` non è una grandezza sola, quindi τ = 0.75 non è una soglia

`record["confidence"]` cambia definizione con il numero di opzioni `k`:

| `k` | cosa ci mette il backend | numerica su `{0.8, 0.15, 0.05}` |
|---|---|---|
| 2 (`noul`) | `max(p, 1-p)`, cioè la probabilità del vincitore | **0.800** |
| > 2 (`choice`, `score`) | entropia normalizzata `1 - H(p)/log(k)` | **0.442** |
| > 2 su CLM | `p1 - mean(p2..pk)` | **0.700** |

Corretto il 2026-10-04: la terza riga e il valore **0.442** sono stati aggiunti quando CLM è
entrato in campagna, e il 0.294 che stava qui prima non corrispondeva a nessuna delle tre
definizioni su quella distribuzione. Il numero era entrato nel paper e in `CLAUDE.md` prima
di essere verificato, che è il verso di questo errore.

Quindi lo stesso τ è una **majority test** su `noul` e un test di "distribuzione quasi
smussata" su `choice`. I due numeri non si confrontano, e un OER aggregato su primitive
miste non è un giudizio solo.

Misurato sul run pulito di E01 (2000 decisioni, stesso checkpoint, stesso modello,
stessa soglia): `confidence >= 0.75` scatta su **0.2917** dei `noul` e **0.0000** dei
`choice`. I due effetti inoltre si compensano in aggregato (noul inflato, choice
deflesso), il che è il motivo per cui il difetto è sopravvissuto: la media pooled
sembrava stabile.

**Correzioni applicate il 2026-10-04.** `aggregate()` pubblica ora `oer_by_primitive` e
`oer_margin_by_primitive`, e ogni backend dichiara `confidence_definition` in
`run_identity`. Aggiunta `oer_margin(τ)`, che legge `probs` ed è quindi confrontabile
per costruzione:

```
margin(i) = p1(i) - p2(i)          # dalle due probabilita' maggiori, da `probs`
OER_margin(τ) = |{ i : ŷa(i) != y(i) AND margin(i) >= τ }| / |S|
```

`oer` e `oer_margin` **non** sostituiscono l'uno l'altro e `probs` è la sola base
confrontabile. Attenzione: **τ = 0.75 è un punto operativo diverso, non un
riscalamento**. Su un `choice` a 4 opzioni `confidence = 0.8` è una distribuzione
quasi piatta (margin 0.05), quindi `oer_margin@0.75` è più severo di `oer@0.75` e i
due numeri non si confrontano numero per numero. Il punto operativo va rifatto sul
dataset dove si spende, non riciclato da un altro modello: è la lezione degli ECE
dichiarati in E15, applicata alle nostre metriche.

Raggio d'azione misurato su tutti i 593 run archiviati (`tools/oer_audit.py`):
**98 run**, tutti di `typed-decisions` (E01, E02, E04, E12). Gli altri **495** sono su
`massive-it` / `massive-en` / `banking77` / `ag-news`, che costruiscono **una sola**
domanda `choice` e sono strutturalmente immuni. Nessuna conclusione pubblicata è
capovolta: la claim di E04 era `OER ≤ 0.009` e sul margin vale **0.0000**, quindi
rafforzata. Su E01 il pooled passa da 0.0435 a 0.0120. I numeri di OER citati su E03,
E15, E16 e E17 sono tutti su dataset all-choice e non cambiano.

#### 3c. Caso misurato: CLM-v0.1-8B

Su `clm-llama` la terza definizione di `confidence` (`p1 - mean(p2..pk)`, quella
dell'API) rende la soglia 0.75 **strutturalmente irraggiungibile per K grande**, e il
criterio di confidenza di  non è mai raggiunto (max 0.6682 su 16
righe). Non è una misura di robustezza. L'`ASR` resta valido perché è relativo.

Il `margin` invece è la via che funziona, con un punto operativo **derivato dal costo**
sul pulito e non riusato: a K=16 τ=0.9239 blocca 4/464 risposte buone e chiude 0.0672 di
errore indotto, con `benefit_over_clean` positivo a ogni soglia da 0.25 a 0.95.
Dettagli in [[E21-clm-hard-negatives]].

### 4. Confidence sui torti
```
mean_conf_wrong = mean( c(i) for i with ŷa(i) != y(i) )
```
Soglia: **≥ 0.75**.

### 5. Calibrazione
- **ECE** (10 bin, top-label) sul dataset avversario
- **Brier** per `noul`/`choice`
- Confronto ECE baseline vs avversario: un attacco che *alza* ECE + OER è più grave di uno che solo abbassa acc.

### 6. Perturbazione (per classificare la qualità dell'attacco)
- **Δtoken**: differenza in lunghezza (token) tra stato pulito e avversario
- **Preservazione semantica**: checklist qualitativa (il gold è ancora vero?) + opzionale Similarity su sentence-embedding. Se il gold cambia → esempio **fuori** da ASR (goal shift, vedi )

### 7. Transfer
```
transfer = ASR sul checkpoint/dataset B di un attacco ottimizzato su A
```
Riportare la matrice A→B. **Tre assi**: checkpoint, dataset, **precisione**.

Sul terzo asse la regola è la stessa degli altri due e va detta esplicitamente: la
**baseline va rimisurata a ogni precisione**. Confrontare "ASR a q4" con "baseline a fp32"
è l'errore di E03 ripetuto in un posto nuovo, ed è il modo più economico di ottenere un
ASR 1.0 falso. E la precisione va nel `run_id`, insieme a backend e compilazione, più una
guardia che verifichi che il backend aperto sia quello richiesto: se il flag non viene
inoltrato, N configurazioni diverse girano sullo stesso grafo e la tabella dice "la
riduzione di precisione non cambia niente" — un null result costruito a mano.
Vedi [[E15-quantizzazione]].

## Cosa NON usare come metrica principale
- Accuracy assoluta sul dataset avversario senza baseline
- "Il modello ha detto X" (aneddotico, N=1)
- Soft accuracy: utile in analisi, non è il criterio della study

## Report minimo per esperimento
Nel **json** del run, i numeri misurati:
```
N, |S_ok|, acc0, accA, drop, ASR, OER(0.75), OER(0.85),
mean_conf_wrong, ECE0, ECEA, Δtoken, ms/decisione
```
Nella **nota** del vault, la parte che non è una misura ma un giudizio:
```
livello criteri (🥉/🥈/🥇) e la frase che lo sostiene
```

> Corretto il 2026-10-01. Il livello stava nell'elenco insieme ai numeri, e finiva per
> non essere in nessuno dei due posti: nessun json lo riportava e nessuna nota lo dichiarava
> in modo uniforme. Un verdetto è un giudizio e va motivato per iscritto; un numero no.

Due cose che la lista non diceva e che servono per leggere una riga:
- **`|S_ok|` è `|S_ok ∩ S_adv|`**: l'ASR si calcola sugli esemmi effettivamente attaccati
  (`metrics.core.asr` solleva se un caso manca, invece di diluirlo).
- **Il `temperature` applicato va registrato** quando si fa uno sweep su K: `temp_bucket`
  cambia a K=5/10/11, quindi due righe vicine possono essere prodotte con sharpness diversa
  e i loro `OER` non sono confrontabili a occhio.
- **La velocita' si riporta in ms per decisione**, mai in ms per batch, e **il device va
  registrato per campione**: `env_fingerprint` scrive la GPU se CUDA è disponibile, quindi
  un run con fallback CPU dichiarerebbe la GPU che non stava usando (~10x di differenza).
  Vedi [[performance-velocita]].

## Note collegate
 · [[protocollo-dataset-valido]] · [[rlcd-calibrazione]] · [[esperimenti]]
