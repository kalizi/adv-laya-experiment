---
tags: [E17, configurazione, difesa, guardrail, robustezza, negativo]
stato: fatto
data: 2026-10-02
study_valid: true
---

# E17: nessuna configurazione chiude il buco

Domanda: **esiste una configurazione di esecuzione che chiude il buco nel guardrail?** Non "quale
configurazione rende l'attacco più debole", che è una scelta, ma se una qualunque lo chiude. Se la
risposta è no, il buco diventa una proprietà del meccanismo invece che della messa a punto, che
è una dichiarazione più forte di "con questa configurazione l'ASR vale 0.87".

Stesso attacco di E03: `laya-multilingual`, `massive-it`, K=4, template `append`, **N=2974**, la
griglia di knob che possono muovere i numeri. `results/E17_configuration.json`, 6 righe,
verificate dagli store con deviazione `0.000e+00`.

## La griglia e i numeri

| configurazione | `fast` | `compile` | `bs` richiesto | `bs` efficace | ASR | OER@0.75 | conf. torto |
|---|---|---|---|---|---|---|---|
| stock | no | no | 16 | 2 | 0.9732 | 0.897 | 0.942 |
| fast | **sì** | no | 16 | 2 | 0.9728 | 0.898 | 0.942 |
| compile | no | **sì** | 16 | 2 | 0.9723 | 0.896 | 0.942 |
| fast+compile | **sì** | **sì** | 16 | 2 | 0.9728 | 0.898 | 0.942 |
| stock-batch1 | no | no | 1 | 2 | 0.9732 | 0.897 | 0.942 |
| fast-batch1 | **sì** | no | 1 | 2 | 0.9728 | 0.898 | 0.942 |

Al pulito l'`OER@0.75` è 0.021 su tutte e sei: il buco non è una condizione del traffico
legittimo, è quello che l'attacco crea.

**Escursione totale: 0.09 punti percentuali di ASR e 0.0013 di `OER@0.75`.** Le due cifre sono
differenze dei valori archiviati, non campi: ASR `0.973210 − 0.972344 = 0.000866`, `OER@0.75`
`0.897781 − 0.896436 = 0.001345`. Il verdetto è calcolato sui numeri (`_verdict` in
`e17_configuration.py`), non scritto: nessuna delle sei righe chiude il buco e nessuna indebolisce
l'attacco.

## Le sei righe sono tre calcoli, e due coppie sono byte-identiche

Il punto più forte della griglia non è che i numeri sono vicini, è che **alcune righe sono lo
stesso calcolo due volte**. Confronto le previsioni riga per riga dai `preds.jsonl`:

| confronto | label identiche (pulito / attaccato) |
|---|---|
| `stock` ≡ `stock-batch1` | **2974/2974 · 2974/2974** |
| `fast` ≡ `fast+compile` ≡ `fast-batch1` | **2974/2974 · 2974/2974** |
| `stock` vs `fast` | 2962/2974 · 2971/2974 (differiscono 12 e 3) |
| `stock` vs `compile` | 2958/2974 · 2972/2974 (differiscono 16 e 2) |

Quindi la griglia copre **tre calcoli distinti**: `stock`, `fast`, `compile`. Il payload lo
dichiara per nome (`n_calcoli_distinti: 3`): contare le sei righe come sei esperimenti
indipendenti stringerebbe l'escursione a caso, perché tre righe su sei sono lo stesso calcolo. E due affermazioni
diventano molto più forti di "i numeri coincidono":

- **`batch_size` non ha prodotto nulla.** Non numeri uguali: previsioni byte-identiche. Il flag
  entra nel `run_id` (che cambia) ma non cambia una decisione.
- **Il fast path prevale sul compile**, come laya dichiara: la combinata dà esattamente `fast`.

`fast` e `compile` sono percorsi davvero diversi (12 e 16 previsioni su 2974 cambiano) e il buco
resta identico. **Tre implementazioni numericamente diverse, stesso esito.**

## Il confronto appaiato, che è quello rigoroso

I denominatori separati differiscono (2277, 2279, 2278): il pulito dà un numero di risposte
corrette leggermente diverso per configurazione, quindi gli ASR pubblicati non sono calcolati sullo
stesso insieme e il confronto, taken così, è indipendente. Quello appaiato si fa sull'intersezione
— le decisioni corrette al pulito **in tutte e tre** le configurazioni, che sono 2270:

| configurazione | ASR appaiato | CI 95% (Wilson) | `OER@0.75` appaiato |
|---|---|---|---|
| stock | 0.9731 | [0.9656, 0.9790] | 0.8899 |
| fast | 0.9727 | [0.9651, 0.9786] | 0.8903 |
| compile | 0.9722 | [0.9647, 0.9782] | 0.8894 |

L'escursione appaiata è **0.000881**, cioè gli stessi 0.09 punti percentuali. E l'ampiezza della
CI al 95% su 2270 decisioni è **0.013389**: **l'effetto della configurazione è 15.2 volte più
piccolo dell'incertezza campionaria.** Tutti e tre i valori stanno nel payload (`paired`), non in
uno script di analisi, e il numero di decisioni mancanti per riga è dichiarato (zero). Perciò il finding regge sul *verso* e sulla *soglia*, non
sull'ordine fine fra le righe — e su quello l'incertezza è già stata detta una volta e non conta
due volte.

## Perché `fast` è una cella e non un dettaglio

`laya.accelerate()` è `strict=False`: se TileLang manca ricade sul forward stock **stampando un
warning**. Una richiesta che ricade sarebbe stata una riga `fast` che non è fast. Il payload porta
`requested_fast` e `fast_engaged` separati, il verdetto elenca `fast_richiesto_e_non_attivato`, e
qui è **vuoto**: nelle righe che lo chiedevano TileLang ha compilato e girato davvero
(`max |Δconf| 0.0216` fra stock e fast, cioè il percorso era davvero diverso).

La cella `fast+compile` esiste perché laya dichiara che **il fast path prevale sul compile**: se le
due combinate dessero qualcos'altro da `fast` sarebbe un bug, e senza la cella non ci saremmo
accorti. Qui danno lo stesso numero, come previsto.

## `--batch-size` è inerte anche qui

`batch_size 1` e `16` danno lo stesso `batch_size_effective` = **2**, cioè lo stesso numero di
gruppamenti su 2974 stati. Non è un caso di questo dataset: `loaders.load_cases` dà a ogni caso il
proprio set di opzioni (rng per caso, perché `--limit` non deve cambiare le domande) e
`laya.predict_batch` applica un solo dizionario `questions` a tutti gli stati di una chiamata. Quindi
su `massive-it`, `massive-en` e `banking77` il batching non può esistere, e il `batch_size` scritto
nel `run_id` descrive una leva che non ha mosso niente.

Su `typed-decisions` il batching è reale (le domande vengono dalla riga del dataset: 400 stati → 4
gruppi da 100), e lì `--fast` è invece **0.80x, cioè più lento** (27.08 → 33.82 ms/stato, campioni
423-446 contro 530-548). La leva si inverte col batch: col batch scarso il fusion vince, col batch
pieno i CUDA graph per shape costano più di quanto guadagni.

## La trappola del campione, che stavo per pubblicare

La prima passata di questo esperimento girò con `--limit 300` e diede **ASR 0.08**, accuracy che
**sale** sotto attacco, `OER@0.75` 0.02. Letto così è "la configurazione non chiude il buco", e
sembrava un risultato. Non lo era: **a 300 casi l'attacco non ha dove atterrire**. A N=2974 dà
0.973. La stessa architettura di codice, lo stesso attacco, e un numero che cambia di 0.89.

Il denominatore dichiarato nella riga è la difesa: `asr_denominator` = 2277 a N pieno contro 225 a
N=300, e su 225 decisioni un ASR è un aneddoto. **Ogni run di questo protocollo va a N pieno**, e
se serve un sottoinsieme va detto nella riga, perché altrimenti il numero si può leggere come una
proprietà della configurazione quando è una proprietà di N.

Nota anche che l'attacco in coda su `laya-multilingual` a K=4 con 300 casi dava 0.08 mentre a
N pieno dà 0.97: la stessa ragione, non una configurazione che indebolisce.

## Cosa NON contiene

Non è una griglia di difese: non ci sono guardrail, filtri o prompt injection. Solo knob di
esecuzione. La domanda è se qualcuno di questi chiude il buco **da solo**. La calibrazione resta
l'unico intervento che lo chiude, e lo chiude solo dove il checkpoint spedisce una temperatura fuori
range ([[E16-calibrazione-difesa]]).

## Secondo checkpoint: stessa griglia, verdetto opposto sul guardrail

Ripetuto su `laya-typed-decisions`, lo stesso attacco e la stessa griglia. Dodici righe in
tutto, sei per checkpoint, verificate con deviazione `0.000e+00`.

| checkpoint | ASR appaiato (3 calcoli distinti) | escursione | CI 95% | rapporto | `OER@0.75` |
|---|---|---|---|---|---|
| `laya-multilingual` | 0.9731 / 0.9727 / 0.9722 | 0.000881 | 0.0134 | **15.2x** | 0.889--0.890 |
| `laya-typed-decisions` | 0.9792 / 0.9792 / 0.9797 | 0.000534 | 0.0131 | **24.5x** | 0.000 |

**Nessuna delle dodici chiude l'attacco**: l'ASR resta sopra 0.97 ovunque. E sul guardrail la
differenza è **fra checkpoint, non fra configurazioni**: `laya-multilingual` tiene
`OER@0.75` a 0.89-0.90 su tutta la griglia, `laya-typed-decisions` a 0.000 su tutta la griglia.
Su ciascuno dei due il valore è costante, e l'escursione fra configurazioni è 15 e 25 volte più
piccola dell'incertezza campionaria.

**Perché divergono sull'OER, e qui ho corretto una spiegazione che avevo dato per buona.**
Avevo attribuito lo `OER@0.75 = 0.000` di `laya-typed-decisions` alla temperatura fuori range
(`choice:11+ = 0.1006`, clampata a 0.5). È **falso**: a $K=4$ il bucket è `choice:3-5`, che
spedisce 1.7602 e non viene clampata. Il clamp non c'entra, e la spiegazione vera è un'altra.

Il confronto delle confidenze dice cosa succede davvero. A $K=4$, su `laya-typed-decisions`, la
confidenza è **bassa ovunque**: mediana 0.095 sulle risposte corrette al pulito e 0.038 sugli
errori al pulito (massimo 0.330). Sotto attacco la confidenza sugli errori sale a 0.354, ma da
una base così bassa che nessuno dei 2935 errori raggiunge 0.75. Su `laya-multilingual`, che a
$K=4$ parte da 1.0000, le mediane sono 0.837 (corrette) e 0.372 (errori) al pulito e 0.999 sugli
errori sotto attacco: l'attacco **spinge la confidenza degli errori sopra quella delle risposte
giuste**.

Quindi l'attacco c'è ed è forte su entrambi (ASR 0.979 e 0.973, accuracy che crolla a 0.013 e
0.021), ma **non produce errore convinto sul primo perché quel checkpoint non è mai convinto**:
la sua temperatura di 1.76 è così conservativa che nemmeno la risposta giusta arriva a 0.75.
Il guardrail a 0.75 non lo "buca": su quel checkpoint non ha niente su cui agire, perché è
tarato dove il modello non guarda. E questo è un punto cieco più forte della firma dell'errore
convinto: non basta che il guardrail sbagli l'attacco, basta che non guardi il posto giusto.

Quindi la versione corretta della tesi, su due checkpoint: **nessuna configurazione di
esecuzione chiude l'attacco, e la differenza nel guardrail è una proprietà del checkpoint che
l'attacco incontra, non una messa a punto che si può scegliere.**

## Il limite onesto

Dodici righe su **due** checkpoint della stessa famiglia, **un** dataset, **una** K, **un**
template. Il finding è sulla direzione e sulla soglia (nessuna configurazione attraversa 0.50),
non sull'ordine fine fra le righe, e resta più piccolo dell'incertezza campionaria su entrambi.

I due checkpoint sono **della stessa famiglia architetturale**: la generalità cross-architettura
resta quella di E15, su una coppia. Quello che i due aggiungono qui è che la griglia di
configurazione è neutra su *diversi* comportamenti del guardrail (0.89 e 0.00), il che è un
argomento più forte del caso singolo: se la configurazione contasse, il guardrail si muoverebbe
su entrambi, e invece resta fermo su ciascuno al suo valore.

## Note collegate

[[E03-option-injection]] · [[E16-calibrazione-difesa]] · [[E15-quantizzazione]] ·
[[performance-velocita]] · [[metriche-attacco]] · 
