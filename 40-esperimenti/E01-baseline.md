---
tags: [esperimento, baseline]
stato: fatto
data: 2026-09-27
---

# E01 — Baseline pulita su typed-decisions

## Velocità e tradeoff

`results/E01_baseline.json`: N=400 casi / 2000 decisioni per checkpoint, `batch_size=16`,
`chunk=32`, `subset_hash f45512a0342cd6de…`, `dataset_sha256 337086d01dfa6871…`, stesso
dataset e stesso device (CUDA, RTX 3080) per tutti e tre. I `run_id`, nell'ordine della
tabella, sono `446605d24277a4dc`, `6824e1aad1add374`, `7bb04ea6538418e1`. **Nessun seme**: il
loader di typed-decisions legge in ordine di file e non estrae numeri casuali, quindi l'ancora
di riproducibilità è `subset_hash` + `dataset_sha256`.

| checkpoint | acc | OER@0.75 pulito | ms/decisione | decisioni/s |
|---|---|---|---|---|
| `laya-typed-decisions` | **0.767** | **0.0000** | 10.759 | 93.0 |
| `laya` | 0.3595 | 0.0435 | 10.424 | 95.9 |
| `laya-multilingual` | 0.3505 | 0.1430 | **5.187** | **192.8** |

Provenienza: `results/E01_baseline.json`, `checkpoints[]`, una riga per checkpoint, `acc` =
`accuracy`, `OER@0.75` = `oer["0.75"]`. CUDA (RTX 3080) per tutte e tre.

Le due colonne di velocità non sono campi grezzi: lo store le calcola da
`latency.total_ms` e da `n_decisions` (`metrics.core.throughput`). Le formule sono
`ms/decisione = latency.total_ms / 2000` e `decisioni/s = 1000 × 2000 / latency.total_ms`,
e tornano cifra per cifra sui tre `run_id`: 21517.35 / 2000 = **10.7587**,
1000 × 2000 / 21517.35 = **92.95**; 20847.35 / 2000 = **10.4237**; 10374.27 / 2000 = **5.1871**.
Il divisore sono le 2000 decisioni **dentro i batch cronometrati**, non il totale del run.
`tools/check_claim_numbers.py` non li può verificare: raccoglie solo valori nell'intervallo
`[0, 1]`, quindi ogni numero in millisecondi gli resta fuori per costruzione.

Il più accurato è il più lento: 0.767 a 10.759 ms contro 0.3505 a 5.187 ms. Il più veloce è
**2.07x** più rapido (`10.7587 / 5.1871 = 2.07`), con l'accuracy peggiore e già il **14.3%** di
errori convinti al pulito (`oer["0.75"] = 0.143` sulle 2000 decisioni). Ed è anche il più
attaccabile: `ASR 0.9981` su MASSIVE-it, da `results/e03_third_checkpoint.json` (campo `e03.asr`
= 0.9981412639405205; K=16, template `append` in coda, N=1000 casi, 538 decisioni al
denominatore; vedi [[E03-option-injection]]). L'angolo "compra quello veloce" del Pareto è
anche quello "è già rotto". Metodo, device e cosa c'è da ottimizzare: [[performance-velocita]].

## Ipotesi

I nostri numeri sul nostro hardware devono avvicinarsi a quelli pubblicati: **0.766** di
accuracy per `laya-typed-decisions` sul test split. Se non si avvicinano, **tutti i drop
successivi sono inaffidabili**.

### Numeri citati, non misurati qui

I numeri pubblicati che questa nota cita arrivano da fuori dagli store: nessuno dei tre
gruppi qui sotto si ricalcola da `results/`.

- `0.766` accuracy pubblicata di `laya-typed-decisions` sul test ([[typed-decisions-dataset]])
- `0.362` e `0.342` accuracy zero-shot pubblicate di `laya` e `laya-multilingual` ([[laya]])
- `33 ms` latenza di riferimento dal model card

Tutto il resto si ricalcola da `results/E01_baseline.json` o
`results/e01_third_checkpoint.json`. `uv run python tools/check_note_numbers.py E01` segnala
**sei** decimali fuori archivio, ed è l'esito atteso: sono tutti e sei spiegati qui.

- `0.766`, `0.362`, `0.342` (tre): pubblicati, non misurati qui, lista sopra.
- `0.9981` (uno): archivio E03, `results/e03_third_checkpoint.json`, citato in "Velocità".
- `0.7469` (uno): dichiarato non ricalcolabile in "Note di implementazione". Il suo gemello
  `0.7500` **non** viene segnalato, ma è un falso verde: il checker lo abbina al `brier` di
  `laya` (0.750041404865 arrotondato a 0.750) e non a nessuna run con `chunk=1`. Vanno letti
  con lo stesso caveat.
- `0.1006` (uno): il valore è vero ma vive nella **stringa** `load_warnings[0]`, quindi nessuno
  dei due checker lo può risolvere a un campo numerico. Dettaglio in "Risultati".
- I millisecondi (10.759, 10.424, 5.187 e le derivate) non compaiono in nessuna lista: quel
  checker raccoglie solo valori in `[0, 1]`.

## Setup
- checkpoint: `convaiinnovations/laya-typed-decisions` (target primario)
- dataset: [[typed-decisions-dataset]], split `test`, N=400 casi / 2000 decisioni
- questions: quelle del dataset, invariate
- dispositivo: [[configurazione-ambiente]] (CUDA, RTX 3080; `env` in `results/E01_baseline.json`)
- copertura: nessun attacco reale. `Cogensec/jailbreak-corpus` è in `REQUESTED`
  (`src/layaadv/datasets/download.py`) e `data/raw/jailbreak-corpus/` è vuota, quindi le righe
  di questa nota sono tutte su dataset pubblici già in `results/`

## Metodo
```python
# implementato: uv run adv e01 (src/layaadv/experiments/e01_baseline.py)
from datasets import load_dataset
import laya, json, hashlib

ds = load_dataset("LocalLLaMA/typed-decisions", split="test")
agent = laya.load("convaiinnovations/laya-typed-decisions")

rows = []
for ex in ds:
    ans = agent.predict(ex["state"], ex["questions"])["answers"]
    rows.append({"id": ex["id"], "pred": ans, "gold": ex["gold"]})

# metriche per tipo: choice=argmax, noul=P(true)>=0.5, score=argmax livello
# + Brier, ECE, OER (wrong & conf>=0.75)
```

Da calcolare **per checkpoint**, così la baseline è riusabile dagli altri:
1. `laya-typed-decisions` (target primario, pubblicato 0.766)
2. `laya` (zero-shot, pubblicato 0.362)
3. `laya-multilingual` (zero-shot, pubblicato 0.342)

### Metriche obbligatorie
| metrica | perché |
|---|---|
| accuracy (argmax) per primitive e totale | drop di riferimento |
| ECE | vedi [[rlcd-calibrazione]] |
| **OER** (wrong AND conf ≥ 0.75) | la soglia del "fischi per fiaschi" →  |
| Brier, soft accuracy | per confronto con il model card |
| p50 latency | confronto con i 33 ms dichiarati (attenzione: batch, non singola chiamata) |

### Righe da loggare
- sha256 del file del dataset
- `laya.__version__`, hash dei pesi
- device, dtype (fp32/bf16/fp16)

## Risultati
`results/E01_baseline.json` · N=400 casi / 2000 decisioni per checkpoint, `skipped: 0`,
`subset_hash f45512a0342cd6de…`, `batch_size=16`, `chunk=32`, seme assente (campo `seed` non
presente nei config dei tre `run_id`). Arrotondamento a 3 decimali metà per eccesso; `acc` è
riportato alla precisione dello store (0.767, 0.3595, 0.3505).

| checkpoint | acc | ECE | OER@0.75 | Brier | choice | noul | score |
|---|---|---|---|---|---|---|---|
| laya-typed-decisions | **0.767** | 0.417 | 0.0000 | 0.400 | 0.732 | 0.857 | 0.726 |
| laya | 0.3595 | 0.154 | 0.0435 | 0.750 | 0.283 | 0.485 | 0.323 |
| laya-multilingual | 0.3505 | 0.238 | 0.1430 | 0.891 | 0.293 | 0.498 | 0.283 |

Provenienza: `results/E01_baseline.json`, `checkpoints[]`; `acc` = `accuracy`,
`choice`/`noul`/`score` = `accuracy_by_primitive`, `OER@0.75` = `oer["0.75"]`.
Le tre righe hanno lo stesso `subset_hash` e lo stesso `dataset_sha256`: un solo dataset.

- Solo `laya-typed-decisions` è in range: distanza dal pubblicato 0.766 = **0.1pp**
  (`100 × (0.767 − 0.766) = 0.1`). Sugli altri due i pubblicati 0.362 e 0.342 diventano
  **0.3595** e **0.3505** misurati, entrambi entro **0.9pp**:
  `100 × |0.3595 − 0.362| = 0.25` e `100 × |0.3505 − 0.342| = 0.85`, cioè 0.9pp arrotondato.
- ECE peggiore sul target (0.417) pur con accuracy alta. I warning di temperatura upstream
  (`choice:11+ = 0.1006`, fuori range `[0.5, 5]`, clampata a 0.5) rendono la confidenza non
  calibrata: sono la **stringa** `load_warnings[0]` in `results/e01_third_checkpoint.json`, che
  porta il valore esatto `0.10058280825614929`. È l'unico numero di questa nota che nessuno dei
  due checker può risolvere a un campo numerico, perché sta dentro una stringa. Documentato e non
  fixato: il warning da solo non spiega il 0.417, che resta il peggiore dei tre (c'è anche su
  `laya`, che ha ECE 0.154, e su `laya-multilingual`, ECE 0.238, non c'è affatto).
- **Riproducibilità verificata**: i due run a config identico `9578ca3366a99f1a` e
  `9578ca3366a99f1a-2` (il secondo creato con `--force-new`) hanno `subset_hash`,
  `dataset_sha256`, `accuracy`, `ece`, `brier` e `oer` identici cifra per cifra, quindi le
  metriche sono deterministiche.

### Latenza (13 chunk da 32 casi, GPU-sincronizzata)

L'unità cronometrata è il **chunk**: `chunk=32` casi inoltrati in batch da 16
(`predict_batch`), quindi 13 campioni su 400 casi (`ceil(400 / 32)`). I valori sono i blocchi
`latency` dei tre `run_id` sopra, cioè gli stessi che producono `ms/decisione`.

| checkpoint | p50 | p95 | mean | std | campioni | runtime del processo |
|---|---|---|---|---|---|---|
| laya-typed-decisions | 1713 ms | 2507 ms | 1655 ms | 565 ms | 13 | 21.7 s |
| laya | 1733 ms | 2310 ms | 1604 ms | 539 ms | 13 | 20.9 s |
| laya-multilingual | 835 ms | 1206 ms | 798 ms | 293 ms | 13 | 10.4 s |

Provenienza: blocchi `checkpoints[].latency` di `results/E01_baseline.json`, cioè i tre
`run_id` della prima tabella; `runtime del processo` = `runtime_s.this_process`. N=400 casi,
seme assente, CUDA.

`runtime del processo` è `runtime_s.this_process` (include caricamento e metriche), non la
somma dei campioni: 21.67 s contro 21.52 s di somma campioni sul target.

Il rerun `--force-new` `9578ca3366a99f1a-2` ha un blocco `latency` più veloce (p50 1545 ms,
std 493 ms) ma **senza `total_ms`**: è una versione di `latency_stats` precedente, quindi da
lì `ms/decisione` non si ricava. La tabella prende quindi i numeri di
`results/E01_baseline.json`, gli unici che sostengono anche le due colonne di velocità.

Attenzione all'unità: qui il p50 è **per chunk di 32 casi**, non per singola chiamata. La
singola chiamata è misurata a parte in E00: **16.6 ms** (p50 16.63, std 0.36, n=100) su uno
stato sintetico, da `results/E00_setup.json` (`checks.latency`). I 33 ms del model card vanno
 confrontati su un caso reale di typed-decisions, che in una sessione non archiviata costava
**33.6 ± 0.7 ms**: quel numero **non è ricalcolabile dal repository**, perché la sessione non è
in archivio. Il caso reale è comunque molto più caro di quello sintetico, e questo si misura:
**1456 token di input in media** sui 400 casi reali (`results/E01_preds.jsonl`,
`usage.input_tokens` dei primi 400 record) contro **160** sullo stato sintetico di E00
(`checks.smoke.usage.input_tokens`).

## Verdetto
- Distanza dal pubblicato 0.766 = 0.1pp, dentro il budget di 3pp → **baseline accettata e
  congelata**. I 3pp sono una tolleranza dichiarata, non una misura. Nessuna indagine su dtype,
  troncatura o split.
- Checkpoint target principale: `convaiinnovations/laya-typed-decisions`. `laya` (0.3595) e
  `laya-multilingual` (0.3505) crollano a ~0.35, non sono baseline utilizzabili per i drop.
- Tutti i drop successivi sono relativi a 0.767, non al 0.766 pubblicato.

## Output
- `results/E01_baseline.json` (metriche + latenza + ambiente)
- `results/E01_preds.jsonl` (1200 record = 400 × 3 checkpoint, 6000 decisioni → base per ASR futuri)
- `results/E01_baseline/runs/<run_id>/latency.jsonl` (sidecar: un campione per chunk, sopravvive al resume)

## Note di implementazione
Il blocco `latency` arriva da `evaluate`, che cronometra ogni `predict_batch` con
`sync_device()` (senza, si misura il launch e non l'inferenza). I campioni
vanno su un sidecar `latency.jsonl` e non in memoria: se fossero solo in RAM, un
run interrotto e ripreso riporterebbe i percentili dell'ultimo segmento e ogni
confronto fra run diventerebbe falso, e il test è in `tests/test_evaluate.py`.

`chunk` è la **granularità di checkpoint**, non un troncamento: nessun caso
viene scartato, cambiano solo la frequenza di flush e il numero di forward pass.
`chunk` e `batch_size` sono ora nel config del run, perché altrimenti due run con
composizione diversa condividerebbero lo stesso `run_id`.

L'unica coppia appaiata in archivio dice cosa la composizione del batch sposta davvero:
stessi 400 casi, stesso checkpoint, `chunk=32` in
`results/E01_baseline/runs/446605d24277a4dc` contro `chunk=16` in
`results/E04_truncation/runs/83f0def50219bdae-2` (filler 0) dà **0/2000 label diverse** e
accuracy 0.767 in entrambi (ricalcolato dai due `preds.jsonl`: `max` delle `probs` per argmax,
`P(true) >= 0.5` per `noul`). Si muovono solo le confidenze: 17/2000 diverse, al massimo
3.3e-3, abbastanza per spostare il Brier alla sesta cifra (0.3995585 contro 0.3995667), quindi
la sensibilità alla composizione del batch è dimostrata **sulle confidenze, non sull'argmax**.

Le versioni precedenti di questa nota citavano, per lo stesso punto, `chunk=1` a 0.7500 e
`chunk=32` a 0.7469 sugli stessi 64 casi. Quei due numeri **non sono ricalcolabili dal
repository**: nessuna delle 448 `config.json` in `results/*/runs/` ha `chunk=1` (i valori
presenti sono 32 su 407 config, 16 su 12 e campo assente su 29, run pre-fix), e la stessa
affermazione è nel docstring di `evaluate`
(`chunk=1 scored 0.7500 where chunk=32 scored 0.7469`).
Sono l'unica altra coppia che `tools/check_note_numbers.py E01` segnala come fuori archivio.

## Note collegate
[[typed-decisions-dataset]] · [[metriche-attacco]] · [[esperimenti]] · [[E00-setup]]
