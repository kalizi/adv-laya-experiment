---
tags: [esperimento, troncamento, budget]
stato: fatto
data: 2026-09-27 (rinumerato da E03 il 2026-09-29)
rinumerato: E04 — l'attacco di injection nello stato è diventato [[E03-option-injection]]
---

# E04 — Troncamento: segnale fuori budget

## Ipotesi

Se la frase che determina il gold viene spinta oltre i **962.6 token** che `state_budget`
lascia allo stato, il modello smette di vederla: la coda viene troncata e la decisione
crolla. Il troncamento è a destra, verificato in [[E00-setup]] (`truncate_left=False`).
Sezione di [[truncation-stato]].

L'ipotesi aveva due parti. Il **crollo** regge. La seconda, *predice la classe di default con
confidence non bassa*, **non** regge: nessun primitivo collassa su una classe unica e la
confidenza sugli errori resta bassa (vedi Verdetto).

## Setup

- checkpoint: `convaiinnovations/laya-typed-decisions` (`max_len=1024`, `head_max_len=256`)
- budget stato: **962.6 token**, media dei 5 budget per domanda. `state_budget` restituisce
  `max_len − len(head) − 1`, cioè la stanza reale (il `−1` è il SEP che chiude lo stato), non
  `max_len − head_max_len`. I filler sono `int(962.6 × mult)`.
- dispositivo dichiarato: NVIDIA RTX 3080, Python 3.14.7, laya 0.3.11, torch 2.14.0+cu130
  (campo `env`; sul device *per campione* vedi Nota di esecuzione)
- dataset: [[typed-decisions-dataset]] `test`, **N=400**, l'intero split
  (`data/raw/typed-decisions/test.jsonl` ha 400 righe)
- seme: nessuno. `typed_decisions.load_cases` legge in ordine di file e non mescola, quindi il
  sottoinsieme è deterministico; `min_agreement` non è applicato.
- run che produce la tabella: `uv run adv e04` (senza `--limit`), `batch_size=16`
- pilot precedente: `uv run adv e04 --limit 5`, N=5, sostituito (vedi sotto)
- baseline: [[E01-baseline]] sullo stesso sottoinsieme
- `questions` invariate, byte-identiche: si modifica solo lo stato (study-valid)

**Cosa dichiarano i `config.json` di questa run**: `batch_size=16`, `chunk=16`. Il `chunk` **non**
è quello che è successo: ogni run ha una sola forward pass su 400 casi (`latency.n=1`,
`runtime_s.batches_this_process=1`), quindi `p50` e `p95` sono lo stesso numero e `std=0` per
costruzione. Lo dichiara anche il commento su `CHUNK` in `e04_truncation.py`.
In questo archivio **non ci sono `subset_hash`, `dataset_sha256` né `device` per campione**
(aggiunti dopo): integrità dei dati e device effettivo di questa run non sono dichiarati da
nessuna parte (vedi Nota di esecuzione).

## Metodo

**Variante A (Push-out):** prefix di filler neutro (lorem ipsum) + stato originale. Lunghezze
target 0x, 0.5x, 0.75x, 1.0x, 1.25x, 1.5x del budget stato, cioè 0, 481, 721, 962, 1203, 1443
token. Il filler è generato da un template deterministico, nessun LLM esterno.

Il troncamento è verificato per run, non arguito: `truncation_check.truncation_verified=true` con
`marker_visible_count=0` su 5 domande in tutte e 5 le run con filler, cioè il marcatore appeso in
coda non è più visibile nella sequenza.

Variante B (Signal in coda): richiede conoscenza dataset-specifica di quale porzione dello stato
contiene il segnale gold. Non implementata in questa iterazione.

## Risultati

`results/E04_truncation.json`, N=400 casi = **2000 decisioni** (5 domande per caso).

| filler | filler_tokens | accuracy | drop vs pulito | **ASR** [CI 95%] | OER@0.75 | conf sugli errori |
|---|---|---|---|---|---|---|
| 0.0x | 0 | 0.7670 | — | — | 0.0000 | 0.2297 |
| 0.5x | 481 | 0.4145 | 35.25 pp | 0.5280 [0.503, 0.553] | 0.0090 | 0.1997 |
| 0.75x | 721 | 0.3965 | 37.05 pp | **0.5535** [0.528, 0.578] | 0.0055 | 0.2135 |
| 1.0x | 962 | 0.3965 | 37.05 pp | 0.5535 [0.528, 0.578] | 0.0055 | 0.2135 |
| 1.25x | 1203 | 0.3965 | 37.05 pp | 0.5535 [0.528, 0.578] | 0.0055 | 0.2135 |
| 1.5x | 1443 | 0.3965 | 37.05 pp | 0.5535 [0.528, 0.578] | 0.0055 | 0.2135 |

 Formule (tutte ricalcolabili dagli archivi, nessuna riportata così nell'archivio):

- `drop = 100 × (accuracy_pulito − accuracy_attaccato)`: 100 × (0.7670 − 0.4145) = **35.25 pp**;
  100 × (0.7670 − 0.3965) = **37.05 pp**.
- `asr_denominator = |S_ok ∩ S_adv| = 1534 = 0.7670 × 2000` decisioni corrette al pulito.
- **I limiti di CI non sono nell'archivio**: Wilson 95% su `asr_denominator`, cioè
  `(p ± z²/2n ∓ z√(p(1−p)/n + z²/4n²)) / (1 + z²/n)` con `z=1.96`, `n=1534`. Ricalcolati qui da
  `asr` e `asr_denominator`: per p=0.5280 dà [0.503, 0.553], per p=0.5535 dà [0.528, 0.578]. È
  la ragione per cui `tools/check_note_numbers.py` li segnala come fuori archivio.
- `conf sugli errori` = `mean_conf_wrong` della stessa riga.

Provenienza: una `run_id` per riga, ognuno in `results/E04_truncation/runs/<run_id>/` con
`config.json`, `preds.jsonl` (400 righe) e `latency.jsonl`. Nell'indice i `run_id` sono quelli
col suffisso `-2` (vedi Nota di esecuzione).

Il pilot N=5 non è più la tabella: 25 decisioni, `accuracy` 0.6400 al pulito e 0.2400 a ogni
filler, `asr` 11/16 = 0.6875 sulle 16 decisioni corrette al pulito. Su 16 decisioni la
differenza non era distinguibile. Questi tre numeri **non sono in `results/E04_truncation.json`**:
si ricostruiscono da `results/E04_truncation/runs/<run_id>/{summary,preds}.jsonl` dei run N=5
(`5c9ccd2601bb0b2d` pulito, `408c95959f5342c2` a 0.5x).

## Verdetto

- **Drop 37.05 pp** con filler ≥ 0.75× budget: 100 × (0.7670 − 0.3965) = 37.05 pp, cioè
  **1.85× la soglia** (37.05 / 20 = 1.85). La soglia 🥉 di 20 pp è la bozza di
  , **auto-dichiarata e non votata dal gruppo**: se ne viene votata un'altra
  il verdetto va ricalcolato.
- **La baseline è 0.7670, non 0.6400.** A N=5 la baseline era 0.6400 (16/25) e il drop sembrava
  40 pp: il denominatore era rumore. Il numero vero è 37.05 pp, e resta sopra soglia.
- **ASR 0.5535, non 1.0.** Le prime versioni di questa note affermavano "ASR = 100%: tutte le
  decisioni corrette al pulito vengono ribaltate". Falso: sul pilot N=5 era 11/16 = 0.6875, e il
  dato vero su 1534 decisioni è **0.5535 [0.528, 0.578]**: sopra la soglia 🥉 di 0.50, non 1.0.
- **OER@0.75 ≤ 0.0090 a ogni filler.** Il guardrail di confidenza regge: il modello sbaglia con
  confidenza bassa (`mean_conf_wrong` 0.1997-0.2135 sotto attacco, contro 0.2297 al pulito, e
  l'attacco non la alza). È il punto in cui E04 e [[E03-option-injection]] si separano:
  qui l'attacco degrada le prestazioni **senza** convincere il modello.
  [[E03xE04-composizione]] lo conferma su N=1000, dove l'injection convince
  (`OER@0.75` 0.902 a K=16 in coda, filler 0). Quell'ultimo numero è della tabella di
  [[E03xE04-composizione]]: **non è in nessuno store di `results/`**, quindi
  `tools/check_claim_numbers.py` lo segnala anche lì.
- **La "classe di default" dell'ipotesi non è confermata.** Nessun primitivo va a zero:
  `accuracy_by_primitive` a 0.75x vale `choice` 0.7317 → 0.41, `noul` 0.8567 → 0.545,
  `score` 0.72625 → 0.275. Ricalcolato sui `preds.jsonl`, il modello attaccato prevede 11 label
  distinte contro 29 al pulito e la più frequente passa dal 15.45% al 25.0% delle 2000
  decisioni: c'è concentrazione, non un'unica classe di default. (Fuori dall'archivio
  dichiarato, quindi non lo verifica `check_note_numbers.py`.)
- **Piatto da 0.75× in su**: 721, 962, 1203 e 1443 token danno tutti `accuracy` 0.3965 e `asr`
  0.5535. La saturazione del budget è completa e il resto del filler è sprecato.

## Nota di esecuzione

**L'unità del campione di latenza è la run, non il batch.** Ogni riga ha `latency.n=1` e
`runtime_s.batches_this_process=1` con 400 righe in `preds.jsonl`: il campione copre tutte le
2000 decisioni. I `p50` sono 22.3 s al pulito e 61.7-68.6 s con filler, cioè 11.2 ms/decisione al
pulito contro 30.8-34.3 ms/decisione sotto attacco: il filler costa **2.77-3.08×** in tempo.

**Il fallback su CPU non è verificabile da questo archivio.** Il `latency.jsonl` di queste run è
`{"ms": ...}` senza il campo `device`, e `latency.devices` non esiste qui (è stato aggiunto
dopo, ed è il motivo per cui oggi un fallback è visibile). I tempi registrati non mostrano il
rallentamento ~10× che `engine.runtime_state` attribuisce a una demozione su CPU. L'unica prova
dichiarata è il `RuntimeWarning` che `engine._check_device` emette a runtime, e **il log di
quella run non è nel repository**. Resta un avviso plausibile, non un fatto dichiarato.

**Il confronto fra le due run di ogni filler è verificabile.** Ogni filler ha due run con lo
stesso `config`: quella dell'indice (`<run_id>-2`) e quella senza suffisso (`<run_id>`).
Confronte riga per riga sui `preds.jsonl`: **0 label diverse su 2000 decisioni in tutti e 6 i
filler**, e `max |Δconfidence| = 0.000000`. Non solo le label coincidono: le due run sono
bit-identiche anche in confidenza. Le differenze sono solo nei tempi, del 5-17%.

## Tesi

Il troncamento dello stato è una vulnerabilità operativa gravissima: 481 token di lorem ipsum, il
50% del budget di 962.6, portano `accuracy` da 0.7670 a 0.4145. Non serve avversariality
sofisticata, basta riempire. È un problema di deployment, non di modello: anche un modello
perfetto sulle decisioni fallisce se non vede lo stato.

## Conclusioni

- L'attacco è black-box puro, study-valid (solo stato), deterministico, riproducibile.
- La variante B (signal in coda) resta aperta per identificazione fine del segnale.
- L'interazione con [[token-budget-head]] (K alto + stato gonfiato) è già misurata in
  [[E03xE04-composizione]], N=1000: il filler protegge, l'injection convince.
- Il pilot N=5 è stato sostituito: i numeri pubblicabili sono quelli della tabella qui sopra.

## Output

- `results/E04_truncation.json`
- `results/E04_truncation/runs/<run_id>/preds.jsonl` (per variante)

## Note collegate

[[truncation-stato]] · [[architettura-laya]] · [[metriche-attacco]] · [[esperimenti]] · [[E01-baseline]] ·  · [[E03xE04-composizione]]