---
tags: [esperimento, strutturale, token-budget]
stato: fatto
data: 2026-09-27
---

# E02 — Collasso dell'accuracy con K opzioni crescente

## Ipotesi

Se il numero di opzioni di una domanda `choice` cresce, l'accuratezza del modello crolla perché
il budget totale della testa (`head_max_len`) è fisso. Con K opzioni ciascuna riceve circa
`(head_max_len − overhead) / K` token, e la rappresentazione semantica delle singole alternative
si degrada. Vedi [[token-budget-head]].

## Setup

- checkpoint: `convaiinnovations/laya-typed-decisions`, `head_max_len=256`, `max_len=1024`
  (`results/e01_third_checkpoint.json`)
- budget stato: 991 token sul set di domande **non espanso** (`results/E00_setup.json`,
  `checks.truncation.state_budget_from_empty`). Espanso a K opzioni la testa cresce e il budget cala
- **E02 però non registra i token per K**: il suo store ha accuracy, ECE, Brier, OER e latenza,
  più `asr` da K=4 in poi (K=2 è il riferimento pulito e non ne ha), e nessun conteggio di token
- dispositivo: RTX 3080, Python 3.14.7, laya 0.3.11, torch 2.14.0+cu130 (da `env` nello store)
- dataset: [[typed-decisions-dataset]] test, N=400 casi / 2000 decisioni per K
- pilot: N=5 (25 decisioni), cinque run del 2026-09-27T16:47 (`76e66f886bc14f11` K=2,
  `abea730d1a337d30` K=4, `daba3d05fdb4a556` K=8, `555ce16452a8e015` K=16,
  `60913a003cb12a31` K=32). Sono in `results/E02_option_count/runs/` ma **non referenziati** da
  `results/E02_option_count.json`, che contiene solo le cinque righe N=400
- comandi: i config dei run archiviano `experiment`, `checkpoint`, `k`, `n_cases`, `dataset` e
  **non registrano `limit` né `seed`**. Il comando esatto non è ricostruibile dallo store; i run
  N=400 sono del 2026-09-30T08:41-08:42

## Metodo

Per ogni caso di test le domande `choice` vengono espanse a esattamente K opzioni
(K = 2, 4, 8, 16, 32), gold sempre presente. I distrattori vengono da una pool fissa di 15
coppie label + descrizione in `expand_choice_question`, poi da `alt_0, alt_1, …` ("alternative
action i") fino a riempire K. Nessun LLM esterno e nessun RNG: a ogni K l'operatore è una
funzione pura della riga, quindi l'unica variabile dichiarata è K, e nessun seme da mettere
nel `run_id`.

**A K=32 la pool non basta**: le opzioni placeholder `alt_i` sono **da 12 a 14** sulle 32, e non
una costante. Contate sui nomi delle opzioni dentro `probs[q]` in
`results/E02_option_count/runs/` di tutti i run a K=32 (`len(probs[q])` non serve: dà
sempre 32): **12** placeholder su 200 domande (18-20 opzioni native), **13** su 310 (19 native),
**14** su 100 (18 native). Le 1415 domande `noul`/`score` a 2 e 4 opzioni non hanno placeholder. Il drop di 13.4pp a K=32 include
quindi l'effetto di 12-14 opzioni prive di contenuto, oltre ai 17-19 distrattori con testo
verace. Le domande `noul` e `score` restano invariate a ogni K.

Valutazione con `layaadv.evaluate.evaluate` e checkpointing `RunStore` per variante. I 400
casi sono gli stessi di E01, stessi `id` nello stesso ordine in `preds.jsonl` (verificato
record per record), quindi le `noul` e le `score` di questa curva sono confrontabili con quelle
del pulito di E01.

## Risultati

`results/E02_option_count.json` · N=400 casi / 2000 decisioni per K, `n_cases: 400` in ogni
config, `run_id` `df3b6e785ec23f27` (K=2), `1ff51cb3edbddab8` (K=4), `83b86a5ab5d4ce4b`
(K=8), `45870dd37ad125f0` (K=16), `7945522eb2423144` (K=32). Nessun seme nei config, quindi
nessuna replica dichiarata: la curva è **single seed per costruzione**, non per scelta.
Arrotondamento metà per eccesso. I numeri a N=5 che erano in questa tabella sono stati
sostituiti: su 25 decisioni non erano distinguibili.

| K | accuracy | drop vs K=2 | `choice` | `noul` | `score` | OER@0.75 |
|---|---|---|---|---|---|---|
| 2 | 0.8230 | — | 0.9183 | 0.857 | 0.726 | 0.0000 |
| 4 | 0.7735 | 4.95 pp | 0.7533 | 0.857 | 0.726 | 0.0000 |
| 8 | 0.7615 | 6.15 pp | 0.7150 | 0.855 | 0.726 | 0.0005 |
| 16 | 0.7590 | 6.40 pp | 0.7050 | 0.857 | 0.726 | 0.0005 |
| 32 | 0.6890 | 13.40 pp | 0.4717 | 0.857 | 0.726 | 0.0005 |

Provenienza: `results/E02_option_count.json`, `results[]`, una riga per K.
Checkpoint `convaiinnovations/laya-typed-decisions` unico per tutte e cinque, N=400 casi /
2000 decisioni per riga, dataset `typed-decisions/test`, `subset_hash` e `dataset_sha256`
uguali a quelli di E01, **seme assente**. `accuracy` = `accuracy`,
`choice`/`noul`/`score` = `accuracy_by_primitive`, `OER@0.75` = `oer["0.75"]`.

Il drop è `100 × (accuracy(K=2) − accuracy(K))` (`metrics.core.drop`, che arrotonda a 2
decimali), non una differenza riportata a mano: 100 × (0.823 − 0.7735) = **4.95**,
100 × (0.823 − 0.7615) = **6.15**, 100 × (0.823 − 0.759) = **6.40**,
100 × (0.823 − 0.689) = **13.40**.

**Il riferimento `K=2` non è la baseline di E01.** Le `choice` native hanno 4 opzioni in 400
casi e 5 in 200 (contato su `results/E01_preds.jsonl`: 1200 record = 400 × 3 checkpoint, quindi
1200/3 = 400 domande con 4 e 600/3 = 200 con 5 per checkpoint); qui sono 2 in tutte le 600. Per
questo la riga K=2 (0.8230) sta 5.6pp sopra la baseline di E01
(`100 × (0.823 − 0.767) = 5.6`, baseline in `results/E01_baseline.json`) e il drop va letto solo
contro K=2: contro il pulito di E01 il drop a K=32 sarebbe 7.8pp (`100 × (0.767 − 0.689) = 7.8`).

Il collasso è tutto nel `choice`. `score` è costante a 0.726 (581/800 corrette a ogni K) e
`noul` si muove di una decisione su 600 (0.857 → 0.855 a K=8 → 0.857, cioè 514/600 contro
513/600). Più netto: a K=32 le etichette `noul` e `score` sono **identiche** a quelle del run
pulito di E01 sugli stessi 400 casi (0/600 e 0/800 diverse, ricalcolato da
`results/E01_preds.jsonl` contro il run K=32). Quindi il collasso è **meccanico, non semantico**:
l'affamamento delle opzioni non toglie comprensione, toglie spazio dove scrivere le alternative.

Anche il conteggio torna: a K=32 le 269 decisioni che K=2 azzeccava e K=32 sbaglia
(`asr = 269 / 1646 = 0.1634`, denominatore = decisioni corrette a K=2) sono tutto il calo:
`1646 − 269 + 1 = 1378 = 0.689 × 2000`, con **una sola** decisione che passa da torta a giusta.
Espandere le opzioni rompe, non aggiunge.

## Verdetto

- **Su N=400 il drop è 13.4pp a K=32, non 28pp.** Il 28pp del pilot
  (`100 × (0.76 − 0.48) = 28`, run `76e66f886bc14f11` e `60913a003cb12a31`, N=5) era rumore su
  25 decisioni. La soglia di 20pp **non è raggiunta**: il solo attacco del budget non basta su
  questo dataset, e la curva è molto più piatta di quanto suggeriva il pilot.
- La degradazione non è lineare: piatta da K=4 a K=16 (4.95 → 6.40pp, cioè 1.45pp in più),
  poi il salto a K=32 (13.40pp complessivi). La saturazione è tra K=16 e K=32, come nel
  pilot, ma l'ampiezza era sovrastimata di 2x (`28 / 13.40 = 2.09`).
- **Il collasso è solo del `choice`**: `score` costante a 0.726 su 800 decisioni, `noul` al più
  una decisione su 600, etichette finali identiche a quelle del pulito. Meccanismo meccanico
  (budget della testa), non perdita di comprensione.
- OER ≤ **0.0005** a ogni K, cioè **1 decisione convinta su 2000** al massimo, e confidenza
  media sul torto 0.2847 a K=32 (`mean_conf_wrong` = 0.28465900): gli errori non sono convinti,
  il guardrail regge.

## Tesi

Il budget finito della testa è una vulnerabilità strutturale: non serve l'ingresso semantico
per degradare le prestazioni, basta affamare le opzioni. Per questo Jev regge 255 opzioni
out-of-the-box ([[jev]]) mentre Laya ha già perso 13.4pp a K=32.

## Stato e prossimo passo

L'attacco è black-box puro (nessuna modifica dei pesi), deterministico e riproducibile. Come
submission study serve un dataset con K >= 30 opzioni (Banking77-like). Il cross con E04
(stato gonfiato più K alto) **non è misurato**: in `results/` non esiste nessuna run con K e
filler insieme. Le versioni precedenti di questa note riportavano "a K=40 l'accuracy va
15.2% → 4.7% con lo stato inflato": quel numero **non è ricalcolabile dal repository** e non
viene tenuto. Il cross resta da fare ([[token-budget-head]]).

## Output

- `results/E02_option_count.json` (5 righe N=400)
- `results/E02_option_count/runs/<run_id>/preds.jsonl` (400 record per ciascuna delle 5 righe
  N=400, 5 record per ciascuna delle 5 righe N=5 del pilot)

## Note collegate

[[token-budget-head]] · [[architettura-laya]] · [[metriche-attacco]] · [[esperimenti]]
