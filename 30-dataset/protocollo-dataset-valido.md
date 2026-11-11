---
tags: [dataset, protocollo, validita]
---

# Protocollo dataset valido

Un attacco senza dataset valido non conta nella study.

## Requisiti minimi

### 1. Ground truth chiaro
Ogni esempio ha: `state`, `questions` (identiche alla baseline!), `gold`. Le `questions` **non si toccano** in fase di attacco → .

### 2. Split fissato prima di iniziare
- scaricare e congelare (hash dei file o commit)
- **mai** scegliere gli esempi da attaccare in base al risultato dell'attacco (selection bias)
- se si filtrano gli ambigui (via `label_agreement`), farlo **prima**, dichiarare il filtro

### 3. Provenienza e licenza
| dataset | origine | licenza (da verificare al download) |
|---|---|---|
| LocalLLaMA/typed-decisions | HF | verificare card |
| AG News / DAIR / SST-5 / XNLI / Banking77 | HF | verificare card |
| `pngwn/typed-decisions-v2` | HF | verificare card |

Regola: se non puoi ripubblicarlo, non è un dataset valido per questa study (regola della pubblicità totale).

### 4. Baseline misurata sullo *stesso* split
`baseline_acc`, `baseline_ECE`, `baseline_OER` girati dal partecipante sul proprio hardware, non citati dal model card. → [[E01-baseline]]

### 5. N esempi dichiarati
- minimo consigliato per un attacco "serio": **N ≥ 100** esempi attaccati
- ASR con N < 20 è aneddoto (intervallo di confidenza enorme)

## Entry compilate (quello che è stato usato davvero)

### `typed-decisions-test` — E01 (baseline), E02, E04
```yaml
nome: typed-decisions-test
hub: LocalLLaMA/typed-decisions
licenza: apache-2.0            # dal manifest, data/manifest.json
split: test
n: 400                         # 2000 decisioni (5 domande per caso)
sha256: 337086d01dfa687105078288f0862d3c40d7e914483febc8323be948dc9590bb
filtro_pre: nessuno            # min_agreement = null in tutti i run
questions_invariate: true      # E02 e' diagnostico e modifica le options: non e' submission
```

### `massive-it-test` — E03 (option injection)
```yaml
nome: massive-it-test
hub: https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz
licenza: cc-by-4.0             # LICENSE nella cartella,Copyright Amazon.com
split: test (partition)
n: 2974                        # 60 intent nel dataset, 59 presenti in test
                               # (cooking_query esiste solo in train)
sha256: d5125df760586376fdcbbda4dd5b0b8fa408dfb6d6219faa8a6dea4d3f74c194
filtro_pre: nessuno
questions_invariate: true      # E03 cambia solo lo stato
head_budget: K<=42 con head_max_len=256; <=24 con 192 (vedi [[token-budget-head]])
```

Entrambi gli hash sono in `data/manifest.json` e coincidono con i `dataset_sha256` nei
`config.json` di ogni run sotto `results/`: lo split è congelato prima di misurare.

## Controlli anti-trucco
- [x] Lo stato avversario ha lo **stesso gold** di quello pulito? E03: `case.gold` è riusato
      identico, l'attacco ricostuisce solo `state` → sì, per costruzione
- [x] Gli esempi scelti erano già sbagliati *prima* dell'attacco? Esclusi dal denominatore
      ASR: si conta solo sulle decisioni corrette al pulito (→ [[metriche-attacco]])
- [x] Le `questions` sono byte-identiche? Verificato: le opzioni sono derivate da un seed
      per riga (`f"{seed}:{id}:{testo}"`), quindi pulito e avversario costruiscono le stesse
      domande; il test `test_option_set_does_not_depend_on_limit` lo copre
- [x] Il drop è relativo alla baseline di quel dataset, non assoluto? E03 riporta sia
      `accuracy_clean` (misurata sullo stesso subset) sia il drop rispetto a quella

## Note collegate
[[metriche-attacco]] ·  · [[typed-decisions-dataset]] ·  · [[E03-option-injection]]
