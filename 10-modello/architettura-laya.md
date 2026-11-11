---
tags: [modello, architettura, modernbert]
---

# Architettura di Laya

## Stack
```
state (testo/email/ticket/JSON)
        │
        ▼
┌─────────────────────────────────────┐
│ Encoder: ModernBERT-large (395M)    │  ← fully fine-tuned, bidirezionale
│  - RoPE, 8192 max pos (uso: 512/1024)│
│  - attenzione alternata:            │
│    un livello globale ogni 3        │
│    (10 su 28 = 0.357, non 1/3)      │
│    18 locali (0.643)                │
│    local_attention=128 → raggiungi- │
│    bilità ±64 token                 │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│ Decision head (addestrato da zero)  │
│  - 2 layer Transformer              │
│  - option-marker scorer             │
│  - act/escalate head                │
└─────────────────────────────────────┘
        │
        ▼
softmax sulle opzioni → distribuzione + confidence
```
**Totale: 421M parametri.**

L'alternanza dell'attenzione è **misurata**, non assunta: letto dal modello caricato
(`tools/e22_attention_structure.py`, archivio in
`results/provenance/e22_attention_structure.json`), 28 livelli di cui **10 globali e 18
locali**. Il progetto diceva "1/3 global, 2/3 local": la frazione realizzata è **0.357**
perché `global_attn_every_n_layers = 3` su 28 livelli dà 10 globali, non 9.33. E la
finestra locale è `local_attention = 128`, che in transformers è una **mezza finestra**:
la maschera è `|query − key| ≤ 64`, quindi un livello locale raggiunge **±64 token** e uno
globale l'intera sequenza. La raggiungibilità è misurata sui pesi (probe a 0/20/60/120/200
token), non letta dalla documentazione.

## Come nasce la risposta: option markers
- Ogni opzione viene valutata al **suo `[MASK]` token**
- I hidden state sono estratti con `torch.gather` e passati allo scorer
- Softmax **solo sulle opzioni di quella domanda**
- Lo **spazio delle risposte è definito a runtime**: nuovi schemi senza retraining

### Conseguenze per gli attacchi
1. Il modello non "legge" il criterio come un LLM: fa match contestuale tra `[MASK]` e i token delle opzioni. → [[option-marker-injection]]
2. Le opzioni condividono un budget di token fisso → [[token-budget-head]]
3. La confidence viene dalla **forma** della distribuzione (entropy-based: `1 − H(p)/log(k)` su Laya; peak-based su Jev). Stessa distribuzione, confidence diverse → soglie da ricalibrare su Laya, non copiate da Jev.

## Budget
| checkpoint | max_len | head_max_len | token per la stato |
|---|---|---|---|
| `laya` (English) | 512 | 192 | ~320 |
| `laya-multilingual` | 1024 | 256 | ~768 |
| `laya-typed-decisions` | 1024 | 256 | ~768 |

- `head_max_len` copre **question + criteria + opzioni**
- `max_len − head_max_len` = budget per lo **stato**
- il troncamento avviene sullo stato (default) → [[truncation-stato]]
- si può alzare a runtime: `agent.cfg["head_max_len"] = 512`

## Una sola forward pass
Tutte le domande di una chiamata sono valutate insieme. Le latenze batched (7.2 ms/q a 10) derivano da lì.

## Router
Vedi [[routing-script-bypass]]. Rilevamento script/lingua in <0.5 ms **prima** della forward pass, poi dispatch al checkpoint. Motivo: il confidence del modello non segnala che il checkpoint non sa leggere quella lingua (Khmer 0.000 acc @ 0.952 conf).

## Note collegate
[[laya]] · [[primitive-typed]] · [[token-budget-head]] · [[truncation-stato]] · [[option-marker-injection]] · [[E22-attenzione-laya]]
