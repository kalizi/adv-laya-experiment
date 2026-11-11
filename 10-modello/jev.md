---
tags: [modello, jev, typesafe]
---

# Jev (TypeSafe AI)

Il modello chiuso che ha scatenato la study.

## Scheda
- **Casa:** TypeSafe AI (2 anni in stealth, lancio 15 settembre 2026)
- **Categoria:** [[system-one-model]]
- **Accesso:** API hosted in early access (waitlist), non weights
- **Costo:** $0.042 per 1M input token, output gratis ($42 per miliardo)
- **Latenza:** 70–500 ms, p50 misurato da terzi 236–276 ms
- **Nome:** William Stanley Jevons (efficienza → domanda di carbone → domanda di intelligenza)

## Come funziona
- Architettura nuova + parallel sampler
- Training: **RLCD** (Reinforcement Learning for Calibrated Decisions)
- Zero type error *per costruzione*
- Fino a **255 opzioni** per `choice` out-of-the-box

## Numeri pubblicati (Jev 1.13.0)
Da benchmark terzi (typed-decisions, 2000 decisioni):

| metrica | Jev | Laya (routed) |
|---|---|---|
| accuracy | 0.727 | 0.766 |
| soft accuracy | 0.580 | 0.471 |
| Brier | 0.148 | 0.062 |
| ECE | 0.144 | 0.213 |
| Banking77 (77 label) | **0.870** | 0.425 |

⚠️ I numeri di Jev sono **pubblicati da terzi, mai rilevati in-repo** da Convai (nessun accesso API). Trattarli come indicativi.

## Jev nella study
- Target ufficiale della proposta originale.
- **Problema:** non c'è API key garantita per tutti, e attaccare l'API hosted costa.
- **Alternative:**
  1. usare [[laya]] come target principale (stesso wire format, pesi scaricabili)
  2. attaccare Jev in black-box puro, dichiarando il costo
  3. attaccare Laya in gray-box e verificare il transfer su Jev

→ vedere [[black-box-vs-gray-box]] e .

## Note collegate
[[laya]] · [[system-one-model]] · [[wire-format-v1-systemone]] · [[link-utili]]
