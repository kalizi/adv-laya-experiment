---
tags: [modello, system-one, jep, kahneman]
aliases: [System One, System 1]
---

# System One model

Categoria di modello annunciata da TypeSafe AI il 15 settembre 2026 con il lancio di [[jev]]. Il nome viene da *Thinking, Fast and Slow* di Kahneman: System 1 è il giudizio veloce ed intuitivo, System 2 è il ragionamento lento.

## Definizione operativa
Un modello System One:
- riceve uno **stato** (testo, email, ticket, documento JSON) + un set di **domande tipizzate**
- restituisce **risposte tipizzate** con **probabilità calibrate** e confidence
- fa tutto in **una sola forward pass** (nessuna generazione token per token)
- non ha prompt, system instruction, temperature, tool calling

## Le tre primitive
Vedi [[primitive-typed]]:
- `choice` → etichetta + distribuzione
- `score` → livello ordinale atteso + distribuzione
- `noul` → P(true) ∈ [0,1]  *(non `null`! è `noul`, errore comune nei tutorial)*

## Perché conta per la study
Il modello **non genera testo**, quindi:
- ✅ niente hallucination sul formato, niente da parsare
- ✅ nessun "prompt injection" nel senso classico (non c'è un LLM da jailbreakare)
- ⚠️ ma c'è comunque un classificatore: la superficie d'attacco è quella di un **encoder + decision head**, vedi [[superficie-attacco]]
- ⚠️ la "type safety" garantisce la **forma** della risposta, non la **verità** (contratto, non accuracy)

## Differenze da un LLM
| | LLM | System One |
|---|---|---|
| Output | stringa generata | valore tipizzato + probabilità |
| Velocità | secondi | ~33–400 ms |
| Type errors | possibili (struttura forzata a valle) | zero per costruzione |
| Verità | non garantita | non garantita (idem) |
| Alucinazione di testo | sì | impossibile |

## Alternative esistenti
- [[jev]] — TypeSafe, hosted, closed
- [[laya]] — Convai Innovations, Apache 2.0, self-hosted
- Ricezione wire compatibile: stesso schema → scelta = deployment, non rewrite (vedi [[wire-format-v1-systemone]])

## Note collegate
[[laya]] · [[jev]] · [[primitive-typed]] · [[rlcd-calibrazione]] · [[superficie-attacco]]
