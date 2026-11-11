---
tags: [modello, api, primitive]
aliases: [choice, score, noul]
---

# Primitive tipizzate

Tre forme di domanda. Nessun'altra. (`noul` non è `null`.)

## `choice`
```python
{
  "type": "choice",
  "instructions": "Which department should handle this email?",
  "criteria": {
    "billing": "invoices, payments, refunds",
    "technical": "bugs, outages, integrations",
    "sales": "pricing, contracts, demos",
    "other": "everything else"
  }
}
```
**Risposta:** etichetta selezionata, probabilità per opzione, confidence.
- Max 255 opzioni (Jev); Laya: meglio **< 20 opzioni** → [[token-budget-head]]
- `criteria` = mappa opzione → descrizione (Jev richiede mappa, Laya accetta anche lista in modo leniente)

## `score`
```python
{
  "type": "score",
  "instructions": "How urgent is this request?",
  "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]
}
```
**Risposta:** livello atteso (0, 1, 2...), distribuzione, confidence.
- 2–10 livelli
- Primitive **più debole** (SST-5: 0.372 per `laya`; 0.282 multilingual)
- [[score-position-bias]] sul multilingual

## `noul`
```python
{
  "type": "noul",
  "instructions": "Is this email a phishing or scam attempt?",
  "criteria": {  # opzionale
    "true": "phishing, scam, credential theft",
    "false": "legitimate business email"
  }
}
```
**Risposta:** P(true) ∈ [0,1], P(false) = 1 − P(true).
- Rendering interno delle options: `["false: ...", "true: ..."]` → l'ordine è **fisso false, true**
- Ed è proprio lì il problema → [[noul-label-bias]]
- Workaround documentato: riscrivere come `choice` a 2 opzioni con chiavi neutre:
```python
{"type": "choice",
 "instructions": "Is this review positive?",
 "criteria": {"A": "yes, the review is positive",
              "B": "no, the review is negative"}}
```

## Forma della richiesta completa
`POST /v1/systemone`, vedi [[wire-format-v1-systemone]].
Le risposte tornano **keyed dai tuoi question ids**.

## Note collegate
[[architettura-laya]] · [[noul-label-bias]] · [[score-position-bias]] · [[token-budget-head]] · [[wire-format-v1-systemone]]
