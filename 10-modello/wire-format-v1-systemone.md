---
tags: [modello, api, wire-format]
aliases: [/v1/systemone, wire format]
---

# Wire format `/v1/systemone`

Stesso protocollo per Jev e Laya. Cambiare `baseUrl` basta.

## Request
`POST /v1/systemone`

```json
{
  "model": "jev-latest",
  "state": "Il testo dell'email/ticket/documento...",
  "questions": {
    "is_phishing": {
      "type": "noul",
      "instructions": "Is this email a phishing attempt?",
      "criteria": {"true": "...", "false": "..."}
    },
    "department": {
      "type": "choice",
      "instructions": "Which department handles this?",
      "criteria": {"billing": "...", "technical": "..."}
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this?",
      "criteria": ["not urgent", "soon", "critical"]
    }
  }
}
```

## Response
```json
{
  "answers": {
    "is_phishing":  {"noul": 0.91, "confidence": 0.84},
    "department":   {"choice": "billing", "probabilities": {...}, "confidence": 0.94},
    "urgency":      {"score": 1.84, "probabilities": {"0": ..., "1": ..., "2": ...}, "confidence": 0.77}
  },
  "model": "laya-english",
  "usage": {"input_tokens": 312, "output_tokens": 0}
}
```

Note:
- Le risposte sono **keyed dai tuoi question ids**
- `usage.output_tokens` è sempre 0 su Laya (nessuna generazione)
- Le chiavi dei `probabilities` di `score` sono **stringhe** sul wire (`{"0": ...}`), come Jev HTTP

## Validazione (fail-fast, 422)
- `model` obbligatorio
- `questions` non vuoto
- `choice` ≤ 255 opzioni; `score` 2–10 livelli
- `choice.criteria` come lista: Jev rifiuta, Laya accetta lenientamente (→ `{label: None}`)
- `noul` è **`noul`**, non `null`

## Model names accettati da `laya-serve`
`jev-latest`, `jev-preview`, `jev-1.13.0`, `jev-1.13`, `laya`, `laya-latest`, + l'id del serving model. Altro → 422.

## Divergenze deliberate di `laya-serve`
1. Campi Laya-only rimossi: `action`, `confidence` su `noul`
2. `model` echo del serving checkpoint
3. `usage.output_tokens` = 0
4. Formula confidence diversa (vedi [[rlcd-calibrazione]])
5. Contesti: i limiti sono quelli del checkpoint (`max_len`/`head_max_len`), non i 64k/32k di Jev

## Avvio server locale
```bash
pip install "laya[serve]"
LAYA_DEVICE=cuda LAYA_PRELOAD=1 laya-serve   # bind 0.0.0.0:8000
```
Variabili: `LAYA_HOST`, `LAYA_PORT`, `LAYA_DEVICE`, `LAYA_PRELOAD`, `LAYA_MODELS`, `LAYA_THREADS`, `LAYA_AUTO_TASK`, `LAYA_API_KEY`.

> Nota per la study: `laya-serve` ha anche backend `fake` (`LAYA_SERVE_BACKEND=fake`) per testare il wiring senza pesi.

## Note collegate
[[jev]] · [[laya]] · [[primitive-typed]] · [[link-utili]]
