---
tags: [risorse, glossario]
---

# Glossario

| termine | definizione |
|---|---|
| **System One model** | Modello che risponde a domande tipizzate su uno stato con probabilità calibrate, in una forward pass, senza generare testo → [[system-one-model]] |
| **Jev** | System One model chiuso di TypeSafe AI → [[jev]] |
| **Laya** | Clone Apache-2.0 di Jev, pesi pubblici → [[laya]] |
| **State** | Il testo/documento in input (email, ticket, JSON) |
| **Question** | Una domanda tipizzata: `choice`, `score` o `noul` → [[primitive-typed]] |
| **`noul`** | Primitive booleana: P(true) ∈ [0,1]. **Non** è `null` |
| **`criteria`** | Mappa (choice) o lista (score) che definisce lo spazio delle risposte |
| **RLCD** | Training: reward = strictly proper scoring rule, REINFORCE + baseline di gruppo → [[rlcd-calibrazione]] |
| **ECE** | Expected Calibration Error: scostamento tra confidence dichiarata e accuracy reale |
| **Brier** | MSE tra probabilità predette ed evento binario |
| **OER** | Overconfident Error Rate: torto E confidente (≥ τ) → [[metriche-attacco]] |
| **ASR** | Attack Success Rate: % di esempi corretti al pulito che l'attacco rovina → [[metriche-attacco]] |
| **Soft accuracy** | Accuratezza della distribuzione intera vs teacher (non solo argmax) |
| **Option marker** | `[MASK]` per opzione, scored dal decision head → [[architettura-laya]] |
| **`head_max_len`** | Budget di token condiviso da question + criteria + opzioni → [[token-budget-head]] |
| **Router** | Selettore di checkpoint per script/lingua, prima della forward pass → [[checkpoint-laya]] |
| **Gray-box** | Attacco con accesso ai pesi (gradient) → [[black-box-vs-gray-box]] |
| **Goal shift** | L'attacco cambia il gold: escluso da ASR, da dichiarare |
| **Teacher ceiling** | Auto-accordo del teacher che ha etichettato il dataset (0.735 su typed-decisions) |
| **Wire format** | Schema HTTP `/v1/systemone`, identico per Jev e Laya → [[wire-format-v1-systemone]] |
| **ModernBERT** | Encoder bidirezionale backbone di Laya (421M large) |
