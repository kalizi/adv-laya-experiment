---
tags: [dataset, secondari]
---

# Dataset secondari

Serve per generalizzare oltre typed-decisions (criterio 🥈/🥇 in ). Tutti scaricabili e pubblicabili (verificare licenza per dataset).

## Classici già usati nei benchmark Laya vs Jev
| dataset | task | primitive suggerite | perché |
|---|---|---|---|
| **AG News** | 4 classi news | `choice` | baseline Jev 0.910 / Laya 0.950 → alta, c'è margine di drop |
| **DAIR Emotion** | 6 emozioni | `choice` | Jev 0.480 / Laya 0.595 → debole, buon bersaglio |
| **Banking77** | 77 intent banking | `choice` | il caso estremo del [[token-budget-head]] |
| **SST-5** | sentiment 5 livelli | `score` | primitive più debole: Laya 0.372, multilingual 0.282 |
| **XNLI** (15 lingue) | NLI | `choice` | cross-lingual → [[routing-script-bypass]] |

## Guardrail / spam / phishing (domini nativi di Laya)
Laya è stato addestrato con dati di email triage e guardrails:
- **Enron spam**, phishing emails, phishing URLs, support tickets, Bitext intents (menzionati nel model card multilingual)
- **PINT / InjectGuard / HackPrompt / NotInject / WildGuard-Benign / BIPIA** → prompt injection su encoder (baseline note: PangolinGuard ModernBERT 84.72% su mist benchmark)
- **AdvBench / STRONGReject / SorryBench** → tecniche di attacco catalogate, adattabili come in [[prompt-injection-encoder]]

⚠️ Attenzione al **contamination**: se il dominio è in-distribution del training, ASR alto è più significativo; se out-of-distribution, controllare che la baseline non sia già a terra (regola del drop relativo in ).

## Corpus già "corretti" per baselines
- `pngwn/typed-decisions` e `pngwn/typed-decisions-v2`: 37.852 righe, split train/cal/test per md5(uid). v2 corregge l'inversione delle label review/escalate della v1 (6.850 righe `review`/`escalate`, 6.127 righe `workflow4`). Cal split riservato al temperature fitting.
  - I checkpoint addestrati su v1 hanno imparato il target invertito: Bayes-optimal apparente 0.082 su escalate = 1 − 0.918.
  - **Per noi:** usare v2 se vogliamo label sane; i checkpoint esistenti restano validi per severity/team/noul/choice.

## Scelta consigliata per la study
1. **Primary:** [[typed-decisions-dataset]] (test 400)
2. **Secondary:** AG News o DAIR Emotion (choice, rapido da girare)
3. **Stress:** Banking77 per il token budget, SST-5 per `score`
4. **Cross-lingual:** XNLI per il router

→ Compilare in [[protocollo-dataset-valido]] con licenza e checksum.

## Note collegate
[[typed-decisions-dataset]] · [[protocollo-dataset-valido]] · [[token-budget-head]]
