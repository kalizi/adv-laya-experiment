---
tags: [moc, laya, study]
---

# Adversarial Laya Study

## Candidato di release

I risultati verificabili, i record compressi con hash, le licenze dei dataset core e i
comandi esatti di ripristino e controllo sono in [results/release/README.md](results/release/README.md).
Il candidato non ha ancora un commit Git; `jailbreak-corpus` non è incluso.

Study aperta: **14 giorni** per produrre input avversari che fanno "dire fischi per fiaschi" a Laya (clone open-source di Jev, il System One model di TypeSafe AI).

Tutto è pubblico per regola: testi, dataset, codice, risultati. Serve per un articolo, un post o un portfolio GitHub.

**Scadenza:** 12 ottobre 2026 (14 giorni da 28 settembre 2026).

## Mappa del vault

### 🧠 Modello
- [[system-one-model]] — che cos'è un modello System One
- [[jev]] — il modello chiuso di TypeSafe AI
- [[laya]] — il clone Apache-2.0
- [[architettura-laya]] — encoder + decision head, senza generazione di testo
- [[primitive-typed]] — `choice`, `score`, `noul`
- [[rlcd-calibrazione]] — come (e se) le probabilità sono calibrate
- [[wire-format-v1-systemone]] — il protocollo HTTP Jev-compatibile
- [[checkpoint-laya]] — i tre checkpoint e il Router

### ⚔️ Superficie d'attacco
- [[superficie-attacco]] — MOC di tutte le leve
- [[attacchi-pubblicati]] — stato dell'arte: cosa è già stato fatto (tanh, Hume, #156/#131/#185)
- [[token-budget-head]] — e options condividono un budget fisso: le options possono essere affamate
- [[noul-label-bias]] — il bug #156: `noul` segue le label, non lo stato
- [[score-position-bias]] — il bug #131: bias di posizione sul primo livello
- [[routing-script-bypass]] — forzare il checkpoint sbagliato
- [[option-marker-injection]] — attaccare lo spazio delle opzioni
- [[truncation-stato]] — far finire la parte critica fuori dal budget
- [[calibrazione-fratturata]] — over-confidence come attacco, non solo come difetto
- [[prompt-injection-encoder]] — cosa sappiamo dagli encoder classifier

### 📊 Dataset
- [[typed-decisions-dataset]] — il benchmark ufficiale (gold standard)
- [[dataset-secondari]] — spam, phishing, intent, guardrails
- [[dataset-italiano]] — supporto IT di Laya, dataset avversari italiani, spaCy it_core_news
- [[protocollo-dataset-valido]] — come costruire un dataset valido per la study

### 🔬 Esperimenti
- [[esperimenti]] — indice
- [[E00-setup]] — setup ambiente, primo giro
- [[E01-baseline]] — replicare la baseline 0.766 (acc 0.767, congelata)
- [[E02-option-count]] — collasso dell'accuracy con K opzioni (diagnostico)
- [[E03-option-injection]] — **nominare un'opzione distratore nello stato** (ASR 0.83-0.89)
- [[E04-truncation]] — spingere il segnale fuori dal budget dello stato
- [[E13-label-semantics]] — le opzioni sono un canale di informazione? (diagnostico)
- [[E03xE04-composizione]] — injection + troncatura: compongono o si annullano?
- [[template-esperimento]] — template per i nuovi esperimenti

### 🛠️ Metodologia
- [[black-box-vs-gray-box]] — livelli di accesso ammessi
- [[metriche-attacco]] — ASR, OER, perturbazione, transfer
- [[configurazione-ambiente]] — uv, hardware, riproducibilità
-  — obbligo di pubblicità (a fine study)

### 🧩 Interpretabilità
- [[MOC-interpretabilita]] — mappa del materiale e dei concetti
- [[attenzione-come-spiegazione]] — perché le mappe di attenzione non spiegano (e cosa misurava E22)
- [[bertology-encoder-mascherati]] — cosa si sa degli encoder mascherati
- [[attribuzioni-e-sanity-check]] — gradienti, LRP, e i test da fare prima di crederci
- [[probe-e-controfattuali]] — probe con control task e mediazione causale
- [[unita-di-analisi]] — neurone, testa, direzione, feature
- [[circuiti-e-interventi]] — patching, knockout, circuiti: da correlazione a causa
- [[perche-gli-attacchi-funzionano]] — trigger universali, feature non-robuste, privilegi
- [[letteratura-collaterale]] — la letteratura a un hop (SAE, induction head, causal scrubbing)
- 23 paper in `60-interpretabilita/paper/` (PDF + manifest sha256, `tools/fetch_papers.py`)

### 📚 Risorse
- [[link-utili]] — repo, HF, PyPI, API
- [[paper-e-riferimenti]] — letteratura rilevante
- [[glossario]] — termini


### Note collegate
[[superficie-attacco]] → [[esperimenti]] → [[metriche-attacco]] → 
