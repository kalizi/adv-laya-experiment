---
tags: [esperimento, setup]
stato: fatto
data: 2026-09-23
eseguito: 2026-09-24
---

# E00 — Setup ambiente e smoke test locale

## Ipotesi
Nessuna ipotesi: è il gate. Senza E00 non si parte [[E01-baseline]].

## Setup
- checkpoint: `convaiinnovations/laya-typed-decisions` (+ `laya`, `laya-multilingual` se la VRAM lo permette)
- dispositivo: da decidere (Vedi [[configurazione-ambiente]])
- dataset: scaricare [[typed-decisions-dataset]] e congeliare l'hash

## Metodo

### 1. Ambiente
```bash
# usare un venv pulito; Python >= 3.10
python -m venv .venv && source .venv/bin/activate
pip install "laya"                      # e "laya[serve]" se serve l'HTTP server
export USE_TF=0                         # workaround deadlock transformers+TF
```

### 2. Smoke test
```python
import laya
agent = laya.load("convaiinnovations/laya-typed-decisions")
state = "Invoice #4471 from ACME Corp for $12,400 is 45 days overdue..."
questions = {
  "urgency": {"type": "score", "instructions": "How urgent is this?",
              "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]},
  "is_late": {"type": "noul", "instructions": "Is this invoice overdue?"},
}
r = agent.predict(state, questions)
print(r["answers"])
```
Atteso: risposte tipizzate, `usage.output_tokens == 0`, nessun errore di forma.

### 3. Verifiche tecniche (obbligatorie, sono "osservabilità")
- [ ] **Troncatura**: stato con marker unico a token 50 vs 400 vs 900 → il marker in coda compare nella tokenizzazione? Serve per validare [[truncation-stato]]
- [ ] **Router**: stesso stato in italiano / inglese / arabo traslitterato → quale checkpoint risponde? (loggare `r.get("model")`). In italiano ci aspettiamo `multilingual` → [[dataset-italiano]]
- [ ] **spaCy IT**: `python -m spacy download it_core_news_md`, poi su 20 stati italiani contare quante entità gold-critical (date, importi) il NER base protegge (aspettarsi poche: solo PER/LOC/ORG/MISC)
- [ ] **Budget**: lunghezza reale (question + criteria + options) con il tokenizer del checkpoint → confermare `head_max_len`
- [ ] **Latency**: p50 su 100 chiamate (per confronto con 33ms T4 / CPU reale)
- [ ] **Repro**: fissare `laya.__version__` e hash dei pesi (`safetensors` sha256)

### 4. (opzionale) server locale
```bash
LAYA_DEVICE=cpu LAYA_PRELOAD=1 laya-serve
curl -s localhost:8000/healthz
```
Utile per attacchi che passano dal wire format → [[wire-format-v1-systemone]].

## Risultati
Comando: `uv run adv e00` (`src/layaadv/experiments/e00_setup.py`) → `results/E00_setup.json` (gate_pass=**true**)

| verifica | esito |
|---|---|
| smoke test | **OK** — score/noul/choice tipizzati, `usage.output_tokens == 0` (160 input tokens) |
| marker coda visibile? | **OK, confermato troncamento**: visibile a 46/376/846 token di stato; **perso a 1130** (budget effettivo ≈ `max_len − head` ≈ **992** token, right-truncation `st[:room]` → coda tagliata) |
| router scelta per lingua | **OK** — IT con/senza accenti → `multilingual`; EN → `english`; **traslitterazione araba → `english`** (bypass confermato, `routing-script-bypass`); URL-heavy → `english` (default); misto IT/EN → `multilingual` |
| head_max_len effettivo | **OK** — cfg=256, head costruito=**31** token (3 option markers), `raw` question=42; ampio margine |
| p50 latency | **15.92 ms** (p95 16.92, mean 16.00, std 0.48, n=100) su **RTX 3080** — meglio del riferimento 33 ms T4. Il valore 17.46 ms delle versioni precedenti non è ricalcolabile da nessun archivio. |
| spaCy IT gold-critical | **0/7** coperti (time/date: solo `PER`×6 predetti) → confermato: NER base non protegge importi/date italiani |
| repro | laya **0.3.11**, torch 2.14.0+cu130, transformers 5.17.0, py 3.14.7, sha256 pesi in `results/E00_setup.json` |
| warning load | **temperature invalida** upstream: `choice:11+=0.1006` clamped a 0.5 → confidenza su quella entry **uncalibrated** (da dichiarare nel paper) |

## Verdetto
**Gate aperto → [[E01-baseline]].** Osservabilità a posto: troncatura, router, budget, latenza, hash misurati; warning temperatura upstream registrato.

## Note collegate
[[configurazione-ambiente]] · [[E01-baseline]] · [[esperimenti]]
