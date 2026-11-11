---
tags: [dataset, italiano, lingua, spacy]
aliases: [italiano, it, dataset IT]
---

# Laya e italiano

## Supporto di Laya all'italiano

| checkpoint | italiano? | note |
|---|---|---|
| `laya` (English) | no | Latin script lo riceve comunque di default in versioni vecchie → degradazione (MASSIVE non-inglese: 0.306 vs 0.451 del multilingue) |
| `laya-multilingual` | **sì** | `it` nella lista 100+ lingue mmBERT; da v0.3.x anche l'italiano Latin senza accenti (email/ticket) viene routato qui |
| `laya-typed-decisions` | **no, English only** | il checkpoint con baseline 0.766: per usarlo in italiano va tradotto il dataset |

- La cifra **specifica per `it`** (MASSIVE intent, 20 opzioni) non è riportata qui: è in `BENCHMARKS.md` upstream per tutte e 51 le lingue → **misurare in [[E01-baseline]]**, non assumere.
- `laya-multilingual` **shippa senza temperature fitted** → ECE raw peggiore; rifittare prima di fidarsi della confidence in italiano → [[calibrazione-fratturata]].
- Le domande/criteria in italiano vs inglese: da testare (sul multilingue si può; il consiglio upstream è comunque "translate state and policies into the language the model was trained on" per il checkpoint giusto).

**Implicazione per la study:** attaccare l'italiano su `laya-multilingual` è lecito e non coperto da tanh (che usava 3 lingue su injection, non un GA in italiano). Usare `laya-typed-decisions` in italiano richiede prima un dataset tradotto dichiarato → [[protocollo-dataset-valido]].

## Dataset avversari in italiano (stato)

### Orientati LLM (esistono)
| dataset | contenuto | licenza |
|---|---|---|
| `Federico82/prompt_injections` | 1000+ iniezioni in 7 lingue **incluse it** (prompt leaking, jailbreak, mode switching) | da verificare |
| `felfri/ALERT_it` | ~15K red-teaming prompts tradotti (MT-OPUS-en-it) + sottoinsieme adversarial **ALERTAdv** con campo `attack_type` | CC BY-NC-SA 4.0 |
| Dataset *Compromesso!* (ACL 2024 SRW) | 418 coppie unsafe QA italiane per many-shot jailbreak; codice+dati su GitHub | da verificare |
| `Cogensec/jailbreak-corpus` | 59.508 attacchi reali+synthetic, 15 lingue **incluse it**, gated | ODC-BY / MIT / Apache per sorgenti |
| `Necent/llm-jailbreak-prompt-injection-dataset` | aggregato 30+ dataset, campo `language` (contiene MultiJail 10 lingue) | da verificare |

⚠️ Tutti pensati per **guardrail/generator**, non per misurare la robustezza di un classifier tipizzato su dati puliti.

### Mancano (lo spazio nostro)
- Adversarial example TextFooler/BERT-Attack-style **in italiano** con gold pulito
- Qualsiasi dataset italiano strutturato per `choice`/`score`/`noul`
- Baseline classification pulita in italiano (opszioni: tradurre uno split di typed-decisions, o PAULA per news classification, o MASSIVE `it` come intent)

→ Bozza percorso: **MASSIVE `it`** (20 intent, già usato nei benchmark Laya) come baseline pulita italiana + traduzione di 100-200 casi typed-decisions come workload decisionale. Entrambi da dichiarare in [[protocollo-dataset-valido]].

## spaCy per italiano (vincolo della ricerca lessicale)

Modelli ufficiali Explosion:

| modello | size | pipeline | vetti |
|---|---|---|---|
| `it_core_news_sm` | 10 MB | tok2vec, morphologizer, tagger, parser, lemmatizer, senter, ner | 0 |
| `it_core_news_md` | 40 MB | idem | 20k |
| `it_core_news_lg` | 541 MB | idem | 500k |

```bash
uv run python -m spacy download it_core_news_md   # compatibile spacy 3.8.x in lock
```
Licenza modelli: **CC BY-NC-SA 3.0** (attribution + noncommercial + sharealike: ok per la study, dichiarare).

### Il vincolo del NER sulla ricerca lessicale
Il NER italiano è addestrato su **WikiNER** e copre solo:
```
PER, LOC, ORG, MISC
```
**Niente `DATE`, `MONEY`, `CARDINAL`** out of the box (discussion ufficiale explosion/spacy#6489 conferma: lo scheme italiano ≠ OntoNotes inglese).

Il NER protegge le entità dalle sostituzioni lessicali ([[E12b-sinonimi-rari]]). Su italiano:
- importi, date, numeri di fattura **non sono protetti** di default
- per typed-decisions (invoice, security) sono esattamente le entità che determinano il gold
- **Soluzione:** EntityRuler o regex custom in coda alla pipeline:
```python
from spacy.pipeline import EntityRuler
ruler = nlp.add_pipe("entity_ruler", before="ner")
ruler.add_patterns([
    {"label": "MONEY", "pattern": [{"TEXT": {"REGEX": r"\d{1,3}(\.\d{3})*(,\d{2})?\s*€?"}}]},
    {"label": "DATE",  "pattern": [{"TEXT": {"REGEX": r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"}}]},
])
```
Più le eccezioni italiane: "1 miliardo di dollari", "scorso dicembre", IBAN, P.IVA, codici fattura (`[A-Z]{2}\d+`).

→ Da validare in E00: su 20 stati italiani, quante entità gold-critical il NER base protegge? (aspettarsi: poche)

## Note collegate
[[E12b-sinonimi-rari]] · [[E00-setup]] · [[E01-baseline]] · [[protocollo-dataset-valido]] · [[dataset-secondari]] · [[routing-script-bypass]]
