---
tags: [risorse, link]
---

# Link utili

## Laya
- Hub (famiglia, 3 checkpoint): https://huggingface.co/convaiinnovations/laya
- Checkpoint specialist: https://huggingface.co/convaiinnovations/laya-typed-decisions
- Multilingue: https://huggingface.co/convaiinnovations/laya-multilingual
- GitHub: https://github.com/NandhaKishorM/laya (branch `research` = dati completi del benchmark)
- PyPI: https://pypi.org/project/laya/ (`pip install laya`, serve `laya[serve]` per l'HTTP server)
- Demo live (ZeroGPU): https://huggingface.co/spaces/convaiinnovations/laya-demo
- Sito: https://laya.convaiinnovations.com/
- Fine-tuning notebook Kaggle 2×T4 (~4–5 h): `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`
- Issue citate: **#156** (`noul` label bias), **#131** (`score` position bias)

## Jev / TypeSafe
- Blog di annuncio: https://typesafe.ai/blog/introducing-system-one-models-and-jev
- Docs wire format: `docs.typesafe.ai`
- Prezzo: $0.042 / 1M input token

## Dataset
- **typed-decisions (ufficiale):** https://huggingface.co/datasets/LocalLLaMA/typed-decisions
- Corpus companion v1/v2 (label corrette): https://huggingface.co/datasets/pngwn/typed-decisions-v2
- Italiani (vedi [[dataset-italiano]]): `Federico82/prompt_injections`, `felfri/ALERT_it`, corpus *Compromesso!*, `Cogensec/jailbreak-corpus`

## spaCy
- Modelli italiani: https://spacy.io/models/it (`it_core_news_sm/md/lg`)
- Gap NER IT (solo PER/LOC/ORG/MISC): explosion/spacy discussion #6489
- Install: `uv run python -m spacy download it_core_news_md`

## Server / interop
- `pip install "laya[serve]"` → `laya-serve` → `POST /v1/systemone` (Jev-compatible)
- Hosted gratis wire-compatible (se mai servisse): impossibl (endpoint Laya compatibile Jev)
- Client alternativi per lo stesso wire: `receptron` (Node), `hs-jev` (Haskell)

## Benchmark / gare correlate
- JevBench (Florian S): Jev 75.4 / SemIf 74.7 / djev 74.3 (v1.2.8)
- Reddit head-to-head jev vs laya, diversi video YouTube, benchmark HF (il "benchmark fight" del settore)
- madewithlaya.com: catalogo di build sulla community (sidecar token-budget preflight, fix noul, cascade Laya→Jev)

## Attacchi / sicurezza (già pubblicati)
- **tanh.xyz prompt-injection test** (22 set 2026) → [[attacchi-pubblicati]]
- Decoding Jev (option-list sensitivity di Hume)
- djev / DiffusionGemma-Jev: hosted https://api.djev.dev (invite), open stack `Davipar/djev-dev`, `razorback16/openjev`, vLLM PR 57250

## Per gli attacchi (letteratura e toolkit)
- Paper encoder vs decoder judge: https://arxiv.org/html/2606.25782
- PangolinGuard (ModernBERT prompt-guard, baseline utile): https://huggingface.co/blog/dcarpintero/pangolin-fine-tuning-modern-bert
- Datasets injection: PINT, InjectGuard, HackPrompt, NotInject, WildGuard, BIPIA
- Attack classics: AdvBench (Zou et al. 2023), STRONGReject (Souly et al. 2024), SorryBench

## Da tenere d'occhio
- Nuovi "System One" in circolazione (terzo entrante segnalato su r/LocalLLaMA)
- Releases di Laya: i bug #156/#131 potrebbero essere chiusi con nuovi checkpoint → rileggere i release note prima di pubblicare

→ anche [[paper-e-riferimenti]]
