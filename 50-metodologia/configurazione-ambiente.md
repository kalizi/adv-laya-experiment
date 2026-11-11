---
tags: [metodologia, ambiente, locale]
---

# Configurazione ambiente

**Solo locale.** Nessun endpoint esterno obbligatorio nella pipeline.

## Layout del repo
```
adversarial-laya/
├── README.md
├── 00-study/ 10-modello/ 20-attack-surface/
├── 30-dataset/ 40-esperimenti/ 50-metodologia/ 90-risorse/
├── src/layaadv/           # pacchetto: model/engine, metrics, evaluate, experiments (E##_*)
├── tests/                 # pytest: metriche, resume, loader
├── data/
│   ├── raw/               # dataset scaricati, immutable + sha256
│   └── adv/               # stati avversari generati (jsonl)
├── results/               # metriche, predizioni, log, run checkpointed
└── .venv/
```

Regole: `layaadv.model.engine` è l'unico modulo che importa `laya`/torch; le
metriche sono funzioni pure; ogni esperimento passa da `layaadv.evaluate` +
`layaadv.io.RunStore` (checkpoint → Ctrl-C e riprendi).

## Ambiente Python
Gestione con **uv**: `pyproject.toml` + `uv.lock` vanno committati; `.venv/` no.
```bash
uv sync                          # replica esattamente da uv.lock
uv run adv e00                   # CLI unica: e00 e01 e02 e03 e04 e12 e13 datasets
uv run pytest                    # test (nessuna GPU richiesta)
uv add <pkg>                     # aggiorna pyproject + lock
```
Dipendenze attuali: `laya`, `datasets`, `huggingface-hub`, `spacy`,
`it-core-news-md` (pin diretto, non su PyPI) (+ `ipython`, `pytest`, `ruff` in `dev`).
`USE_TF=0` (deadlock transformers/TF) è impostato da `layaadv` all'import:
esportarlo a mano non serve più.
Registrare: `laya.__version__`, sha256 dei safetensors, `torch.__version__`, dtype, device
(lo fa `io.env_fingerprint()` in ogni result).

## Hardware di riferimento (ipotesi, da misurare in [[E00-setup]])
| device | attesa | note |
|---|---|---|
| T4 (colab/kaggle free) | ~33 ms/q, 103–332 q/s batched | il riferimento dei modelli |
| CPU moderna | orders of magnitude più lento | batch piccoli; L0-L1 basta così |
| GPU locale (se c'è) | da misurare | |

Se non c'è GPU: i checkpoint 421M girano in fp32 su CPU per un test set di 400; l'intero sweep di attacco va dimensionato di conseguenza (N=100–200, non 400×varianti).

## GPU: quale checkpoint per volta
- `Router(preload=True)` tiene 2 checkpoint → VRAM doppia
- per sweep lunghi: caricare **un checkpoint alla volta** (`laya.load`, fine del processo), loggare il massimo di VRAM

## Offline / locale puro
- Una volta scaricati pesi (`~/.cache/huggingface`) e dataset (`data/raw/`), la valutazione è interamente offline
- Snapshot HF: `pip install huggingface_hub` → `hf_hub_download` + sha256 fissati nei log
- Nessuna chiave API richiesta per Laya
- (opzionale) Jev API: solo se qualcuno vuole il confronto L0 su Jev, a suo costo → [[link-utili]]

## Riproducibilità checklist
- [x] `uv.lock` + `pyproject.toml` nel repo (già impostati)
- [x] sha256 di ogni file in `data/raw/` (in `data/manifest.json`)
- [x] hash pesi checkpoint (mai committare `.safetensors`): in ogni result file
- [x] seed per qualsiasi generazione (filler, campionamenti GA)
- [ ] **versione del codice: nessun commit.** Il repo è `git init` senza commit: 59M di
      risultati in `results/` non sono versionati. Fino al primo commit, "quale codice ha
      prodotto questo numero" si risponde solo con i `config.json` dei run (che hanno
      dataset hash, subset hash, seed, chunk, batch_size, hash pesi). Il codice è
      l'unica cosa che manca.

## Note collegate
[[E00-setup]] · [[E01-baseline]] · [[configurazione-ambiente]] · [[esperimenti]]
