# Candidato di release, senza commit

Questa directory rende verificabili i risultati dal repository senza distribuire i dataset grezzi o i pesi dei modelli. Non esiste ancora un commit Git: `snapshot.json` identifica i file del candidato con SHA-256, ma non sostituisce un commit fissato e pubblicato.

## Record e ripristino

`records.manifest.json` elenca 376 `preds.jsonl` citati dai riepiloghi, inclusi i bracci E01 ed E16, con dimensione e SHA-256 per file. I sette `records-*.tar.zst` contengono soltanto quei record (ID, tipi, label, probabilità, gold e uso), in percorsi `results/<esperimento>/runs/<run_id>/preds.jsonl`. Gli archivi complessivi occupano circa 79 MB. Le 41 sonde storiche E12d, E12f ed E12i non hanno record grezzi e sono dichiarate esplorative.

Da una copia del progetto, nella radice:

Serve `zstd` installato nel sistema oltre alle dipendenze del lockfile.

```sh
uv sync --locked
uv run python tools/release_records.py verify
for archive in results/release/records-*.tar.zst; do tar --zstd -xf "$archive"; done
uv run python tools/verify_results.py
uv run python tools/verify_e16_append.py
```

La verifica degli archivi funziona anche prima dell'estrazione. Con i record estratti si controllano risultati e inventario. `results/release/e03_note_exceptions.json` classifica i 32 valori della nota E03 che non sono campi diretti del suo scope; `uv run python tools/release_note_audit.py` fallisce se compaiono eccezioni nuove o cambiano le occorrenze.

## Dataset e licenze

`data/raw/` non è distribuita in questo candidato. Per rifare i forward, `uv run python tools/release_fetch_core.py` scarica solo i tre dataset core e confronta i sette file con gli hash di `data/manifest.json`. Su una macchina che ha già i file, `uv run python tools/release_data_check.py` esegue soltanto il controllo locale. Se una fonte cambia byte, il controllo fallisce e non si deve sostituire il manifest storico con il nuovo download. I tre dataset dei claim principali sono:

| Dataset usato | Origine | Licenza dichiarata | Provenienza locale |
|---|---|---|---|
| typed-decisions | [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | Apache-2.0 | `data/manifest.json`, split train/test |
| banking77 | [mteb/banking77](https://huggingface.co/datasets/mteb/banking77) | MIT nella card corrente; il manifest iniziale riportava erroneamente CC BY 4.0 | `data/manifest.json`, split train/test |
| MASSIVE 1.0 | [Amazon MASSIVE](https://github.com/alexa/massive) | CC BY 4.0 | `data/manifest.json`, `data/raw/massive-1.0/LICENSE` locale |

Le card di [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/tree/main) e [banking77](https://huggingface.co/datasets/mteb/banking77/tree/main), oltre al [NOTICE di MASSIVE](https://github.com/alexa/massive/blob/main/NOTICE.md), sono state consultate il 2026-10-07. Le licenze degli altri dataset del manifest non sono usate per autorizzare una loro redistribuzione: alcuni sono gated, non commerciali o segnati `unknown`/`see-card`. `jailbreak-corpus` non è stato scaricato né usato.

## Esecuzione degli esperimenti principali

I comandi seguenti usano gli stessi parametri dichiarati nei file di scope; per riprodurre la stessa identità occorrono anche i pesi identificati dagli SHA-256 nei `config.json` e nel manifest di provenienza E03. Batch, device e runtime possono influire sui bit delle probabilità.

```sh
uv run adv e03 --checkpoint convaiinnovations/laya-typed-decisions --dataset massive-it --device cuda --batch-size 16 --seed 0 --pad 0 --template prepend --template append
uv run adv e03 --checkpoint convaiinnovations/laya-typed-decisions --dataset massive-en --device cuda --batch-size 16 --seed 0 --pad 0 --template prepend --template append
uv run adv e16 --checkpoint convaiinnovations/laya-typed-decisions --dataset massive-it --device cuda --batch-size 16 --replicates 5 --template append --position tail
uv run python tools/e21_clm_distribution.py > results/provenance/e21_clm_distribution.json
```

Il backend CLM richiede `llama-server`, `clm-serve`, GGUF e teste separati: setup e limiti sono in `40-esperimenti/E21-clm-hard-negatives.md` e `10-modello/clm.md`. I forward completi sono costosi; gli archivi permettono di verificare i numeri senza GPU.

## Gate del candidato

```sh
uv run python -m pytest -q
uv run ruff check .
uv run python tools/verify_results.py
uv run python tools/verify_e16_append.py
uv run python tools/run_identity_audit.py
uv run python tools/release_note_audit.py
uv run python tools/release_records.py verify
uv run python tools/release_data_check.py
uv run python tools/release_snapshot.py verify
```

`run_identity_audit.py` segnala anche gli store storici con identità incompleta; il controllo fallisce solo su incoerenze. Il gate è riferito ai byte elencati in `snapshot.json`. Senza commit e senza un checkout pulito non è ancora una release pubblicabile.
