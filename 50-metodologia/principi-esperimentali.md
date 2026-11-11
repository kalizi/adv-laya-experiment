---
tags: [metodologia, principi]
data: 2026-09-27
---

# Principi sperimentali

Norme trasversali per tutti gli esperimenti (E00-E12+). Ogni esperimento deve rispettare questi principi per essere considerato valido.

## 1. Riproducibilita'

- Ogni risultato include: checkpoint + sha256 pesi, dataset + split + hash, versioni Python/laya/torch, device, git commit.
- Comando esatto nel payload (`uv run adv e## --limit N`).
- RunStore con `run_id = sha256(config)`: stessa config = stesso run, resume automatico.
- Numeri misurati, mai citati dal model card.

## 2. Isolazione

- `layaadv/model/engine.py` è l'unico modulo che importa laya/torch.
- `layaadv/metrics/core.py` è puro: zero dipendenze modello/IO.
- Nessun esperimento fa il suo loop: solo `layaadv/evaluate.py` (chunk + batch + checkpoint).
- Path solo in `layaadv/paths.py`, hash solo in `layaadv/hashing.py`.

## 3. Misura prima di attaccare

- Baseline pulita sullo stesso sottoinsieme prima di qualsiasi attacco.
- Filtri dichiarati (`min_agreement`, `limit`) e registrati nel payload.
- ASR calcolato solo su decisioni corrette al pulito (altrimenti un attacco "vince" su errori preesistenti).

## 4. Batching corretto

- Tutto passa da `engine.predict_batch`, che raggruppa per firma delle questions.
- Default `--batch-size 16`: misurato su RTX 3080 10GB, 32 fa OOM.
- Laya a OOM sposta il modello su CPU in silenzio: engine avvisa con RuntimeWarning.

## 5. Checkpointing

- `io.RunStore` per computazioni lunghe: append per chunk, resume di default.
- Un config diverso non si mischia mai (run separati, `ConfigMismatch`).
- `--no-resume` e `--force-new` disponibili ma non usati senza motivo.

## 6. Validazione empirica

- Ogni attacco verifica che il meccanismo atteso sia attivo (es. marker probe per troncatura).
- Risultati includono la prova del meccanismo, non solo il drop.
- Se il meccanismo non è attivo, il risultato è invalido anche se il drop c'è.

## 7. Statistica

- N dichiarato, con `n_decisions` e `n_cases` separati.
- Per primitive breakdown (`accuracy_by_primitive`, `brier_by_primitive`).
- Confidence intervals o error bars quando N >= 30.
- Significanza statistica (McNemar o bootstrap) per confronti tra varianti.

## 8. KISS/DRY

- Se due esperimenti condividono logica, va in libreria, non copiata.
- Minimum code that solves the problem: niente feature speculative.
- Commenti in inglese per docstring, italiano per note che rimandano al vault.

## 9. Output

- `results/E##_*.json` con payload strutturato.
- `results/E##/runs/<run_id>/preds.jsonl` per predizioni grezze.
- Nota vault aggiornata con ipotesi, verifica, tesi, conclusioni.

## 10. Fallimento loud

- "Completed" è sbagliato se qualcosa è stato saltato in silenzio.
- Warning catturati e loggati (es. temperature non valide, fallback CPU).
- Incertezza dichiarata, mai nascosta.
