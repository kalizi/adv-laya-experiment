---
tags: [metodologia, performance, velocita, tradeoff, hardware]
stato: fatto
data: 2026-10-01
---

# Performance e tradeoff: sicurezza contro velocità

Misure su una RTX 3080 (10 GB, capability 8.6), torch 2.14+cu130, bf16, `batch_size=16`,
`chunk=32`. L'unita' è il **millisecondo per decisione**, non il millisecondo per batch.

## L'unita' conta, e il device va registrato

Un p50 per batch non è confrontabile: si muove con `batch_size`, con quante domande porta
un caso e con la **lunghezza** delle sequenze. Misurato su `typed-decisions` (400 casi, stati
medi 189 token, chunk da 32): i primi 32 casi danno chunk da 197 token, la run intera
contiene chunk fino a **570**. Lo stesso carico, misurato sul primo pezzo, dava 4.67 ms/decisione
contro i 10.76 della run completa: **fattore 2.3 da solo**. Un numero di velocita' preso su un
campione favorevole è peggio di nessun numero.

`ms_per_decision` è l'unica unita' che tiene insieme modelli, dataset e K diversi, ed è quella
che serve a un Pareto contro ASR e OER. Va in `metrics.core.throughput`, calcolata sui
**decisioni dentro i batch cronometrati** (non sul totale della run: dopo un resume la
differenza sarebbe la frazione riusata).

> **Il device va scritto per campione, e `env_fingerprint` non basta.** Il suo `device`
> riporta la GPU se CUDA è disponibile *in generale*, quindi un run che laya ha demoto a CPU
> a metà (il fallback silenzioso dell'OOM, ~10x piu' lento) scrive lo stesso
> "NVIDIA GeForce RTX 3080" di un run sano. Misurato: due run con config identico
> (stesso checkpoint, stesso dataset, stesso K, stessa `dataset_sha256`) avevano p50 558 ms e
> 5551 ms, entrambi dichiarati su RTX 3080. Da ora ogni campione porta `device` e `dtype`, e
> il blocco `latency` elenca i device visti: piu' di uno dentro una run rende il fallback
> visibile invece che nascosto.

## Il tradeoff, sui tre checkpoint della famiglia

`typed-decisions` test, 400 casi x 5 domande = 2000 decisioni, da `results/E01_baseline.json`:

| checkpoint | encoder | acc | OER@0.75 pulito | ms/decisione | decisioni/s |
|---|---|---|---|---|---|
| `laya-typed-decisions` | ModernBERT-large | **0.7670** | **0.0000** | 10.759 | 93.0 |
| `laya` | ModernBERT-large | 0.3595 | 0.0435 | 10.424 | 95.9 |
| `laya-multilingual` | mmBERT-base | 0.3505 | 0.1430 | **5.187** | **192.8** |

Il modello piu' accurato è **il piu' lento**, e il piu' veloce è **2.07x** piu' rapido ma ha
l'accuracy peggiore e sbaglia gia' convinto nel 14.3% dei casi *prima* di qualsiasi attacco
(OER@0.75 pulito 0.143 contro 0.000).

Il riscontro che interessa per la campagna: **il modello veloce è anche quello piu' attaccabile.**
Su MASSIVE-it `laya-multilingual` ha ASR **0.9981** e OER@0.75 **0.995** sotto attacco, contro
ASR 0.86-0.90 del checkpoint specialista che è 2x piu' lento. Il ramo "compra quello veloce"
del Pareto è anche il ramo "è gia' rotto": non è un tradeoff, è un angolo peggiore in
entrambe le dimensioni. Il ramo "compra quello preciso" costa 2.07x e non è piu' sicuro.

## Cosa c'è da ottimizzare, misurato

Il forward è compute-bound e gia' al roofline bf16 della 3080: 772 ms per 160 righe da 197
token su ModernBERT-large torna con il conto dei FLOP (circa 44 TFLOP a ~50% di utilisatione).
**Non c'è un kernel piu' veloce disponibile su questa macchina**, e la tabella dice perché:

| leva | guadagno | costo |
|---|---|---|
| **cache del tokenizer** | **1.7%** su typed-decisions, **8.6%** su MASSIVE | nessuno, id bit-identici |
| `torch.compile` | **31%** (1.44x) | ~55s di warmup, numeri diversi |
| raggruppamento per lunghezza | 9.4-9.9% dei token paddati | cambia la composizione del batch |
| `batch_size` 16 → 32 | dentro il rumore (±3%) | non ha piu' fatto scattare l'OOM |
| `flash-attn`, `tilelang` | non installati, `nvcc` assente in PATH | — |
| encoder piu' piccolo | **2.07x** | −42pp di accuracy |

### La cache del tokenizer (fatta, `CachedTokenizer`)

`build_sequence` ricalcola l'intera testa (testo della domanda piu' tutte le opzioni) **per ogni
stato**, anche se la testa dipende solo dalla domanda. A K=16 quella testa sono 95 dei 111
token, e 17 delle 18 chiamate al tokenizer di uno stato sono identiche a quelle dello stato
 precedente. Il proxy memoizza solo `__call__` e inoltra tutto il resto, quindi non è una
reimplementazione: un cambiamento nel modo in cui laya usa il tokenizer compare come attributo
mancante, non come una codifica sbagliata e silenziosa.

Verificato sul modello reale, run completa: `input_ids`, `attention_mask` e `marker_pos`
**bit-identici**, guadagno 1.7% (typed-decisions) e 8.6% (MASSIVE). Il guadagno scala con
quanto il forward è amortizzato: su typed-decisions il forward è gia' ~90% del tempo, su
MASSIVE gli stati sono 12 token e la parte Python pesa di più. Il "99.6% nel forward" che
`CLAUDE.md` riportava vale per il primo caso e non per il secondo.

### `torch.compile` (esposto come `--compile`, non il default)

`laya.load()` non inoltra `compile`: esiste solo su `Agent`, quindi fino a oggi l'opzione era
irraggiungibile dal progetto. Ora `--compile` fa le due righe che `Agent(compile=True)` fa.
Misurato: 5.275 → 3.652 ms/decisione (1.44x), ma **solo 2 delle 15 confidence restano identiche**
(max |Δ| 0.0031) e il warmup costa 55s, perche' 96 casi reali producono 71 shape di sequenza
diverse. Fusione riordina le riduzioni, quindi cambia i numeri: per questo `compiled` entra nel
config del run e i due run non condividono il `run_id`. Un run compilato e uno eager non si
confrontano, e va detto quando si pubblica.

### Raggruppamento per lunghezza (proposto, non fatto)

Ordinare i casi per lunghezza dello stato prima di formare i batch taglia il 9.4% (MASSIVE) e
il 9.9% (typed-decisions) dei token paddati: lo spreco di padding scende dal 14.5% al 3.7% e
dal 33.4% al 20.3%. Non è stato fatto perche' cambia la composizione del batch, quindi cambia
le previsioni in bf16, e il progetto tratta quello come un ostacolo alla riproducibilita': 9% non
valgono un esperimento che non è confrontabile con nessuno. Se serve, va fatto come `--sort
by-length` con `sorted` dentro il config del run, non come default silenzioso.

## Cosa non è una leva

Il fallback CPU di laya è una trappola, non un prezzo: 10x piu' lento, e fino a oggi
indistinguibile dai numeri salvati. Se compare `devices` con piu' di un elemento dentro una run,
quel run non è confrontabile con nessuno e va rifatto con `--batch-size` piu' piccolo.

## Note collegate

[[E01-baseline]] · [[E03-option-injection]] · [[configurazione-ambiente]] · [[metriche-attacco]] ·
 · [[metriche-attacco]] · [[principi-esperimentali]]
