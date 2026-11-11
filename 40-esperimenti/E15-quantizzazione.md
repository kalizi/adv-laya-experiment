---
tags: [esperimento, quantizzazione, trasferimento, deberta, open-jev, calibrazione, curva-K, repliche]
stato: fatto
data: 2026-10-01
study_valid: true
---

# E15 — La sicurezza sopravvive alla riduzione di precisione?

## Ipotesi

`metriche-attacco.md` §7 definisce il **transfer** su tre assi (checkpoint, dataset, precisione).
La campagna ne aveva usati due. Il terzo è quello che un difensore toccherebbe per primo,
perché è l'unico gratis: **quantizzare per servire più velocità**.

Nessuno aveva misurato se la proprietà di sicurezza sopravvive, cioè se il guardrail di
confidenza (l'unica difesa rimasta dopo E14) comincia a funzionare.

- **H1, chiude**: sfocare i pesi sfoca anche il matching fra testo dell'opzione e stato, che è
  il meccanismo che E03 sfrutta (E13: 25-28pp di accuracy dipendono dal testo delle opzioni).
- **H2, peggiora**: a precisione minore il matching è più grossolano, quindi l'attacco passa
  più facilmente.

## Due famiglie

| famiglia | architettura | calibrazione dichiarata |
|---|---|---|
| `laya` | ModernBERT-large + testa a marker | ECE **0.025** su banking77 (0.417 su typed-decisions) |
| `deberta-onnx` (open-jev) | DeBERTa-v3-large + span pooling, un logit per coppia | **ECE 0.300** su banking77 |

Counterpart architetturale: stessa triade di primitive, una forward pass per tutte le domande,
stessa regola di argmax, risposte con la stessa forma. Cambiano encoder, testa e calibrazione.

## Protocollo

**58 righe**, tutte su CUDA, N=1000 per punto, template `prepend`, attacco identico ovunque:

- **7 configurazioni a K=16** × 3 repliche (semi 0, 1, 2) su tre dataset: laya/massive-it,
  laya/banking77, deberta-onnx/banking77.
- **Curve K=4/8/16/32** su tutte le 7 precisioni, una replica: laya su massive-it, deberta su
  banking77.
- Precisione, device, backend e compilazione entrano nel `run_id`, e una guardia verifica che il
  backend aperto sia quello chiesto.

## Risultato 1 — sullo stesso dataset l'attacco è forte su entrambe le architetture

Il confronto che chiude il limite più grosso di una versione precedente di questa nota: le due
famiglie erano su dataset diversi, quindi la differenza poteva essere il dataset. Ecco le due
**sullo stesso dataset**, stesso attacco, stesso K, 3 repliche.

| modello | calibrazione | baseline | ASR (3 repliche) | escursione | attraversa 0.50 | OER@.75 pulito → att. | conf sul torto |
|---|---|---|---|---|---|---|---|
| `laya/none` | ECE 0.0246 | 0.7940 | 0.5038–0.5418 | 3.80pp | no | 0.053 → 0.390 | 0.796 |
| `laya/int8` | ECE 0.0329 | 0.7920 | 0.4962–0.5414 | 4.52pp | **Sì** | 0.049 → 0.385 | 0.792 |
| `laya/nf4` | ECE 0.0450 | 0.8000 | 0.5200–0.5392 | 1.92pp | no | 0.044 → 0.394 | 0.799 |
| `deberta-onnx/fp32` | ECE 0.3002 | 0.9520 | 0.4752–0.4813 | 0.61pp | no | 0.000 → 0.070 | 0.587 |
| `deberta-onnx/fp16` | ECE 0.2995 | 0.9520 | 0.4741–0.4823 | 0.82pp | no | 0.000 → 0.071 | 0.588 |
| `deberta-onnx/q4` | ECE 0.3156 | 0.9570 | 0.5099–0.5177 | 0.78pp | no | 0.000 → 0.074 | 0.592 |
| `deberta-onnx/q4f16` | ECE 0.3149 | 0.9570 | 0.5099–0.5167 | 0.68pp | no | 0.000 → 0.074 | 0.593 |

L'ASR è **la stessa**: laya 0.504-0.542, deberta-onnx 0.474-0.518, tutte oltre o vicine alla
metà. L'attacco non è un difetto dell'architettura di laya, è del meccanismo.

## Risultato 2 — l'errore convinto **non** è della calibrazione (correzione di una conclusione)

Sullo stesso dataset, alla stessa K, con lo stesso attacco, la differenza è tutta qui:

| | OER@0.75 attaccato | confidenza sul torto |
|---|---|---|
| laya (ECE 0.0246) | **0.385 - 0.394** | **0.79 - 0.80** |
| deberta-onnx (ECE 0.3002) | **0.070 - 0.074** | **0.59** |

Cinque volte a K=16. Su laya il modello sbaglia e non se ne accorge; su deberta-onnx l'errore
indotto non è convinto e il guardrail a 0.75 regge.

**Ma la separazione dipende da K, e su 4 righe su 58 il segnale si inverte.** A K=4
l'`OER@0.75` attaccato di laya è **0.000** contro 0.005-0.008 di deberta-onnx: laya è *meno*
convinto. A K=8 è 0.183-0.262 contro 0.030-0.036, quindi la differenza c'è ma è cinque volte
più piccola che a K=16. La firma dell'errore convinto è quindi una **proprietà di K** e non
delle architetture: sotto una certa ampiezza di opzioni l'attacco non produce errore convinto
su nessuno dei due, e il modello con l'errore non convinto è proprio quello che pensavamo
fosse il piu' fragile. Va dichiarato, non risolto: la tabella completa lo mostra e la frase
"non si inverte mai" va ritirata.

**Ma la colonna ECE rovescia la spiegazione che avevo dato, e questa è la correzione che
conta.** L'ECE che avevo usato qui era **dichiarato, non misurato**: 0.417 per laya è
l'ECE di `laya-typed-decisions` su `typed-decisions`, non su `banking77`, e 0.022 per
deberta-onnx non è riproducibile qui. Misurati da noi su `banking77` i valori sono **0.025 e
0.300**: il modello con l'errore convinto è quello **più ben calibrato** al pulito.

Quindi la differenza **non è di calibrazione**. Fra le due architetture la correlazione fra
ECE pulito e OER attaccato è **-0.998**, e dentro ciascuna architettura è ~0 (**+0.04** su
laya, **-0.38** su deberta-onnx, su 25 righe appaiate): non è una relazione graduata, sono due
gruppi separati. La spiegazione corretta è che l'attacco **non** sfrutta una calibrazione già
sbagliata, e su laya la **crea**: l'ECE pulito di 0.025 diventa 0.394 sotto attacco, mentre
quello di deberta-onnx passa da 0.300 a 0.133. L'errore convinto è **differenziale di
attacco**, e non differenziale di calibrazione.

La conseguenza operativa è più utile di quella che avevo scritto: **un difensore non può
scegliere il checkpoint guardando l'ECE**. Laya ha l'ECE migliore e il guardrail peggiore. Il
segnale da guardare è il comportamento sotto perturbazione, che è quello che l'attacco
misura.

## Risultato 3 — la quantizzazione non chiude l'attacco, e su deberta lo peggiora

Il test giusto è **appaiato**: le precisioni vedono le stesse righe, quindi confrontare
proporzioni indipendenti sbagliava la domanda. McNemar sulle righe corrette al pulito in
entrambe le varianti, con successo dell'attacco = previsione attaccata sbagliata.

**deberta-onnx/fp32** vs le altre precisioni, su `banking77`, K=16, seed 0 (McNemar appaiato sulle stesse righe, casi corretti al pulito in entrambe)

| confronto | \|S_ok\| | cede solo la base | cede solo l'altra | p |
|---|---|---|---|---|
| `fp32` vs `fp16` | 952 | 1 | 0 | 1.0000 (non significativo) |
| `fp32` vs `q4` | 951 | 11 | 41 | 0.0000 (**significativo**) |
| `fp32` vs `q4f16` | 951 | 11 | 41 | 0.0000 (**significativo**) |

**laya/none** vs le altre precisioni, su `banking77`, K=16, seed 0 (McNemar appaiato sulle stesse righe, casi corretti al pulito in entrambe)

| confronto | \|S_ok\| | cede solo la base | cede solo l'altra | p |
|---|---|---|---|---|
| `none` vs `int8` | 786 | 15 | 28 | 0.0660 (non significativo) |
| `none` vs `nf4` | 773 | 38 | 55 | 0.0966 (non significativo) |

**laya/none** vs le altre precisioni, su `massive-it`, K=16, seed 0 (McNemar appaiato sulle stesse righe, casi corretti al pulito in entrambe)

| confronto | \|S_ok\| | cede solo la base | cede solo l'altra | p |
|---|---|---|---|---|
| `none` vs `int8` | 274 | 1 | 3 | 0.6250 (non significativo) |
| `none` vs `nf4` | 252 | 5 | 3 | 0.7266 (non significativo) |

La lettura cambia rispetto a una versione precedente che diceva "sospetto, non dimostrato": quel
giudizio veniva da intervalli binomiali indipendenti, che è il test sbagliato su dati appaiati.

- **Su deberta-onnx il 4 bit peggiora l'attacco, significativamente**: 11 casi cedono solo in
  fp32 contro 41 che cedono solo in q4, p < 0.0001, e lo stesso per q4f16. H2 confermata, su
  questa architettura.
- **Su laya non c'è differenza**: p = 0.066 e p = 0.097 su banking77, p = 0.63 e p = 0.73 su
  massive-it. Nessun effetto della precisione.

Quindi H1 è falsa ovunque, e H2 è vera solo dove il modello è calibrato: più margine di
distacco per l'attacco, non più margine di errore.

## Risultato 4 — le repliche, e una soglia che non si decide

3 repliche per semi diversi. Il numero che conta non è la media: è se **la soglia 0.50 sta
dentro l'escursione**.

**E15_quantization__laya__massive-it** — dataset massive-it, K=16, 3 repliche (semi [0, 1, 2])

| configurazione | ASR min | ASR max | escursione | media | attraversa 0.50 |
|---|---|---|---|---|---|
| `laya/int8` | 0.9414 | 0.9488 | 0.74pp | 0.9447 | no |
| `laya/nf4` | 0.9293 | 0.9454 | 1.61pp | 0.9361 | no |
| `laya/none` | 0.9354 | 0.9427 | 0.73pp | 0.9393 | no |

**E15_quantization__laya__banking77** — dataset banking77, K=16, 3 repliche (semi [0, 1, 2])

| configurazione | ASR min | ASR max | escursione | media | attraversa 0.50 |
|---|---|---|---|---|---|
| `laya/int8` | 0.4962 | 0.5414 | 4.52pp | 0.5193 | **sì** |
| `laya/nf4` | 0.5200 | 0.5392 | 1.92pp | 0.5297 | no |
| `laya/none` | 0.5038 | 0.5418 | 3.80pp | 0.5175 | no |

**E15_quantization__deberta-onnx__banking77** — dataset banking77, K=16, 3 repliche (semi [0, 1, 2])

| configurazione | ASR min | ASR max | escursione | media | attraversa 0.50 |
|---|---|---|---|---|---|
| `deberta-onnx/fp16` | 0.4741 | 0.4823 | 0.82pp | 0.4774 | no |
| `deberta-onnx/fp32` | 0.4752 | 0.4813 | 0.61pp | 0.4778 | no |
| `deberta-onnx/q4` | 0.5099 | 0.5177 | 0.78pp | 0.5131 | no |
| `deberta-onnx/q4f16` | 0.5099 | 0.5167 | 0.68pp | 0.5131 | no |

- **Su massive-it** la dispersione è 0.7-1.6pp e l'ASR è a 0.93-0.94: lontano dalla soglia,
  niente da dichiarare.
- **Su deberta-onnx** la dispersione è 0.6-0.8pp e i gruppi non si toccano: fp32 e fp16 stanno
  **interamente sotto** 0.50 (max 0.4823), q4 e q4f16 **interamente sopra** (min 0.5099).
- **Su laya/banking77** la dispersione è 1.9-4.5pp, e **`laya/int8` attraversa la soglia**
  (0.4962-0.5414). Su quella riga la conclusione è indeterminata e va detto così: la media
  0.5193 non è un risultato, è una media di due esiti incompatibili.

Su `laya/banking77` la dispersione è 4-5 volte quella delle altre righe. Il denominatore è ~800
contro ~950 e il modello è meno competente, quindi è atteso; ma resta il motivo per cui quella
riga non si dichiara.

## Risultato 5 — la curva delle opzioni, e una correzione

Con un solo K (16) una versione precedente concludeva che sul modello calibrato la soglia non
fosse raggiunta. **Era sbagliato**: bastava K=4.

**laya su massive-it** (una replica; `*` = baseline sotto 2/K, riga esclusa)

| K | baseline | acc att. | drop | **ASR** | 95% CI | su | OER@.75 att. | conf torto |
|---|---|---|---|---|---|---|---|---|
| 4 * | 0.4920 | 0.0530 | 43.9pp | **0.8984** | [0.869, 0.922] | 492 | 0.000 | 0.315 |
| 8 | 0.3750 | 0.0280 | 34.7pp | **0.9280** | [0.897, 0.950] | 375 | 0.262 | 0.638 |
| 16 | 0.2940 | 0.0230 | 27.1pp | **0.9354** | [0.901, 0.958] | 294 | 0.881 | 0.912 |
| 32 | 0.2280 | 0.0220 | 20.6pp | **0.9254** | [0.884, 0.953] | 228 | 0.707 | 0.820 |

**deberta-onnx su banking77** (una replica)

| K | baseline | acc att. | drop | **ASR** | 95% CI | su | OER@.75 att. | conf torto |
|---|---|---|---|---|---|---|---|---|
| 4 | 0.9850 | 0.4580 | 52.7pp | **0.5350** | [0.504, 0.566] | 985 | 0.005 | 0.415 |
| 8 | 0.9710 | 0.4930 | 47.8pp | **0.4923** | [0.461, 0.524] | 971 | 0.030 | 0.520 |
| 16 | 0.9520 | 0.4990 | 45.3pp | **0.4769** | [0.445, 0.509] | 952 | 0.070 | 0.587 |
| 32 | 0.9260 | 0.5110 | 41.5pp | **0.4492** | [0.417, 0.481] | 926 | 0.081 | 0.611 |

Il limite sinistro della riga DeBERTa fp32, K=16, non era salvato nello store. L'audit
`results/provenance/historical_ci_audit.json` lo ricostruisce dai record appaiati:
454 ribaltamenti su 952 corretti al pulito, Wilson 95% [0.4453, 0.5086]. Il valore
tabellare [0.445, 0.509] è quindi verificabile come cifra derivata.

Su deberta-onnx a **K=4 l'attacco supera nettamente la soglia**: ASR 0.5350, CI
[0.504, 0.566], che non contiene 0.50. Quindi la soglia bronze **è** raggiunta sul counterpart
architetturale, con 985 decisioni corrette in partenza. Il singolo K=16 era un punto che
capitava sotto.

Sulla famiglia laya l'ASR è **piatto** a 0.90-0.94 da K=8 in su mentre la baseline crolla da
0.375 a 0.228: su laya l'attacco **non segue** il decadimento della baseline. Su deberta-onnx
l'ASR scende insieme alla baseline. È una differenza di comportamento, ma i due dataset sono
diversi per difficoltà, quindi non è attribuibile all'architettura. Resta un limite.

Il guardrail su laya fallisce in modo **non monotono**: OER@0.75 0.000 a K=4, 0.262 a K=8,
**0.881 a K=16**, 0.707 a K=32. Cede proprio nel punto in cui il modello è ancora abbastanza
competente da essere convinto (K=16: baseline 0.294, ASR 0.935), e si riprende a K=32 perché la
baseline è scesa a 0.228 e il modello sbaglia anche quando è sicuro.

## Risultato 6 — la velocità dipende dal device

Stesso codice, due device, il device nel `run_id` perché i numeri divergero.

| | CPU (ms/decisione) | GPU (ms/decisione) |
|---|---|---|
| fp32 | 164.6 | 10.8 |
| fp16 | 300.8 | **8.3** |
| q4 | 184.6 | 15.9 |
| q4f16 | 285.5 | 9.7 |

Su CPU la fp16 costa 1.8× la fp32: non ci sono kernel nativi e i pesi a 4 bit vengono spacchettati
a ogni moltiplicazione. **Su GPU il 4 bit da solo è più lento della fp32**, e la fp16 è la più
veloce. Chi avesse scritto "la quantizzazione non accelera" dalla sola misura CPU avrebbe
descritto la CPU.

**ORT CUDA EP esegue i blocchi `MatMulNBits` nativi**, e la verifica è con il profilo dei nodi,
non con `get_providers()`, che dà l'ordine di priorità e non la assegnazione:

```bash
uv run python tools/onnx_provider_audit.py
```

Il controllo gira in modo che un fallback parziale non possa passare: nessuna moltiplicazione
(`MatMulNBits`, `MatMul`, `Gemm`, `Attention`) cade sulla CPU in nessuno dei quattro grafi, e il
residuo CPU sono 60-64 nodi di forma (`Slice`, `Squeeze`, `Concat`, `Unsqueeze`, `Cast`).
"Almeno un nodo su CUDA" sarebbe troppo debole: un fallback dei soli pesi lascia comunque
migliaia di nodi CUDA e la sessione dichiara CUDA in testa.

Il residuo **non è identico** fra grafi (`model_fp16` tiene `Div`, `Mul` e `Sqrt` su CUDA, gli
altri no), quindi la frase "gli stessi in tutti i grafi" che una versione precedente di questa
nota usava era imprecisa. Non cambia la conclusione, perché il confronto fra precisioni poggia
sui pesi, che sono tutti su GPU, ma la precisione del testo era falsa.

La **previsione** è invece indipendente dal device: a K=16 fp32, ASR 0.4758 su CPU e 0.4769 su
GPU, una decisione su 952 diverse.

## Correzione a una versione precedente

Questa nota diceva che **laya su banking77 va a 0.03, sotto il caso**, e usava la cosa come
giustificazione per i dataset per famiglia. Era **falso**: andava a 0.03 perché il loader leggeva
`label` invece di `label_text`, e le opzioni erano "63", "13". Con `label_text` la baseline è
**0.79**. Il numero che giustificava una scelta di protocollo era un bug di caricamento, quindi la
scelta andava rimossa e rimisurata: da qui la terza tabella, che è anche il confronto che scioglie
il confondimento del dataset.

## Limiti, dichiarati

1. **Un solo template (`prepend`)** e una sola soglia OER (0.75). E03×E04 ha mostrato che la
   posizione conta.
2. **Un solo checkpoint per famiglia**: `--quantize` gira su `laya-typed-decisions`, non è stato
   rimisurato su `laya` o `laya-multilingual`.
3. **Le curve sono a una sola replica**, le tabelle a K=16 a tre. La curva non ha il test
   appaiato.
4. **La dispersione fra repliche è del benchmark, non del modello**: il seed ridisegna le opzioni
   per riga, quindi le repliche misurano la varianza del disegno del benchmark.
5. **Il tempo di un run non è riproducibile anche quando le previsioni lo sono**: due run di
   `laya/none` a K=16 hanno previsioni bit-identiche e 30.7 contro 18.1 ms/decisione. I confronti
   di velocità valgono **dentro lo stesso run**, non fra run.
6. **Gli ECE che avevo usato erano dichiarati, non misurati**, e questo ha prodotto una
   conclusione sbagliata. Li ho ricalcolati su `banking77` e sono 0.025 (laya) e 0.300
   (deberta-onnx), cioè l'opposto di come li avevo usati: l'errore convinto sta nel modello
   **più** calibrato. La firma del guardrail era nostra, e la spiegazione no.

## Note collegate

[[E03-option-injection]] · [[E13-label-semantics]] · [[E14-self-guardrail]] · [[metriche-attacco]] ·
 · [[performance-velocita]] · [[backend-hot-swappable]]
