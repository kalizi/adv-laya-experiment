---
tags: [esperimento, superficie, character-level, unicode, confusables, stealth]
stato: fatto
data: 2026-09-30
study_valid: true
---

# E12f — omoglifi per livelli di confusabilità (prova A, B, C)

## La domanda
[[E12e-omoglifi-uts39]] (**prova A**) usò i confusabili che UTS #39 dichiara confondibili: è
l'attacco più furtivo, perché un umano non li distingue. Ma il principio dell'attacco è più
largo di UTS #39: anche un carattere che **si vede a occhio** segue la stessa linea, perché il
principio è *cambio di script*, non *imitazione della forma*. **β per `b`** ne è l'esempio
chiaro: si vede, ma è greco, e il modello byte-level non fa la differenza.

Domanda operativa: **la confusabilità visiva predice l'efficacia dell'attacco?**

| prova | cosa sostituisce | invisibilità |
|---|---|---|
| **A** | confusabili UTS #39 (greco/cirillico) | dichiarata dallo standard |
| **B** | greci plausibili **non** confusabili per UTS #39 | visibile a occhio |
| **C** *(controllo)* | greci senza analogo latino: φ θ ψ ξ Ω | evidentemente diverso |

C è un controllo che ho aggiunto: senza un livello che non assomiglia a niente, "B funziona"
potrebbe voler dire solo "qualsiasi greco funziona". Con C la domanda è chiusa.

Nota sulla prova B: `β` in UTS #39 esiste, ma mappa su **ß** (sessa tedesca), non su `b`. `ε`
mappa su **ꞓ**. Quindi lo standard non dice che siano confondibili con quelle lettere latine,
che è esattamente la distinzione che la prova B testa.

## Setup
N=30, baseline 0.7533 su 150 decisioni, `laya-typed-decisions`, solo lo stato modificato.
Copertura delle tabelle (non coincidono, ed è un dato):

- **A**: 19 lettere — `a b c d e h i j l o p q r s u v w x y`
- **B**: 9 lettere — `b e k l m n t x z`
- **C**: le stesse 19 di A

## Risultati
### Curva di dose
| dose | A: ASR | Δtok | B: ASR | Δtok | C: ASR | Δtok |
|---|---|---|---|---|---|---|
| 25% | 0.3982 | 100 | 0.1858 | 44 | 0.4336 | 101 |
| 50% | 0.5841 | 176 | 0.4513 | 80 | 0.6195 | 178 |
| 100% | **0.6991** | 286 | **0.6814** | 135 | **0.6372** | 292 |

**A dose piena i tre livelli sono indistinguibili**: 0.6991, 0.6814, 0.6372, cioè 79, 77 e 72
flip su 113 decisioni corrette. La differenza è di due flip: rumore.

### Il costo in token, ecco dove i livelli divergono
| prova | ASR/token a dose 100% |
|---|---|
| **B (visibile)** | **0.00505** |
| A (confusabile) | 0.00244 |
| C (controllo) | 0.00218 |

La prova B costa **135 token** dove A costa **286**, a parità di ASR. Il motivo è la lunghezza in
byte: β ε κ λ μ η τ χ ζ sono tutti a 2 byte UTF-8, mentre il set di A include caratteri
cirillici a 3 byte (ҽ, Ӏ, ѡ, ӏ) che il tokenizer<byte> frammenta in più token.

**Quindi il livello invisibile è il meno efficiente.** Non è un difetto di A: è il prezzo
dell'invisibilità, che si paga in byte.

### Confronto within-letter
Quattro lettere sono coperte da A e da B insieme, quindi lo stesso carattere latino viene
sostituito in due modi:

| lettera | A (confusabile) | B (visibile) |
|---|---|---|
| `b` | Ь → ASR 0.0177 | β → ASR 0.0265 |
| `e` | ҽ → ASR 0.2301 | ε → ASR 0.2035 |
| `l` | Ӏ → ASR 0.0619 | λ → ASR 0.0265 |
| `x` | х → ASR 0.0000 | χ → ASR 0.0000 (`x` quasi assente dagli stati) |

Dentro la stessa lettera le differenze sono piccole e non si stabilisce un ordine. Con una
singola lettera il numero di sostituzioni è troppo basso perché la misura separi qualcosa: va
detto, non forzato.

### Il collasso del vocabolario è comune a tutti e tre
| prova | label distinte | sparite |
|---|---|---|
| A | 12 → 7 | `3` `continue` `false` `harmful` `observe` `success` |
| B | 12 → 7 | le stesse sei |
| C | 12 → **5** | le sei più `0` |

Il controllo è il peggiore in collasso, ma non di molto: anche caratteri che non somigliano a
niente fanno sparire metà delle risposte.

**OER 0 in tutti e nove i bracci**, confidenza sul torto 0.19-0.25 ovunque. Nessun livello
produce errori convinti, quindi nessuno soddisfa i .

## Perché B è 2× più economico in token: il test
L'ipotesi era la lunghezza in byte: β ε κ λ μ η τ χ ζ sono a 2 byte UTF-8, mentre il set di A
includeva cirillici a 3 byte (ҽ, Ӏ, ѡ, ӏ) che il tokenizer<byte> frammenta.

Test falsificabile: filtrare A per lunghezza UTF-8 e rimisurare. Funziona, e su un aspetto solo:

| set | lettere | Δtoken | ASR | ASR/token |
|---|---|---|---|---|
| A completa (invisibile) | 19 | 286 | 0.6991 | 0.00244 |
| **A solo greco (2 byte)** | **5** | **100** | 0.5310 | **0.00529** |
| B (visibile) | 9 | 135 | 0.6814 | 0.00505 |
| C (controllo) | 19 | 292 | 0.6372 | 0.00218 |

**L'ipotesi dei byte è confermata sul costo, non sull'efficacia.** Limitare A al greco costa
100 token invece di 286 (2.9× in meno) e l'efficienza per token sale a 0.00529, che pareggia il
livello visibile. Ma l'ASR assoluto scende a 0.5310, perché i confusabili **di soli 2 byte sono
solo 5**: `c→ϲ`, `o→σ`, `p→ϱ`, `u→υ`, `v→ν`. Tutti gli altri cross-script che UTS #39 dichiara
confondibili sono cirillici a 3 byte.

Quindi *invisibilità + costo basso* esiste, ma solo per 5 lettere, e con un ASR più basso
(0.53 contro 0.70). La conclusione operativa è un'altra, e più semplice:

> **usare caratteri a 2 byte, confusabili o no**. L'efficienza per token è ~0.005 con
> caratteri a 2 byte e ~0.002 con caratteri a 3, indipendentemente dalla somiglianza. I
> 3 byte sprecano budget, e il set invisibile di UTS #39 è soprattutto 3 byte per quello.

## Verdetto
- **La confusabilità visiva non predice l'efficacia.** A dose piena: 0.6991, 0.6814, 0.6372.
  Il modello è ingannato dal **cambio di script**, non dalla somiglianza della forma. Anche
  φ per `a` (prova C, 25% → 0.4336) funziona quanto α per `a`.
- **L'invisibilità è gratis, ma non è nemmeno necessaria.** Se serve, la prova A la dà. Se non
  serve, si prende il livello più efficiente, che è il più *visibile*. Non c'è trade-off fra
  furtività ed efficacia: c'è solo fra furtività ed efficienza per token, e la furtività la
  paga.
- **Non serve copertura ampia**: 9 lettere (prova B) bastano a eguagliare 19 lettere (prova A).
  L'attacco colpisce quello che tocca, non la copertura.
- **Il meccanismo resta la corruzione in place** (confermato in [[E12d-omoglifi]]: spazi
  distribuiti ≈ omoglifi). I livelli di confusabilità sono un grado, non una soglia.
- **Il collasso del vocabolario non dipende dalla somiglianza**: 12 → 7 label con omoglifi
  invisibili, 12 → 5 con caratteri che non somigliano a nulla. Perde la capacità di esprimere
  metà delle risposte indipendentemente da come l'attacco appaia.

## Cosa resta
- **Verificare l'invisibilità su un umano vero**: sia A sia B sono dichiarazioni di
  confusabilità, A normativa e B a occhio. Un campione di revisori chiuderebbe il cerchio, e
  qui non è misurabile.
- Aggirare il vincolo di copertura con sostituzione **multi-carattere** (`mn` → `mн`): aggira
  sia il limite delle 5 lettere greche sia il costo dei 3 byte, se i byte di una sequenza
  brevissima venissero tokenizzati insieme.

## Note collegate
[[E12e-omoglifi-uts39]] · [[E12d-omoglifi]] · [[E12c-simbolo]] · [[E03-option-injection]] · [[E04-truncation]] ·  · [[superficie-attacco]] · [[esperimenti]]

## Output
Nessun file in `results/`: sonda one-off.
```bash
uv run python tools/e12f_homoglyph_tiers.py   # A, B, C e confronto within-letter
uv run python tools/e12f_greek_only.py        # test dei 2 byte
```

## Archiviata il 2026-10-05, con la stessa limitazione di E12d ed E12i

`results/E12f_homoglyph_tiers.json`, 17 righe con `level`, `dose`, `asr_denominator` e
`oer_adv_conf_075` per riga. I numeri **riproducono la tabella qui sopra** (A 0.6991, B 0.6814,
C 0.6372 a dose piena, 79/77/72 su 113), quindi la nota aveva ragione e il paper attribuiva qui i
valori di un altro esperimento.

Come E12d ed E12i, sono **sintesi senza `io.RunStore`**: `verify_results.py` le conta fra le righe
senza run_id e non le ricalcola dai record grezzi. Il controllo che le copre è
`check_claim_numbers.py`.
