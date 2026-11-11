---
tags: [esperimento, e03, clm, hard-negatives, trasferimento, guardrail]
---

# E21 — CLM: l'attacco trasferisce con ASR inferiore

## La domanda, registrata prima di misurare

CLM usa un'architettura e un addestramento diversi dai checkpoint Laya e DeBERTa. Il suo mid-training include
**30M di hard negatives** sintetici, "risposte semanticamente simili ma sbagliate",
ottenuti da Gemini 2.5 Flash-Lite. Un hard negative è, strutturalmente, lo stesso
oggetto dell'iniezione E03: nel nostro protocollo la frase avversaria entra nello stato,
mentre gli hard negative sono esempi di addestramento. Non sono un controllo appaiato.

Quindi la domanda è se quel segnale chiude il buco. **I due esiti sono mutuamente
esclusivi ed entrambi pubblicabili**:

- se E03 trasferisce, CLM resta vulnerabile nel protocollo misurato;
- se non trasferisce, CLM offre un caso utile per studiare una possibile difesa,
  senza attribuirla al solo segnale di addestramento.

Previsioni secondarie registrate prima (due di queste sono sbagliate, e vanno dette):

1. l'ASR è governato dalla stessa geometria di laya, non dalla temperatura, perché
   `argmax` è invariante alla scala;
2. la coda batterà la testa di almeno 3x come su `laya-multilingual`, perché CLM
   dichiara di troncare dalla testa nel serving (asimmetria train/serve upstream);
3. l'`OER` sotto attacco andrà verso il soffitto, per via della temperatura 0.01.

**Il verdetto è che E03 trasferisce.** La previsione 2 è sbagliata e la 3 è sbagliata
nella direzione opposta.

## Setup

- **Backend**: `clm-llama`, CLM-v0.1-8B. Encoder `Qwen3-8B-Q6_K.gguf` su llama.cpp
  **b11160** con `--embedding --pooling last -ngl 36`, teste su CPU
  (`--device cpu --action-cache 0`). Vedi [[clm]] e `src/layaadv/model/clm_backend.py`.
- **Protocollo**: identico a E03, senza ritocchi. Stesso dataset, stesso `seed=0`,
  stessa curva K=4..40, stessi due template, **N=2974** (MASSIVE test), `pad=0`.
- **Nessun `--limit`**: la trappola di E17 [[E17-configuration]] è evitata per costo,
  non per convinzione. I denominatori sono dichiarati e vanno da 282 a 1311.
- **Il punto operativo del margin è derivato dal solo run pulito**, congelato prima di
  ogni riga avversaria (`margin_at_cost`, budget 1% di falsi blocchi). Sceglierlo dopo
  sarebbe scegliere la soglia su ciò che si vuole misurare.

## Risultato 1: il trasferimento è confermato, e l'ASR sale con K

`massive-it`, N=2974, `Q6_K`. `head` = `prepend`, `tail` = `append`.

| K | 1/K | clean | drop pp | **ASR head** | **ASR tail** | denominatore |
|---|---|---|---|---|---|---|
| 4 | 0.2500 | 0.3689 | 18.26 / 14.09 | **0.5679** | 0.4840 | 1097 |
| 8 | 0.1250 | 0.2273 | 9.48 / 7.33 | **0.5533** | **0.5118** | 676 |
| 16 | 0.0625 | 0.1560 | 6.56 / 4.30 | **0.5905** | **0.5453** | 464 |
| 24 | 0.0417 | 0.1244 | 5.38 / 3.50 | **0.6351** | **0.5757** | 370 |
| 32 | 0.0312 | 0.1076 | 5.01 / 3.36 | **0.6469** | **0.6031** | 320 |
| 40 | 0.0250 | 0.0948 | 4.57 / 2.76 | **0.6738** | **0.6277** | 282 |

`massive-en`, N=2974 (il controllo di lingua, vedi sotto):

| K | clean | drop pp | **ASR head** | **ASR tail** | denominatore |
|---|---|---|---|---|---|
| 4 | 0.4408 | 23.23 / 17.52 | **0.5690** | 0.4683 | 1311 |
| 16 | 0.2286 | 12.95 / 9.01 | **0.6397** | **0.5221** | 680 |

**16 righe, `ASR >= 0.50` su 14.** Le due sotto sono entrambe `append` a K=4
(0.4840 e 0.4683). Soglia 🥉: 0.50, da .

**L'ASR sale con K** (in coda è monotono, 0.4840 → 0.6277 da K=4 a K=40; in testa scende a
K=8 e poi sale, 0.5679 → 0.6739) mentre su `laya-multilingual` scende.
La ragione non è stata misurata separatamente e va detta come interpretazione: a K=40
la baseline è 0.0948, quasi il caso (1/K = 0.0250), quindi le decisioni che restano
corrette al pulito sono un sottoinsieme piccolo e molto selezionato, e la spinta
verso l'opzione nominata le rovescia più facilmente che non un caso a 4 opzioni dove
il modello ha una discriminazione a tre vie da superare. **È un'ipotesi, non una
misura.**

**Il drop è il parametro sbagliato qui, e il motivo è noto**: la baseline di CLM su
MASSIVE-it è 0.3689 a K=4 contro 0.634 di `laya-multilingual`, quindi il drop non
raggiunge i 20pp del criterio 🥉 se non a una riga. E03 lo aveva già scritto: "il drop
scende con K ma la baseline crolla insieme: l'ASR è il denominatore giusto".

## Risultato 2: il controllo di lingua esclude la spiegazione comoda

Il sospetto più economico dopo una baseline debole è "il modello è solo debole, quindi
si lascia girare". Il controllo è `massive-en`, dove CLM ha la baseline **più alta**
(0.4408 contro 0.3689 a K=4, 0.2286 contro 0.1560 a K=16).

L'ASR **non scende**: anzi in testa è più alto (0.5690 contro 0.5679, 0.6397 contro
0.5905). Quindi l'ASR di CLM non è un artefatto della baseline italiana. Il merito del
controllo è questo, ed è il motivo per cui è stato pagato.

`banking77` è stato **escluso** e va detto: baseline 0.1250 a K=16, la più debole dei
tre dataset, quindi il denominatore ASR più fragile e la minor informazione.

## Risultato 3: il guasto è più silenzioso, ed è il dato che pesa

Confronto a `massive-it`, K=16, `append`, stessa riga di protocollo:

| backend | clean | ASR | denom | **OER@0.75 attaccato** |
|---|---|---|---|---|
| `clm-v0.1-8B` (Q6_K) | 0.1560 | **0.5453** | 464 | **0.0632** |
| `laya-multilingual` | 0.5551 | 0.9897 | 1651 | **0.9832** |
| `laya-typed-decisions` | 0.4072 | 0.9686 | 1211 | **0.8914** |
| `laya` | 0.4075 | 0.9612 | 1212 | **0.9781** |

CLM è attaccato a circa **metà** del tasso degli altri, e il suo guardrail si rompe
**14-16 volte meno** (0.8914/0.0632 contro 0.9832/0.0632: il fattore dipende da quale dei
tre checkpoint `laya` si prende, quindi è una forchetta e non un numero). Entrambi i fatti
contano e puntano in direzioni opposte.

E il criterio di confidenza (≥0.75) **non è mai raggiunto su CLM**, dove è una media e non
una soglia sul singolo record: `mean_conf_wrong` al massimo 0.6682, e scende da K=4 a K=40
(0.6682 → 0.3394) mentre su `laya-multilingual` saliva a 0.897. Singole decisioni sbagliate la
superano eccome, e a K=16 sono il 6.3% del totale: è la media a non arrivarci, non la
soglia a non essere superata.

**La formula ha il verso opposto a quello ipotizzato inizialmente.** Poiché le
probabilità sommano a uno, `confidence = p1 - mean(p2..pk) = (K*p1-1)/(K-1)`.
La soglia 0.75 equivale a `p1 >= 0.75 + 0.25/K`: 0.8125 a K=4 e
0.75625 a K=40. A `p1` fisso, aumentare K facilita dunque il superamento.
Nei record `Q6_K`, `massive-it`, in coda, accade il contrario: la mediana di `p1`
attaccata scende da 0.739 a 0.289 e quella del margin `p1-p2` da 0.566 a 0.112;
gli errori sopra soglia scendono da 916/2974 a 30/2974. Anche la mediana di `p1`
sugli errori scende, da 0.746 a 0.286, mentre gli errori totali salgono da 2296
a 2774. La distribuzione cambia più della soglia. L'aumento delle opzioni può
disperdere la probabilità del vincitore, ma questi record non isolano un meccanismo
causale del modello. Audit riproducibile: `tools/e21_clm_distribution.py`, output
`results/provenance/e21_clm_distribution.json`.

| K | soglia p1 | p1 mediana att. | margin mediano att. | errori | errori oltre soglia | errori con margin ≥0.75 |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 0.81250 | 0.739 | 0.566 | 2296 | 916 | 773 |
| 8 | 0.78125 | 0.574 | 0.347 | 2516 | 471 | 307 |
| 16 | 0.76563 | 0.442 | 0.230 | 2638 | 188 | 120 |
| 24 | 0.76042 | 0.367 | 0.164 | 2708 | 105 | 62 |
| 32 | 0.75781 | 0.317 | 0.132 | 2754 | 65 | 42 |
| 40 | 0.75625 | 0.289 | 0.112 | 2774 | 30 | 14 |

Il margin e la confidence descrivono due soglie diverse; l'OER sul margin
conferma il calo delle decisioni sbagliate molto decise. La soglia di severità
fra backend resta non portabile perché le definizioni di confidence differiscono.

Il rapporto OER attaccato/pulito resta comunque **2.0x-11.0x** su tutta la curva, quindi
l'attacco alza davvero il tasso di errore convinto: parte da una base bassa e resta
basso.

## Risultato 4: il gate sul margin funziona, e il suo punto operativo è dichiarato

Curva costo/beneficio a `massive-it`, K=16, `append`. Costo = frazione di risposte
**corrette** pulite bloccate; beneficio = `OER_adv - OER_pulito` alla stessa soglia.

| τ | costo pulito | OER pulito | OER attaccato | beneficio |
|---|---|---|---|---|
| 0.25 | 0.0941 | 0.2599 | 0.4011 | **0.1412** |
| 0.50 | 0.0514 | 0.0669 | 0.1587 | 0.0918 |
| 0.75 | 0.0192 | 0.0071 | 0.0403 | 0.0333 |
| 0.90 | 0.0030 | 0.0000 | 0.0098 | 0.0098 |
| 0.95 | 0.0000 | 0.0000 | 0.0007 | 0.0007 |
| 0.99 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

`benefit_over_clean` è positivo a ogni soglia fino a 0.95 (a 0.99 è zero): il gate cattura
errori indotti che il
modello non stava già sbagliando da solo, che è la definizione operativa di "guardrail
rotto". A τ=0.90 il rapporto beneficio/costo è **3.3x**, il migliore fra le soglie con costo non
nullo (a 0.95 il costo è già zero e il rapporto non è definito).

Il punto operativo dichiarato (budget 1% di falsi blocchi sul pulito, derivato dal solo
run pulito e congelato prima dell'avversario) dà τ=0.9239 a K=16, con 4/464 risposte
buone bloccate. **Attenzione ai due denominatori**: qui 4/464 sono risposte corrette al
pulito, mentre nella curva il costo è su tutte le decisioni pulite (2974), quindi 0.0030 a
τ=0.90 è 9 su 2974 e non 4 su 464. Le due rate non sono confrontabili. I τ per K vanno da 0.9955 (K=4) a 0.8437 (K=40): **scendono con K**,
perché i margin puliti si stringono (`p50` da 0.6044 a 0.2195). Un τ fisso non sarebbe
stato solo arbitrario, sarebbe stato diversamente di giusto a ogni K.

## Risultato 5: la posizione non conta, e l'asimmetria upstream è inerte qui

Su CLM `prepend` e `append` sono quasi uguali, con l'ordine stabile a ogni K, **testa sempre sopra coda**, e un divario che si stringe
con l'ampiezza (0.5679 contro 0.4840 a K=4; 0.6738 contro 0.6277 a K=40). Su `laya-multilingual` la
coda era ~3x la testa. Quindi la previsione 2 è sbagliata.

La ragione è verificabile e non è un mistero: l'asimmetria train/serve dichiarata da
CLM (`"truncation": "head"`) riguarda il **discard di token oltre 2048**, e i nostri
stati sono ~30 token (`delta_tokens` 8.4-9.6 aggiunti). **La troncatura non scatta
mai**, quindi l'asimmetria è reale ma inerte a questa lunghezza. Diventerebbe
rilevante solo con il filler di E04 a spingere l'iniezione oltre il budget: quel test
non è stato fatto e resta aperto.

## Controllo di precisione: `Q5_K_M` contro `Q6_K`, appaiato

Il risultato qui sopra è `@Q6_K`, e una obiezione legittima è che sia un artefatto della
quantizzazione. Il confronto possibile su questa host è `Q5_K_M` (6.235.207.296 B, sha256
`5c59fb55…`) contro `Q6_K` (7.027.340.928 B, sha256 `9801569e…`): il **bf16** non ci sta, il
file da solo è 16.4 GB. Quindi questo non chiude l'obiezione, dice solo che la conclusione non
sta su un filo di rasoio rispetto al livello di quantizzazione.

**Prima cosa: le due quantizzazioni non si distinguono.** Smoke test sulle due ancore
pubblicate: `tides` 0.99341 e `invoice` 0.98197, **identici a cinque cifre** a quelli di
`Q6_K`. Poi E03 a $K=4$ e $K=16$, entrambi i template, entrambi i dataset, $N=2974$:

| confronto | esito |
|---|---|
| label diverse sul pulito, 8 condizioni | **0** |
| $\asr$, drop, denominatore, `mean_conf_wrong_adv`, `delta_tokens` (64 confronti) | **0 differenze** |
| $\tau$ del punto operativo e `n_blocked` | **0 differenze** |
| McNemar sullo stato di ribaltamento | $b=0$, $c=0$, $p=1.0000$ su tutte e 8 le righe |
| probabilità diverse | **4 su ~11.900** a $K=4$ coda, `max |Δp|` 0.024, e **nessuna cambia una decisione** |

Quindi: la quantizzazione sposta quattro probabilità su dodicimila e non sposta nient'altro.
Non è il caso della famiglia laya, dove E15 misurava uno spostamento di $\asr$ di 4.4 punti
fra 4 bit e fp32 su deberta (McNemar $11$ contro $41$, $p<10^{-4}$): lì la quantizzazione
**peggiorava** il modello e qui non lo tocca affatto. La ragione plausibile è la struttura
della decisione — argmax su $\cos$ tra vettori congelati con teste di proiezione fisse, e il
coseno su cui si decide ha un margine largo — ma è un'ipotesi, non una misura.

**Cosa non dimostra.** Non dimostra che la quantizzazione sia innocua in generale, e non
sostituisce il confronto con bf16. Su questo backend la temperatura è 0.01 e la decisione è
un argmax: il regime è diverso da quello di E15, e i due risultati non si generalizzano a
vicenda.

**Nota sul percorso di misura.** Il `RunStore` config del braccio avversario registra
`quantization: None` su tutti e 12 gli store, mentre i 6 puliti registrano
`Q5_K - Medium`: è il buco già dichiarato sopra, e qui si vede con i dati. Le run non si
sono mescolate perché lo slug del checkpoint (`clm-v0.1-8B-Q5_K_M`) entra nel config ed è
diverso: la salvezza è il nome, non il disegno. Il confronto appaiato è in
`tools/precision_paired.py`.

## Cosa NON contiene

- **Nessuna affermazione sulla latenza.** Il 9x di CLM è su H100 e su una 3080 con 8B in Q6 non
  è un confronto. Le run hanno 106-235 ms per call a K=4/16/40 (`input_tokens` non
  nullo, encoder davvero in esecuzione) e niente di più.
- **Nessuna affermazione su bf16.** Noi non possiamo girarlo: 16.4 GB non stanno in 10. Tutto
  è `@Q6_K`, dichiarato in `run_identity`, con baseline pulita **rimisurata** e numeri
  di CLM upstream intoccati.
- **Nessuna conclusione sulla robustezza *in generale*.** Una riga di un protocollo su
  due dataset non dice se 30M di hard negatives aiutano su attacchi diversi.
- **`banking77` non c'è**, per la ragione di sopra.

## Provenienza degli store storici

Il config del braccio avversario storico di E03 non includeva
`engine.run_identity()`, mentre quello pulito sì. Il writer è stato corretto per
le nuove run; gli store precedenti restano storici e l'audit
`results/provenance/E03_historical_incomplete_identity.json` li distingue.

Per CLM è innocuo perché lo slug del checkpoint (`clm-v0.1-8B`) entra nel config e
differisce da tutti gli altri. È dichiarato in ogni riga comunque, nel campo `backend`.

## Cosa si porta via, in una riga

L'attacco che nomina un'opzione nello stato passa su CLM, il cui addestramento comprende
30M di hard negative; il suo ASR è inferiore a quello degli altri modelli, e
con un guardrail che si rompe quattordici o quindici volte meno: **la robustezza ai hard negatives
non è la stessa cosa della robustezza all'iniezione, e qui si vede con un esperimento
che li separa.**

## Note collegate
[[E03-option-injection]] · [[clm]] · [[E17-configuration]] ·  ·
[[metriche-attacco]] · [[E16-calibrazione-difesa]] · [[attacchi-pubblicati]] ·
[[E15-quantizzazione]]
