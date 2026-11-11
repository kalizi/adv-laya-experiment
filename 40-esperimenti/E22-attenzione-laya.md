---
tags: [esperimento, attenzione, laya, meccanismo]
stato: fatto
---

# E22 — Dove guarda Laya quando la frase iniettata è in coda

**Fuori dal percorso critico e dal paper.** Serve ai post della serie e al blog, e la
domanda è «perché succede su Laya». Vale **solo per Laya**: DeBERTa (open-Jev) ha
attenzione disentangled e CLM è un Qwen3-8B su llama.cpp, quindi le mappe non sono
confrontabili fra modelli e non sostengono i confronti cross-architettura di 113 e 115.
Correlazione fra attenzione e ASR, **non** prova causale: un modello, un dataset.

## La previsione, registrata prima di misurare

L'iniezione di E03 funziona molto meglio in coda che in testa. Sullo stesso checkpoint,
stesso dataset, stessa frase (disegno 2×2 di [[E03-option-injection]], K=16, N pieno):
`Risposta corretta: {opzione}.` dà **ASR 0.8208 in testa e 0.9686 in coda**. Su
`laya-multilingual` il rapporto è ancora più largo (coda ~3x la testa).

Il meccanismo candidato era l'attenzione. Letto dal modello caricato
(`results/provenance/e22_attention_structure.json`): 28 livelli, **10 globali e 18
locali** (0.357, non 1/3 come diceva la nota), `local_attention=128` quindi
**raggiungibilità ±64 token**, misurata sui pesi (globale: tutta la sequenza).

Due previsioni, scritte prima di guardare una sola mappa:

- **H1**: la frase in coda riceve **più** attenzione di quella in testa, e la differenza
  è portata dai **livelli globali**, perché nei locali le due posizioni sono
  geometricamente simmetriche.
- **H2**: l'attenzione **dai marker** è maggiore per la frase in **testa**, perché nei
  livelli locali i marker sono a ~45 token dalla frase in testa e ~58 da quella in coda.

Criteri fissati prima: *falsificazione di H1* = `all` in testa ≥ `all` in coda sui
globali; *falsificazione di H2* = `markers` in coda ≥ `markers` in testa. Piuà un
controllo: **frase neutra** (stessa cornice, nessuna opzione nominata) alle stesse due
posizioni, e **caso pulito** con span abbinati. Se attacco e neutro prendono la stessa
attenzione, il modello non distingue la nominazione.

**Il verdetto è che H1 ha ragione nella direzione e torto nel meccanismo, e H2 è
falsificata.** E il controllo dice di più di entrambe.

## Setup

- **Checkpoint**: `convaiinnovations/laya-typed-decisions` (target congelato), CUDA/bf16,
  `laya 0.3.11`, `torch 2.14.0+cu130`, pesi con SHA-256 nel payload.
- **Dataset**: `massive-it` test, **K=16**, `seed=0`, **N=2974** (1211 decisioni corrette
  al pulito). Nessun `--limit`: la trappola di [[E17-configuration]].
- **Frase**: una sola, `Risposta corretta: {opzione}.`, con l'opzione distrattore scelta
  come in E03. Stessa frase in testa e in coda: la posizione è l'unica cosa che cambia.
- **Braccia**: `clean`, `neutral`, `attack`, ognuno con lo span misurato in testa e in
  coda (il pulito le misura entrambe nello stesso forward).
- **Codice**: `src/layaadv/experiments/e22_attention.py` (`uv run adv e22`), estrazione in
  `src/layaadv/model/engine.py`, figure in `tools/e22_attention_figure.py`.

## Risultato 0: l'attacco riproduce E03 alla quarta cifra

| braccio | accuracy | drop | **ASR** | denominatore |
|---|---|---|---|---|
| clean | 0.4072 | — | — | — |
| neutro in testa | 0.4674 | −6.02pp | 0.1561 | 1211 |
| neutro in coda | 0.3921 | +1.51pp | 0.2089 | 1211 |
| **attacco in testa** | 0.0794 | +32.78pp | **0.8208** | 1211 |
| **attacco in coda** | 0.0141 | +39.31pp | **0.9686** | 1211 |

Gli ASR dell'attacco sono **gli stessi numeri del disegno 2×2 congelato di E03** (0.8208
e 0.9686): stessa frase, stesso bersaglio, stessa K, stesso seme. Non è una
coincidenza, è la verifica che questo run sta misurando lo stesso attacco.

Il **controllo neutro non è nullo**, e va detto: una frase che non nomina nessuna
opzione ribalta comunque il 15.6% (testa) e il 20.9% (coda) delle decisioni corrette, e
in testa fa addirittura *salire* l'accuracy di 6pp. Quindi su questo checkpoint
**qualsiasi frase in più nello stato sposta le decisioni**, e l'ASR dell'attacco va letto
contro il neutro, non contro zero.

## Risultato 1: la coda prende più attenzione, ma non per colpa dei livelli globali

Massa d'attenzione sullo span, media per livello e per tipo (N=2974):

| profilo | globali | locali | rapporto coda/testa |
|---|---|---|---|
| attacco in testa | 0.0893 | 0.0952 | — |
| **attacco in coda** | **0.1347** | **0.1556** | **1.51x / 1.64x** |
| neutro in testa | 0.0992 | 0.1117 | — |
| neutro in coda | 0.1387 | 0.1670 | 1.40x / 1.50x |
| pulito in testa | 0.0512 | 0.0443 | — |
| pulito in coda | 0.0492 | 0.0431 | 0.96x / 0.97x |

Confronto appaiato per caso (differenza coda − testa, sui 2974 casi, 28 livelli):

| quantità | media | globali | locali | casi in cui vince la coda |
|---|---|---|---|---|
| da tutte le query | +0.0551 ± 0.0825 | **+0.0454** | **+0.0605** | 59021/83268 (70.9%) |
| dai marker | +0.0354 ± 0.0727 | +0.0406 | +0.0326 | 47622/83266 (57.2%) |

- **H1, direzione**: confermata. La frase in coda prende ~1.5x l'attenzione di quella in
  testa, su 28 livelli e 2974 casi, con il 71% delle coppie (caso, livello) a favore.
- **H1, meccanismo**: **falsificato**. La differenza non è portata dai livelli globali:
  sui locali è *più* grande (+0.0605 contro +0.0454). Non c'è un eccesso a lungo raggio
  da invocare.
- **H2**: **falsificata**, e nel verso opposto. I marker danno più attenzione alla frase
  in coda (+0.0354), non a quella in testa.

## Risultato 2: il controllo è il risultato

| confronto (a parita' di lunghezza dello span) | casi | media | globali | locali |
|---|---|---|---|---|
| attacco vs **neutro**, in testa | 866 | −0.0101 | −0.0071 | −0.0118 |
| attacco vs **neutro**, in coda | 866 | −0.0095 | −0.0045 | −0.0123 |
| attacco vs **pulito**, in testa | 1472 | +0.0448 | +0.0319 | +0.0519 |
| attacco vs **pulito**, in coda | 1472 | +0.0899 | +0.0737 | +0.0988 |

La frase che **funziona** prende **meno** attenzione della frase neutra che funziona 4-5
volte meno (0.8208-0.9686 contro 0.1561-0.2089), a entrambe le posizioni. E la differenza
contro il pulito dice che una frase qualunque in quelle posizioni prende molta più
attenzione del testo che c'era: **la presenza di una sentence attira attenzione, il suo
contenuto no**.

Quindi: su questo checkpoint la quantità di attenzione che la frase riceve **non
spiega** l'ASR. La previsione meccanicistica era sbagliata nel modo più utile, cioè
con un controllo che la rovescia.

## Estensione: secondo checkpoint, `laya-multilingual` (2026-10-08)

Stesso protocollo, `adv e22 --checkpoint convaiinnovations/laya-multilingual`, N=2974,
1211→1651 decisioni corrette al pulito. Il modello è **diverso**: **22 livelli, 8 globali e
14 locali** (0.364) invece di 28 e 10/18, e hidden 768 invece di 1024. Quindi il layout si
legge dal modello anche qui, e non è lo stesso.

| braccio | accuracy | ASR | denominatore | glob | loc |
|---|---|---|---|---|---|
| clean | 0.5551 | — | — | 0.0592 | 0.0391 |
| neutro in testa | 0.4990 | 0.2090 | 1651 | 0.0664 | 0.0464 |
| neutro in coda | 0.1833 | **0.7426** | 1651 | 0.0940 | 0.0785 |
| attacco in testa | 0.1422 | 0.7620 | 1651 | 0.0709 | 0.0447 |
| attacco in coda | 0.0061 | **0.9897** | 1651 | 0.0888 | 0.0543 |

**La conclusione principale replica**: la frase che attacca prende *meno* attenzione di
quella neutra in entrambe le posizioni (−0.0021 in testa, −0.0184 in coda, a parità di
lunghezza) e produce un ASR molto più alto. In testa è ancora più netto che sul target:
l'attacco prende la **stessa** attenzione di niente (−0.0008 contro lo span posizionale) e
ribalta il 76% delle decisioni.

**La sotto-claim sui livelli non replica, e va ritirata.** Sul target la differenza
testa/coda era portata dai livelli locali (+0.0605 contro +0.0454); qui la portano i
**globali** (+0.0179 contro +0.0096). Quindi «quale tipo di livello porta la differenza»
non è una proprietà del meccanismo ma del checkpoint, e la previsione H1 va dichiarata
morta in quella forma.

**E qui emerge una differenza fra i due modelli che nessuna storia di attenzione
spiega**: su `laya-multilingual` la frase neutra **in coda ribalta da sola il 74.26%**
(1226/1651), contro il 20.89% del target. Cioè su questo checkpoint la posizione fa quasi
tutto il lavoro e nominare l'opzione aggiunge 25 punti, mentre sul target nominare
l'opzione aggiunge 78 punti. Due modelli, due meccanismi diversi a parità di protocollo, e
la quantità di attenzione non distingue nessuno dei due.

## Perché la domanda globale/locale non è decidibile qui

Su `massive-it` a K=16 lo stato è **corto**: ~24 token, sequenza ~126, testa ~101, marker
fra 16 e 96. La distanza media dai marker allo span è **45.8 token in testa e 57.9 in
coda**: **entrambi dentro la finestra ±64**. L'asimmetria globale/locale non ha modo di
esprimersi, ed è il motivo per cui i due tipi di livello danno differenze simili.

Per decidere la domanda serve uno **stato lungo**, dove la frase in coda cada oltre i 64
token e i livelli locali non possano vederla: è un esperimento diverso, dichiarato come
non fatto. Il progetto diceva "finestra 128" e la raggiungibilità reale è ±64: con 128
si sarebbe concluso che entrambe le posizioni sono sempre raggiungibili, e con 64 che
non lo sono quasi mai.

## Metodo, e le sue trappole

L'estrazione sta in `layaadv/model/engine.py`, l'unico modulo che tocca `laya`/torch e le
API private. Il loop resta quello di `evaluate.py`: `collect(part, preds)` viene chiamato
per chunk e le sue righe finiscono nello store con le predizioni.

- L'encoder va in **eager** per la durata della chiamata (SDPA non restituisce i pesi:
  avvisa e ridà `None`) e viene ripristinato. Un hook per modulo cattura i 28 tensori.
- **Un caso per forward**: in bf16 la composizione del batch sposta le previsioni, e il
  controllo di coerenza con il run stock deve essere esatto.
- Lo span è localizzato in tre modi, e la riga dice quale ha vinto (`located_by`):
  token id della frase (funziona in testa), **diff** con lo stato pulito (funziona in
  coda), posizione (il controllo pulito). La tokenizzazione non è compositiva: una frase
  in coda **non** si trova con i suoi propri token id, e senza il diff lo span in coda
  sarebbe silenziosamente un altro.
- Lo span del controllo pulito è abbinato **per lunghezza** allo span reale della frase
  (`engine.attention_span_tokens`). Contare la frase tokenizzata da sola dà un controllo
  più corto, e su 1502 casi su 2974 lo stato è più breve dello span richiesto e viene
  clampato: quei casi sono esclusi dai confronti contro il pulito
  (`n_cases_comparabili_con_il_pulito`), e il numero è pubblicato.
- Il confronto è **appaiato per caso** e a parita' di lunghezza dello span. La frase
  neutra è più lunga di un token circa e su 2108 casi i due span non coincidono.

## Quanto rumore c'è nella tubatura

Il forward eager non è bit-identico a SDPA. Isolato con
`tools/e22_attention_numerics.py` su 300 casi:

- la **tubatura** con SDPA riproduce il run stock: **0/300** decisioni diverse;
- **eager** contro stock: **9/300** (3.0%). Tutta la differenza residua è
  dell'implementazione dell'attenzione, non del codice.

Sulla run piena (N=2974): 58/2974 (2.0%) sul pulito, 57-63/2974 (1.9-2.1%) sul
neutro, 10-11/2974 (0.3-0.4%) sull'attacco. Il profilo di quei casi appartiene a una decisione
diversa da quella pubblicata, ed è il motivo per cui il numero è in ogni riga.

## Limiti dichiarati

- **Un modello, un dataset, una K, un seme.** Correlazione, non causalità. Le sezioni
  «secondo checkpoint» e «distanza» allargano a due modelli e a due geometrie, ma restano
  su un dataset e una K.
- Il rapporto «coda ~3x testa» è di `laya-multilingual` e **non** è rimisurato qui.
- La geometria vicina ha lo stato corto (24 token) ed entrambi gli span dentro la finestra:
  la domanda globale/locale si decide con la geometria **lontana** (E22b), non con quella
  vicina.
- La frase neutra non è un controllo nullo: ribalta da sola il 15-21% delle decisioni.
- Il controllo pulito è abbinato solo dove lo stato è più lungo dello span (1472 casi su
  2974 per l'attacco, 452 per il neutro).
- Le mappe non sono confrontabili con DeBERTa o CLM: architetture di attenzione diverse.

## Estensione: la distanza, per decidere globale/locale (E22b, 2026-10-08)

A K=16 su `massive-it` lo stato è corto (~24 token) ed entrambi gli span stanno **dentro**
la finestra ±64: la distinzione globale/locale non poteva esprimersi. Cambiare K non serve
(lo stato comincia sempre subito dopo la testa, quindi la distanza dai marker allo span in
coda è la lunghezza dello **stato**, non quella della testa). L'unica leva è allungare lo
stato: 200 token di filler neutro (quello di E04) messi **davanti** alla frase, così la
frase resta leggibile e non troncata. La frase finisce a **239-254 token** dai marker,
fuori dalla finestra. `adv e22b`, N=2974, 7 bracci.

I bracci **vicini** non sono rimisurati: vengono dagli store di E22 e sono appaiati per id,
quindi la differenza fra distanze è sullo stesso caso.

| braccio | distanza | accuracy | ASR | denominatore |
|---|---|---|---|---|
| clean (vicino) | ~45 | 0.4072 | — | — |
| attacco in testa, vicino | ~46 | 0.0794 | 0.8208 | 1211 |
| attacco in coda, vicino | ~58 | 0.0141 | 0.9686 | 1211 |
| neutro in testa, vicino | ~46 | 0.4674 | 0.1561 | 1211 |
| neutro in coda, vicino | ~58 | 0.3921 | 0.2089 | 1211 |
| **attacco in testa, lontano** | ~239 | 0.0794 | **0.8398** | 1211 |
| **attacco in coda, lontano** | ~254 | 0.0615 | **0.8604** | 1211 |
| neutro in testa, lontano | ~239 | 0.3655 | 0.3699 | 1211 |
| neutro in coda, lontano | ~254 | 0.3144 | 0.4220 | 1211 |
| pulito in testa, lontano | ~239 | 0.3366 | — | — |
| pulito in coda, lontano | ~254 | 0.3305 | — | — |

### H3 confermata, e con la firma che la previsione chiedeva

La massa d'attenzione sullo span, media per tipo di livello:

| braccio | vicino glob | vicino loc | lontano glob | lontano loc |
|---|---|---|---|---|
| attacco in testa | 0.0893 | 0.0952 | 0.0313 | **0.0208** |
| attacco in coda | 0.1347 | 0.1556 | 0.0356 | **0.0198** |
| neutro in testa | 0.0992 | 0.1117 | 0.0362 | **0.0253** |
| neutro in coda | 0.1387 | 0.1670 | 0.0375 | **0.0220** |

E la firma decisiva, **dai marker**: allo span lontano la massa dai livelli **locali** è
**esattamente 0.0000** (era 0.0553 in testa e 0.0878 in coda), mentre quella dai globali
no. La finestra ±64 non è una convenzione, è un muro: il modello non può leggere quello che
la maschera gli nasconde. La quota dei globali nel profilo infatti si rovescia (glob/loc:
0.94 → **1.51** in testa, 0.87 → **1.79** in coda).

Normalizzato sulla massa che va allo stato (per togliere la diluizione: uno span è una
frazione minore di una sequenza più lunga), la quota sullo span passa da **0.60-0.77 a
0.05-0.06**: un crollo di 15-20x, molto più grande di quanto la sola diluizione possa
spiegare.

### H4: l'attacco regge, quindi i globali bastano

Alla distanza lontana l'ASR è **0.8398 in testa e 0.8604 in coda**, entrambi sopra la
soglia 0.50 e non lontani dai valori vicini (0.8208 / 0.9686). Quindi sì: i livelli
globali bastano a far funzionare l'attacco, e i locali non sono necessari.

### Il risultato che non avevo previsto: il vantaggio della coda è uno stato corto

Alla distanza lontana la differenza testa/coda **scompare**, in attenzione
(**+0.0009** contro +0.0551 del vicino) e in ASR (rapporto coda/testa **1.02x** contro
1.18x). In altre parole: «la coda funziona meglio della testa» non è una proprietà della
posizione, è una proprietà della frase che sta **vicina alla fine di uno stato corto**.
Appena lo stato è lungo, le due posizioni diventano equivalenti. È un'affermazione più
forte e più utile di H1, e non era prevista.

### Confounding dichiarato: lo stato lungo è già una perturbazione

Il filler non è neutro per il modello. L'accuracy pulita passa da 0.4072 a 0.3366/0.3305,
e il neutro in coda ribalta il **42.2%** contro il 20.9% del vicino. Quindi all'ASR alla
distanza lontana contribuiscono due cose — la frase leggibile dai globali e uno stato che
da solo è più fragile — e con questo disegno non si separano. La differenza fra distanze
(che è il confronto appaiato) le separa meglio, ed è quella su cui poggia H3.

## Archivio

| file | cosa |
|---|---|
| `results/E22_attention.json` | target: 5 bracci, curve per livello, confronti appaiati |
| `results/E22_attention__laya-multilingual.json` | secondo checkpoint, stesso protocollo |
| `results/E22b_attention_distance__laya-typed-decisions.json` | geometria lontana: 7 bracci + curve vicine importate |
| `results/E22_attention/runs/<run_id>/` | store del target, un record per caso con i profili |
| `results/E22b_attention_distance/runs/<run_id>/` | store della geometria lontana |
| `results/provenance/e22_attention_structure.json` | layout dell'attenzione letto dal modello |
| `results/provenance/e22_attention_numerics.json` | isolamento del rumore eager/SDPA |
| `results/provenance/e22_attenzione_{all,markers}.pdf` | figure del target |
| `results/provenance/e22_attenzione__laya-multilingual_{all,markers}.pdf` | figure del secondo checkpoint |
| `results/provenance/e22b_distanza_{all,markers}.pdf` | figure della distanza (vicina vs lontana) |

Le figure sono ricalcolate dagli store e il tool fallisce se divergono dal payload
(`tools/e22_attention_figure.py`, `--payload` per scegliere lo scope). Comandi:
`uv run adv e22`, `uv run adv e22b`.
