---
tags: [esperimento, genetico, meta-ottimizzazione, lessico, wordnet, negativo]
stato: fatto
data: 2026-09-30
study_valid: true
---

# E12b — sinonimi rari da WordNet, una posizione per generazione

## Perché questo operatore
Il primo operatore lessicale provato, la sostituzione di bigrammi, era risultato inerte: 0 flip su
600 generazioni, e il suo ramo è stato rimosso. La lettura naturale era "il modello è insensibile
alla superficie lessicale". Ma quel test era debole, e uno dei suoi due limiti era mio:

1. il vocabolario era scritto a mano in italiano e non toccava **zero** parole degli stati
   (il dataset è in inglese)
2. non usava WordNet, quindi non poteva andare "fuori distribuzione" in modo misurato

E12b sostituisce l'operatore con uno derivato dagli stati reali e controlla se il finding
regge. **Il finding non regge: la superficie lessica esiste ed è debole, non inerte.**

## Setup
- checkpoint: `convaiinnovations/laya-typed-decisions`, dataset typed-decisions test, N=30
- accesso: L1 black-box, stessa fitness di E12 (V2 = flip + confidenza + KL)
- **selezione delle parole**: spaCy `en_core_web_md`, si tengono NOUN/VERB/ADJ/ADV non
  stopword. Esclusi connettivi (CCONJ/SCONJ), articoli (DET), preposizioni (ADP), ausiliari,
  punteggiatura (PUNCT/SYM), numeri e `PROPN`. I sinonimi con maiuscola iniziale sono esclusi
  come nomi propri (`service` dava "Robert William Service")
- **scelta del sostituto**: WordNet, solo lemmi dello stesso POS e con
  `Lemma.count()` **minore** del termine originale. Si prendono i più rari, ≥3 per posizione.
  `--pick commonest` e `--max-freq-ratio` allargano la scelta, vedi lo sweep
- **protocollo**: 1 posizione nuova per generazione, ≥3 candidati diversi per posizione,
  10 generazioni, la mutazione **si accumula** sullo stato migliore. Early stopping sul
  primo flip; `--patience N` (default 0) per il taglio a plateau
- vincoli: numeri invariati (portano il gold), drift Jaccard ≥ 0.5
- comando: `uv run adv e12b --limit 30`
- output: `results/E12b_rare_synonym.json`, `results/E12b_best.jsonl`

## Risultati

| metrica | valore |
|---|---|
| baseline pulita | 0.7533 |
| casi con almeno un flip | **5/30** (16.7%) |
| ASR a livello decisione | **0.0619** (7 flip su 113 decisioni corrette) |
| generazione mediana del primo flip | 6 |
| posizioni medie attaccate per caso | 9.13 |
| candidati medi per caso | 11.17 |
| confidenza media sul torto | **0.2627** |
| casi senza superficie (0 sinonimi) | **0/30** |

Confronto con la prima versione: 600 generazioni e 0 flip. Ora 291 generazioni e 5 casi.

## Perché il risultato è debole: WordNet non è un dizionario di contesto

Il problema non è l'attacco, è il sostituto. WordNet lavora su **sinneti**, non sul contesto,
e sugli stati tecnici produce parole che sono rare ma sbagliate:

| originale | WordNet propone | problema |
|---|---|---|
| `service` | `Robert William Service` | nome proprio di un poeta |
| `named` | `advert` | non è un sinonimo in nessun uso tecnico |
| `force` | `hale` | verbo arcaico |
| `model` | `manikin` | mannequin, non "modello" |
| `agent` | `federal agent` | sposta il significato tecnico |

Il ciclo accumula anche più mutazioni sullo stesso stato, quindi lo stato finale non è
"un altro modo di dire la stessa cosa": è **un testo diverso**. Eppure i numeri restano
intatti, perché il filtro numerico è il primo controllo e non passa mai.

La conseguenza è che **non sappiamo se i 5 flip siano un attacco o rumore**. La misura lo dice:

| segnale | lettura |
|---|---|
| confidenza media sul torto 0.2627 | **scarsa**: il modello sbaglia a tentoni, non è convinto |
| OER (conf ≥ 0.75) | **0.0** |
| KL shift medio | 0.0097-0.1077, piccoli |
| flip con 2 posizioni e flip con 9 | nessuna relazione con la dose di perturbazione |

Un ASR del 6% con confidenza 0.26 e zero OER non soddisfa : è la firma di
un modello che sbaglia un po' di più su testi degradati, non di una superficie controllabile.

## Il bug che ho trovato a metà strada
Il primo run dava 0/30 anche con l'operatore nuovo, il che era sospetto. La causa: indicizzavo
i token spaCy su `text.split()`. **Su uno stato reale spaCy dà 100 token e `split()` 37**, perché
la punteggiatura JSON e i trattini si attaccano alle parole. La guardia `words[i] != original`
scartava ogni candidato, e l'attacco non partiva mai. Il risultato "0 flip" era garantito a
priori, indipendentemente dal modello.

La correzione è ricostruire il testo percorrendo i token di spaCy
(`tests/test_e12_rare_syn.py::test_substitute_uses_spacy_offsets_not_split`).

**Questo è il punto metodologico**: la prima misura era negativa e sembrava confermare un
finding già scritto nel vault. Non la confermava: era il mio bug. Un risultato che conferma
quello che ti aspetti è il peggiore modo di scoprire un errore, perché non hai motivo di
sospettarlo.

## Sweep: sinonimi meno rari (`--pick`, `--max-freq-ratio`)

L'ipotesi da testare era "i sinonimi *troppo* rari sono anche quelli sbagliati, quindi
rilassare la soglia dovrebbe migliorare l'attacco". Per separarla ho reso entrambe le leve
misurabili: `--pick` (quale parte della pool prendere) e `--max-freq-ratio` (quanto può essere
più comune il sostituto). N=30, baseline 0.7533, 113 decisioni corrette.

| configurazione | casi con flip | ASR | confidenza sul torto | OER | KL medio | candidati/caso |
|---|---|---|---|---|---|---|
| `rarest` (protocollo base) | 3/30 | 0.0354 | 0.222 | 0 | 0.0370 | 11.1 |
| `commonest` | 2/30 | 0.0177 | 0.289 | 0 | 0.0019 | 11.1 |
| `mid` | 1/30 | 0.0088 | 0.508 | 0 | 0.0016 | 11.1 |
| `rarest`, soglia ×1 | 3/30 | 0.0265 | 0.225 | 0 | 0.0759 | 11.5 |
| `rarest`, soglia ×5 | 1/30 | 0.0088 | 0.289 | 0 | 0.0005 | 13.1 |
| `rarest`, soglia aperta | 1/30 | 0.0088 | 0.289 | 0 | 0.0005 | 13.1 |

**L'ipotesi è falsa, e in direzione opposta.** Alzare la soglia non migliora: 3/30 → 1/30.
E il motivo è che la soglia **non è la leva**: la pool di WordNet è già dominata dai sinonimi
rari (misurato: 333 posizioni con soglia stretta contro 393 con soglia aperta, +18%), quindi
rilassarla aggiunge pochissimo, e quello che aggiunge è rumore.

La leva vera è *quale parte della distribuzione* si campiona, e l'ho misurata in `commonest`,
che prende i sinonimi più comuni: sono anche i più fedeli al contesto.

| originale | `rarest` (rumore) | `commonest` (fedele) |
|---|---|---|
| `internal` | intragroup, interior, intimate | national, **inner**, home |
| `exceed` | outgo, outmatch, outperform | **pass**, transcend, top |
| `service` | overhaul, divine service | **help**, avail, military service |
| `model` | fashion model, manakin | **framework**, simulation, example |
| `rotate` | circumvolve, splay | **turn out**, spread out, revolve |

## Cosa questo dice sui flip di E12b

Il confronto `rarest` contro `commonest` è la misura che distingue **superficie d'attacco** da
**distruzione di significato**:

- con sinonimi rumorosi (`rarest`): 3/30, ASR 0.0354, KL 0.037
- con sinonimi che **preservano il significato** (`commonest`): 2/30, ASR 0.0177, KL 0.0019

Restano dei flip anche con sostituti fedeli, quindi non era **solo** distruzione di testo. Ma
il KL crolla di 20× (0.037 → 0.0019): la perturbazione rumorosa sposta la distribuzione molto
di più, e quello che sposta non è un segnale utile. La conclusione è che **la traccia di
superficie autentica è ~0.02 ASR**, non 0.06, e resta sotto i  in ogni
configurazione: OER 0 ovunque, confidenza 0.22-0.51.

Nota sui numeri: il primo `mid` riportava 3/30, poi 1/30 dopo aver corretto lo slicing. Il 3/30
era artefatto di un bug mio nel modo di ruotare la pool, non una misura. Vale la pena dirlo
perché è la seconda volta in questa nota che un numero cambia per un bug e non per il modello.

## Modello spaCy: `sm` → `md`

Passato a `en_core_web_md` come richiesto. Il guadagno è reale ma piccolo, e va detto con la
sua misura:

| | `sm` | `md` |
|---|---|---|
| token con POS di contenuto | 829 | **888** |
| token ancora `X` (non classificati) | 56 | **38** |
| entità `ORG` rilevate | 0 | **5** |

Non cambia la conclusione: il filtro è fatto di POS, e `md` classifica il 7% in più di parole
senza sbloccare una superficie nuova. Il modello non era il collo di bottiglia, i sostituti lo sono.

## Verdetto
- **La superficie lessica è poco inerte, non inerte.** Il primo operatore era inerte e il suo
  ramo è stato rimosso; questo lo corregge nella portata senza metterlo in discussione:
  l'attacco c'è, ed è debole (ASR 0.009 con la take `mid`, OER 0).
- **La soglia di frequenza non è la leva.** Alzarla peggiora (3/30 → 1/30). La pool è già
  dominata dai rari, e i sinonimi rari sono quelli che sbagliano il significato.
- **Il limite è la qualità dei sostituti, non la ricerca**: con 11 candidati e ~9 posizioni
  tentate per caso il ciclo sta usando tutta la superficie disponibile.
- **WordNet da solo non basta** per un attacco lessico: `commonest` è la misura con i sostituti
  più fedeli che si possano ottenere da WordNet, e dà ASR 0.0177 con OER 0. Il modo giusto
  resta [[E03-option-injection]], dove si nomina un'opzione: un'istruzione, non una parola.
- Il codice resta utile come **positivo debole con le sue cave**: 107 test verdi, di cui 19
  sull'operatore, e uno che verifica che WordNet sia caricabile (se manca, l'esperimento
  degenererebbe a 0 candidati e riporterebbe 0 flip come se fosse un finding).

## Cosa resta
- **Glossario tecnico al posto di WordNet**: `commonest` è il tetto di quello che si può
  ottenere da WordNet, e dà 0.0177. Un glossario di dominio (con sinonimi che valgono per
  *quel* testo) è l'unico modo di sapere se la superficie reale è più alta. Il codice non
  cambia, cambia solo `rare_synonyms`.
- Il confronto che manca davvero: E03 con ricerca automatica della parola al posto della
  scelta manuale.

## Note collegate
[[E03-option-injection]] · [[E13-label-semantics]] ·  · [[superficie-attacco]] · [[ga-bigrammi]] · [[esperimenti]]

## Output
- `results/E12b_rare_synonym.json` (per caso: lineage, posizioni usate, storia per generazione)
- `results/E12b_best.jsonl` (stato migliore + lineage per caso)
- `results/e12b_modes/` (lo sweep `--pick` × `--max-freq-ratio`, un JSON per configurazione)

## Riprodurre lo sweep
```bash
uv run adv e12b --limit 30 --pick rarest    --force-new
uv run adv e12b --limit 30 --pick commonest --force-new
uv run adv e12b --limit 30 --pick mid       --force-new
uv run adv e12b --limit 30 --pick rarest --max-freq-ratio 5 --force-new
```