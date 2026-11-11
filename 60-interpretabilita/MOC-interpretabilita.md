---
tags: [interpretabilita, moc, letteratura, meccanismo]
---

# Interpretabilita' — mappa del materiale

Perche' esiste questa cartella: la campagna ([[esperimenti]]) ha misurato **che** l'attacco
funziona ([[E03-option-injection]], ASR 0.83-0.99) e **dove** guarda il modello
([[E22-attenzione-laya]]), ma non **perche'** il meccanismo regga. E22 ha chiuso una strada
con un controllo: la quantita' di attenzione ricevuta dalla frase non spiega l'ASR, e la
frase neutra che funziona 4-5x meno prende *piu'* attenzione. Risposta utile, domanda
aperta. Qui sta la letteratura che serve per riaprirla nel modo giusto.

**La lezione che attraversa tutti i 23 paper**: le mappe di attenzione e le attribuzioni
per token sono il metodo piu' intuitivo e il meno affidabile
([[attenzione-come-spiegazione]]), e ogni metodo va **testato su se stesso** prima di
essere creduto ([[attribuzioni-e-sanity-check]]). Il progetto ha gia' questa disciplina
(`tools/noise_floor.py`, `tools/oer_audit.py`): qui si tratta di applicarla agli
strumenti di spiegazione.

## I sette concetti, in ordine di lettura

1. [[attenzione-come-spiegazione]] — perche' i pesi di attenzione non sono una spiegazione,
   e cosa misurava E22 a sua insaputa.
2. [[bertology-encoder-mascherati]] — cosa si sa degli encoder mascherati (probe, pipeline,
   euristiche superficiali), e cosa trasferisce a ModernBERT.
3. [[attribuzioni-e-sanity-check]] — gradienti, IG, LRP, e i test di randomizzazione che
   distinguono una spiegazione da un rilevatore di bordi.
4. [[probe-e-controfattuali]] — probe con control task e mediazione causale: come dire
   "il modello usa X" senza che sia la sonda a sapere X.
5. [[unita-di-analisi]] — neurone, testa, direzione, feature: quale unita' regge
   un'analisi, e l'audit delle spiegazioni automatiche.
6. [[circuiti-e-interventi]] — circuiti, patching, knockout: il metodo con cui si passa
   da "correlazione" a "causa" dentro il modello.
7. [[perche-gli-attacchi-funzionano]] — trigger universali, feature non-robuste, privilegi
   di istruzione: le spiegazioni candidate del perche' E03 funzioni.

## Letteratura ad un hop

[[letteratura-collaterale]] — dictionary learning, logit/tuned lens, induction head,
causal scrubbing, editing: materiale che non risponde alla nostra domanda ma cambia gli
strumenti con cui possiamo porla.

## I paper, per famiglia

| famiglia | paper |
|---|---|
| attenzione-come-spiegazione | [[attenzione-non-e-spiegazione]] · [[attenzione-non-non-e-spiegazione]] · [[attenzione-e-interpretabile]] · [[flusso-di-attenzione]] |
| teste-e-struttura | [[sedici-teste-meglio-di-una]] · [[teste-specializzate]] |
| bertology-encoder-mascherati | [[cosa-guarda-bert]] · [[bert-riscopre-la-pipeline]] · [[bertology-primer]] |
| attribuzioni-e-sanity-check | [[sanity-check-salienza]] · [[interpretabilita-transformer]] |
| probe-e-controfattuali | [[control-task-probe]] · [[mediazione-causale]] |
| circuiti-e-interventi | [[circuito-ioi-gpt2]] · [[rome-editing]] · [[attribution-patching]] |
| perche-gli-attacchi-funzionano | [[trigger-universali]] · [[esempi-avversari-feature]] · [[gerarchia-di-istruzione]] |
| unita-di-analisi | [[spiegare-i-neuroni]] · [[verso-la-monosemanticita]] · [[audit-spiegazioni-neuroni]] · [[tuned-lens]] |

## Dove si aggancia al resto

- Il meccanismo ipotizzato in [[E03-option-injection]] ("match tra `[MASK]` e token delle
  opzioni") e' un'ipotesi **meccanicistica non testata**: nessun circuito, nessun
  intervento, solo comportamento esterno. [[circuiti-e-interventi]] e' il metodo per
  testarla.
- [[E13-label-semantics]] ha mostrato che il testo delle opzioni pesa 27-29pp: anche
  quello e' un risultato comportamentale. La domanda "dove sta, nel modello, quel peso"
  non ha ancora risposta.
- [[E18-nomination-detector]] e [[E20-paraphrase-evasion]] hanno costruito una difesa sulla
  nominazione. Una difesa fondata sul meccanismo sarebbe piu' forte di una fondata sulla
  regola lessicale, ed e' quello che [[E23-meccanismo-injection]] prova a ottenere.

## Stato

- [x] 23 riferimenti scaricati e verificati con sha256 (`paper/manifest.json`,
      `uv run python tools/fetch_papers.py --verify`).
- [x] Concetti mappati in sette note, piu' la sintesi della letteratura collaterale.
- [x] Paper letti e condensati in una nota per riferimento.
- [x] **E23, braccio `depth`**: misurato a N pieno il 2026-10-09
      ([[E23-meccanismo-injection]]). L'attacco entra negli stati dei marker **dopo il
      livello 16**, satura dal 23, e scrive sul marker dell'opzione **nominata** (41% di
      recupero da solo) invece di danneggiare quello dell'oro (12%).
- [x] **E23, braccio `heads`**: 448 knockout (`adv e23b`). **P3 falsificata**: nessuna
      testa singola muove l'ASR di 5pp (massimo 4.35) e le dieci migliori insieme il
      9.72%, con il 32% delle decisioni pulite cambiate. Le dieci insieme valgono **meno**
      della somma dei singoli (114 contro 190): ridondanza, non sinergia. Le profondità
      18-27 sono inerti.
- [x] **E23, braccio `channel`**: P5 confermata. La nominazione conta nello **stato**
      (+0.7597 sopra il suo controllo neutro) e 8-26x meno negli altri campi. Non previsto:
      la capacità di **mirare** non è confinata (errore diretto 0.7419 nelle criteria di
      un'altra opzione): cambia quanto una frase sposta la decisione, non se il modello sa
      leggerla.
- [x] **E23, braccio `form`**: P4 falsificata in **entrambi** i versi. La stringa non
      serve (ordine invertito, senza spazi, refuso: ASR 0.8877-0.9686) e il riferimento non
      serve (la **negazione** — "la risposta corretta è *non* X" — ASR 0.8233, diretto
      0.9478). Il meccanismo è **associazione lessicale**: riconosce il nome dell'opzione
      in qualsiasi forma e non valuta cosa gli sta attorno. Conseguenza per la difesa: un
      detector lessicale è fragile esattamente dove il modello è robusto.
- [x] **E23, braccio `defense`**: M5, la difesa interna. Un rivelatore **causale e non
      lessicale** (rigira la decisione senza la frase e misura quanto sale un'opzione sola)
      rileva il **91.4%** degli attacchi su `massive-it` con **0.81%** di falsi blocchi, e
      sopravvive alle forme che il detector lessicale di E18 perde (`spaceless`: 0.8786
      contro 0.0000). La `concentration` invece non separa (0.28).

## Note collegate
[[esperimenti]] · [[E22-attenzione-laya]] · [[architettura-laya]] · [[paper-e-riferimenti]]
· [[black-box-vs-gray-box]]
