---
tags: [esperimenti, moc]
---

# Esperimenti

Ogni esperimento = una nota dal [[template-esperimento]]. Stato: `idea` → `in-corso` → `fatto` → `scartato`.

## Indice

### Setup
| id | titolo | stato |
|---|---|---|
| [[E00-setup]] | Ambiente locale, smoke test, verifica troncatura/routing | fatto |
| [[E01-baseline]] | Baseline pulita su typed-decisions (i NOSTRI numeri) | fatto |

### Categoria A — strutturali
| id | ipotesi | leva | stato |
|---|---|---|---|
| [[E02-option-count]] | acc crolla con K opzioni crescente | [[token-budget-head]] | fatto (N=400) |
| [[E03-option-injection]] | nomina un'opzione distratore nello stato (etichette intatte) | [[superficie-attacco]] | fatto (livello 🥇) |
| [[E04-truncation]] | segnale in coda fuori budget → default class | [[truncation-stato]] | fatto (N=400) |
| E05 | keyword mirroring sposta l'argmax | [[option-marker-injection]] | idea |
| E06 | far cadere sul checkpoint sbagliato via script | [[routing-script-bypass]] | idea |
| [[E13-label-semantics]] | le opzioni sono un canale di informazione? (non submission) | [[superficie-attacco]] | fatto |
| [[E03xE04-composizione]] | injection + troncatura: compongono o si annullano? | [[superficie-attacco]] | fatto |
| [[E03xE04-laya-multilingual]] | la composizione su `laya-multilingual`: il filler ripara il guardrail? | [[token-budget-head]] | fatto (no) |
| [[E12b-sinonimi-rari]] | sinonimi rari (WordNet) sulla superficie lessica | [[superficie-attacco]] | fatto (debole) |
| [[E14-self-guardrail]] | il modello come proprio detector (difesa self-guardrail) | [[superficie-attacco]] | fatto (non funziona) |
| [[E12c-simbolo]] | un simbolo invece di una sostituzione sinonimiale | [[token-budget-head]] | fatto (è E04) |
| [[E12d-omoglifi]] | sostituzione character-level con omoglifi Unicode | [[superficie-attacco]] | fatto (ASR alto, OER 0) |
| [[E12e-omoglifi-uts39]] | omoglifi con tabella normativa UTS #39, non scritta a mano | [[superficie-attacco]] | fatto (tesi verificata) |
| [[E12f-omoglifi-livelli]] | A (confusabile) / B (visibile) / C (controllo): la somiglianza predice l'attacco? | [[superficie-attacco]] | fatto (no) |
| [[E12i-underscore]] | `_` fra le parole: attacco in-distribuzione | [[superficie-attacco]] | fatto (inerte) |
| [[E12l-zerowidth]] | caratteri a larghezza zero (invisibili) | [[superficie-attacco]] | fatto (ASR 0.70, OER 0) |
| [[E12m-swap-tasti]] | riordino su tasti adiacenti (costa 0 token) | [[superficie-attacco]] | fatto (inerte) |
| [[E22-attenzione-laya]] | quanta attenzione prende la frase iniettata, in testa e in coda? (fuori dal paper) | [[architettura-laya]] | fatto (no: la quantita' non spiega l'ASR) |

### Fuori dalle categorie: il programma di spiegazione
- **[[E23-meccanismo-injection]]** — il meccanismo interno dell'iniezione, cinque bracci
  misurati a N pieno il 2026-10-09: patching sui marker (`adv e23`), knockout delle teste
  (`adv e23b`), canale del formato (`adv e23c`), forma della nominazione (`adv e23d`),
  difesa interna (`adv e23e`). Predizioni registrate prima della misura: P1, P2, P5
  confermate; P3, P4 falsificate.
- Il materiale che lo rende possibile: [[MOC-interpretabilita]] (23 paper scaricati e
  verificati con sha256 in `60-interpretabilita/paper/`).

### Categoria G — meccanismo (interpretabilita')
| id | ipotesi | leva | stato |
|---|---|---|---|
| [[E23-meccanismo-injection]] | quale componente interna fa scegliere l'opzione nominata, a quale profondità, in quale campo | [[circuiti-e-interventi]] | fatto (tutti e cinque i bracci, M5 compreso: difesa interna al 91% di detection, 0.8% falsi blocchi) |

### Categoria B — bias noti
| id | ipotesi | leva | stato |
|---|---|---|---|
| E07 | `noul` nativo vs `choice` neutra, gap sistematico | [[noul-label-bias]] | idea |
| E08 | gold al livello 0 di `score` sovrastimato | [[score-position-bias]] | idea |

### Categoria C — semantici
| id | ipotesi | leva | stato |
|---|---|---|---|
| E09 | decomposition in N pezzi → FNR crescente | [[prompt-injection-encoder]] | idea |
| E10 | context manipulation (role/simulazione) | [[prompt-injection-encoder]] | idea |
| E11 | dominio out-of-distribution (zero-shot ~ chance) | [[laya]] | idea |

### Categoria D — confidence
| id | ipotesi | leva | stato |
|---|---|---|---|
| E14 | OER (errori confidenti) baseline e post-attacco | [[calibrazione-fratturata]] | idea |

### Categoria E — gray-box (solo se si scelgono i pesi)
| id | ipotesi | metodo | stato |
|---|---|---|---|
| E15 | perturbazioni gradient-based sul decision head → transfer | FGSM/HotFlip sull'embedding | idea |

### Categoria F — meta-ottimizzazione (prior art adattato)
| id | ipotesi | metodo | stato |
|---|---|---|---|
| ~~E12~~ | GA a bigrammi + fitness a distribuzione | rimosso il 2026-10-01: operatore inerte | negativo, 0 flip |

**Il ramo del GA a bigrammi è stato rimosso.** L'esperimento era negativo con un risultato già
scritto (zero flip in 30 casi su 600 generazioni), la libreria che ne restava è quella che
[[E12b-sinonimi-rari]] riusa, e tenere un esperimento inerte costa una riga in tre note. Non è
una perdita: il suo posto nella storia è la frase "la sostituzione di bigrammi non sposta
niente", che [[E12b-sinonimi-rari]] ripete con il numero accanto.

Le categorie B-D usano E07-E11 ed E14, la gray-box E15, e gli id bassi restano liberi. E13 è una
misura diagnostica già eseguita, non un esperimento di attacco.

## Regole di corsa
1. Un esperimento → una nota → un output in `results/`
2. Ogni nota cita: checkpoint+hash, dataset+split, N, comando
3. Numeri **misurati**, mai citati dal model card
4. Prima baseline, poi attacco: senza [[E01-baseline]] niente drop

## Template
→ [[template-esperimento]]

## Inbox
- [ ]assegnare id E02–E11 a chi partecipa
- [ ] condividere `results/` in un'unica cartella locale con layout fisso

- **[[E17-configuration]]** — nessuna configurazione di esecuzione chiude il buco nel
  guardrail: 6 knob, ASR 0.9723-0.9732, `OER@0.75` 0.896-0.898 su tutto. Escursione 0.09pp.
- **[[E18-nomination-detector]]** — detector sulla nominazione esplicita dell'opzione:
  chiude il residuo che la calibrazione non chiude, ASR 1.0000 → 0.0000 al costo di 0.5% di
  traffico legittimo (0.47% in inglese). Meccanico, non un modello.
- **[[E19-detector-labels]]** — l'etichetta del detector conta più dell'architettura:
  stesse condizioni di E14, l'etichetta `has_prompt_injection` (forma NotInject) taglia i falsi
  positivi da 0.227 a 0.020. Il detector meccanico (E18) batte il classificatore binario su
  entrambi gli assi.
