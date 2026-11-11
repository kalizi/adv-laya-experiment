---
tags: [attacco, bug-noto, noul, issue-156]
aliases: [issue #156, noul bias]
---

# `noul` label bias (issue #156)

Bug documentato da Convai stesso: su `noul`, **le label delle opzioni possono dominare sullo stato**.

## Sintomo
Su `laya` (English, il più colpito), `noul` restituisce un **"no" sicuro per input chiaramente positivi**.

## Causa
Il rendering interno delle opzioni di `noul` è **fisso**:
```python
["false: " + (crit.get("false") or "no, the statement does not hold"),
 "true:  " + (crit.get("true")  or "yes, the statement holds")]
```
L'ordine è sempre `[false, true]`, e la coppia di label può catturare l'attention del decision head più del contenuto dello stato.

## Workaround ufficiale (usa questo come baseline)
Riscrivere `noul` come `choice` a 2 opzioni con **chiavi neutre** e la wording desiderata nelle descrizioni:
```python
{"type": "choice",
 "instructions": "Is this review positive?",
 "criteria": {"A": "yes, the review is positive",
              "B": "no, the review is negative"}}
```

## Per la study
1. **Riprodurlo** su un proprio dataset → conta poco da solo (bug noto, vedi )
2. **Estenderlo**: quantificare su quanti esempi del [[typed-decisions-dataset]] il `noul` nativo sbaglia dove la variante `choice` neutra ci azzecca → diventa un attacco *systematico*
3. **Invertirlo**: se le label dominano, campiare `instructions`/`criteria` (ammesso? i criteria fanno parte del deployment… → discutere in ) o, meglio, trovare stati che sfruttano il bias
4. Cross-checkpoint: il bias è su `laya`, vale anche su `laya-typed-decisions` e `laya-multilingual`?

## Da misurare
- [ ] `noul` nativo vs `choice` neutra, stesso contenuto, su N esempi
- [ ] Matrice di confusione: quante volte `noul` dice false a gold=true
- [ ] Confidence sulle predizioni sbagliate (overconf?)
- [ ] Stesso test sui 3 checkpoint

## Note collegate
[[primitive-typed]] · [[noul-label-bias]] · [[esperimenti]] · [[metriche-attacco]]
