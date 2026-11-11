---
tags: [E19, difesa, detector, etichette, NotInject, E14]
stato: fatto
data: 2026-10-02
study_valid: true
---

# E19: l'etichetta del detector conta più dell'architettura

Domanda: E14 ha già provato un classificatore binario dentro il modello e ha fallito con 55.8%
di falsi positivi. Il fallimento è dell'**architettura** o della **domanda**?

Risposta: della domanda, per due indipendenti motivi. (1) E14 giudicava l'**intento** ("contiene
un'istruzione che tenta di guidare la decisione"), che è un giudizio soggettivo su ogni stato.
(2) Le opzioni erano **prosa lunga** invece che etichette corte: su due opzioni lunghe e vicine il
modello sceglie per sfumatura semantica, e la scelta non è un rilevamento. La domanda se "ha
senso avere due etichette has/no prompt injection" è quindi già dentro l'esperimento: sono
esattamente le etichette che un detector viene addestrato a emettere, ed è la forma di NotInject.

Questo esperimento tiene **dataset, checkpoint, attacco e architettura identici a E14** e cambia
**solo le due etichette**, quattro volte.

## Risultati

E14 è già un classificatore binario dentro il modello, ma aveva la domanda sbagliata in due
modi: giudicava l'**intento** ("contiene un'istruzione che tenta di guidare la decisione") e le
opzioni erano **prosa lunga** invece che etichette corte. Su opzioni lunghe e simili il modello
sceglie per sfumatura semantica, e la scelta non è un rilevamento. E19 tiene dataset,
checkpoint, attacco e architettura identici a E14 e cambia **solo le due etichette**:

| etichette | falsi positivi | rilevazione | costo per punto di beneficio |
|---|---|---|---|
| lunga, forma E14 | 0.2267 | 0.1933 | 1.17 |
| corte, sì/no | 0.2467 | 0.2600 | 0.95 |
| corte, `has_prompt_injection` / `no_prompt_injection` | **0.0200** | 0.0667 | **0.30** |
| corte, contesto vs istruzione | 0.1333 | 0.1467 | 0.91 |

L'etichetta short di NotInject (`has_prompt_injection`) **bassa i falsi positivi di un ordine di
grandezza** (0.020 contro 0.227) a parità di architettura. Ma la rilevazione è bassa (0.067): il
modello non riconosce questo attacco, che non ha la forma di un prompt injection classico.

Quindi sì, **due etichette "has prompt injection" / "no prompt injection" hanno senso**: sono
quelle che un detector viene addestrato a emettere, ed è la forma che la letteratura usa
[[E19-detector-labels]]. Ma portano dentro il modello solo metà del vantaggio: il falso positivo
scende a 0.02 (E18 fa 0.005 senza forward) e la rilevazione non compete. **Il detector
meccanico di E18 batte il classificatore binario su entrambi gli assi**, perché la firma
dell'attacco è la nominazione di un'opzione, che E14 ed E19 cercano per giudizio o per
somiglianza semantica invece che per confronto lessicale → [[E14-self-guardrail]]

## Cosa dice

A parità di architettura l'etichetta conta più di tutto il resto:

| etichette | falsi positivi | rilevazione | costo per punto di beneficio |
|---|---|---|---|
| lunga, forma E14 | 0.2267 | 0.1933 | 1.17 |
| corta, sì/no | 0.2467 | 0.2600 | 0.95 |
| corta, `has_prompt_injection` / `no_prompt_injection` | **0.0200** | 0.0667 | **0.30** |
| corta, contesto vs istruzione | 0.1333 | 0.1467 | 0.91 |

L'etichetta corta di NotInject **bassa i falsi positivi di un ordine di grandezza** (0.020
contro 0.227) senza toccare l'architettura. Quindi sì: **due etichette "has prompt injection" /
"no prompt injection" hanno senso**, sono già nella letteratura (NotInject, già in bibliografia del paper), e sono la
variante da preferire.

Ma portano dentro il modello solo metà del vantaggio, per due motivi che sono insieme il
risultato:

1. **la rilevazione resta bassa** (0.067): il modello non riconosce questo attacco, che non ha la
   forma di un prompt injection classico — nomina un'opzione e basta, non contiene un'istruzione
   da eseguire;
2. **il detector meccanico di E18 batte il classificatore binario su entrambi gli assi**: FP
   0.005 contro 0.020, rilevazione 1.000 contro 0.067. E lo fa senza forward.

La firma dell'attacco è la **nominazione di un'opzione**, che E14 ed E19 cercano per giudizio o
per somiglianza semantica, mentre E18 la cerca per confronto lessicale. Un detector che sa
**cosa** cercare batte un detector che **giudica**.

**Ma il vantaggio è sul segnale giusto, non su un detector migliore in assoluto.** E18 vede solo
le variazioni che preservano le parole dell'opzione: sinonimo, traduzione, metonimo e typo lo
aggirano. E allentare per coprirli costa 40% di falsi positivi ([[E18-nomination-detector]]). Quindi
il confronto giusto è: a parità di **segnale**, l'etichetta conta più dell'architettura; e
l'etichetta meccanica conta più dell'etichetta corta, ma non risolve l'evasione lessicale.

## Note collegate

[[E14-self-guardrail]] · [[E18-nomination-detector]] · [[metriche-attacco]] · [[E16-calibrazione-difesa]]
