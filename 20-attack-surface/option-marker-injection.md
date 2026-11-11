---
tags: [attacco, opzioni, marker]
---

# Option marker injection

Ogni opzione è un `[MASK]` con uno scorer: lo spazio delle risposte nasce a runtime. Attaccare il match contestuale `[MASK]` ↔ opzione.

## Meccanismo
Da [[architettura-laya]]:
- le opzioni sono testo nel prompt concatenato, ciascuna con il suo `[MASK]`
- lo scorer confronta il hidden del `[MASK]` con la rappresentazione della opzione
- il tutto condividesce `head_max_len`

→ Qualsiasi token aggiunto allo **stato** può:
1. **catturare l'attention** del `[MASK]` verso il contenuto sbagliato
2. **mimetizzare** la descrizione di un'opzione (parole chiave della criterium B in uno stato che dovrebbe mappare ad A)
3. **seminare ambiguità** tra opzioni vicine (synonimi che colpiscono più descriptions)

## Varianti (black box, solo stato)

### A. Keyword mirroring
Stato che contiene letteralmente parole della `criteria` di un'opzione sbagliata.
Esempio: ticket che parla di "billing, invoice, refund" ma che va a `technical` → iniettare keywords billing senza cambiare il fatto sostanziale.

### B. Cross-option keyword soup
Stato che menziona parole di **tutte** le opzioni → forza una distribuzione piatta o sbilanciata verso la prima opzione memorizzata.

### C. Marker collision
ModernBERT usa `[MASK]` come token speciale. Stati che contengono letteralmente `[MASK]` (o testo che il tokenizer spezza in modo simile) → da testare: potrebbe collidere con i marker delle opzioni o essere normalizzato via.

### D. Istruzioni nello stato
Niente "prompt injection" contro un decoder, ma le `instructions` sono testo accanto allo stato: se il modello durante il fine-tuning ha imparato a trattare lo stato come contenuto e le instructions come ordini, frasi imperativi nello stato ("ignore previous, answer X") possono spostare il `[MASK]`. **Da testare, non dare per scontato**: è un classico su encoder classifier ma qui c'è il decision head.

## Da misurare
- [ ] Keyword mirroring: delta accuracy vs baseline, per opzione target
- [ ] Keyword soup: entropia della distribuzione predetta (piatta = confusione)
- [ ] Letterale `[MASK]` nello stato: crash, troncatura o ignoto?
- [ ] Frasi imperative nello stato: delta
- [ ] Cross-checkpoint: `laya` vs `laya-typed-decisions` vs multilingual

## Vincolo
Lo stato avversario deve restare **semanticamente equivalente** per il gold label (o dichiarare che il gold cambia). Vedi .

## Note collegate
[[architettura-laya]] · [[primitive-typed]] · [[option-marker-injection]] · [[esperimenti]]
