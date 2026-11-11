---
tags: [attacco, routing, checkpoint]
---

# Routing / script bypass

Far finire una richiesta sul **checkpoint sbagliato**, sfruttando che la decisione è presa prima della forward pass e che la confidence non segnala l'errore.

## Contesto
Il [[checkpoint-laya|Router]] sceglie in <0.5 ms con script detection + function-word heuristic:
- Latin script → `laya` (English)
- non-Latin → `laya-multilingual`
- URL, email, dotted names **non contano come words**
- la heuristic è best-effort: per lingue Latiniche senza word-list il segnale può mancare del tutto
- hint `lang_guess=` o codice `en`/`eng` → forzano English

## Leve

### 1. Script swap / traslitterazione
Testo non-inglese **traslitterato in Latin** (es. arabo/indiano/hindi in lettere latine) → l'heuristic lo legge come "non English non abbastanza" ma lo script è Latin → cade su `laya` (English) che **collassa fuori dall'inglese**.
- Nota: il router ora tratta spagnolo/francese/italiano/portoghese Latin senza accenti come multilingual (fix recente): verificare il comportamento attuale, non quello documentato nel changelog.

### 2. Contaminazione con URL/email
Un email address o una URL come "parole" non contano per l'heuristic → uno stato breve dominato da URL potrebbe non avere segnale sufficiente → abstain → default.

### 3. Codice lingua forzato
Se il deployment passa `lang=`/`lang_guess=` lato client (o `LAYA_AUTO_TASK`), un codice `en` su testo non-inglese è un bypass diretto. Da verificare come il deployment della study lo configura → [[E00-setup]].

### 4. Farsi mandare il checkpoint sbagliato *dopo* averlo scelto correttamente
Anche con routing corretto: `laya-multilingual` è più debole sull'inglese (0.619 vs 0.684). Se il deployment fissa `model=` a multilingual per via dei dati multilingue, tutti gli stati inglesi partono svantaggiati.

## Impatto atteso
Khmer su `laya`: **0.000 accuracy a 0.952 confidence**. Se riesci a far cadere qualsiasi stato su quel checkpoint, hai massima accuracy 0 e massima overconfidence. È il caso limite; l'interesse è nelle **regioni intermedie** (lingue Latiniche ambigue, testi mistilingue).

## Da misurare
- [ ] Stesso stato in 10 lingue: quale checkpoint risponde? Acc e confidence per coppia
- [ ] Traslitterazione Latin di stati non-Latin: acc delta
- [ ] Stati dominati da URL/email: quale checkpoint?
- [ ] Mappatura: `lang_guess` quirk (`en_US`, `None`, vuoto) → checkpoint effettivo

## Note collegate
[[checkpoint-laya]] · [[calibrazione-fratturata]] · [[metriche-attacco]] · [[esperimenti]]
