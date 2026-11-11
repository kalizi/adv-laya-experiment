---
tags: [sintesi, superficie, character-level, token, negativo, metodo]
stato: fatto
data: 2026-09-30
study_valid: true
---

# E12 sintesi: cosa muove questo modello e cosa no

Questa nota raccoglie nove esperimenti (da [[E12b-sinonimi-rari]] a
[[E12n-multicarattere-bidi]]) e li riduce a una legge sola. Tutte le misure sono su
`laya-typed-decisions`, N=30, baseline 0.7533 su 150 decisioni, 113 corrette al pulito.

> **Due avvertenze sui numeri di questa tabella**, da prima che diventino un artefatto
> storico:
> 1. **`batch_size` diverso fra le famiglie di tool.** E12c, E12b ed E14 girano a 16,
>    E12d/e/f/g/i/l/m/n giravano a 8. In bf16 la composizione del batch sposta una
>    previsione (misurato in questo progetto), quindi un Δtoken di 49 con ASR 0.168
>    (E12c, bs 16) e uno di 49 con ASR 0.62 (E12e, bs 8) **non sono lo stesso esperimento**.
>    Le righe sono confrontabili **dentro** la stessa famiglia di tool, non fra famiglie.
>    Dal 2026-10-01 tutti i tool sono a 16, quindi i numeri futuri non avranno più il
>    problema: quelli qui sotto sì, e non sono stati ricalcolati (servirebbe un giro GPU).
> 2. **N=30, 113 decisioni corrette.** Sotto le ~100 decisioni un ASR è un aneddoto
>    ([[metriche-attacco]]: "N dichiarato"). Qui si dichiara, e la differenza fra 0.65 e
>    0.70 non è un segnale.

## La legge

> **Correzione del 2026-10-01.** Questa sezione diceva "l'ASR è funzione dei token che
> l'attacco aggiunge **dentro** le parole", ed è falsificata dalla tabella qui sotto, che
> sta nello stesso file: la riga E12d ha **Δtoken = −25** (negativo) con **ASR 0.6726**,
> il secondo valore più alto della campagna. Sostituire lettere con spazi dentro le parole
> non aggiunge token, e funziona meglio di quasi tutto il resto. Anche la versione
> "due condizioni" data più avanti non regge, perché richiede anch'essa che i token siano
> *aggiunti*. La formula che i dati supportano è quella di `CLAUDE.md`:
> **corruzione in place del contenuto**.

**L'ASR è funzione della corruzione in place del contenuto.** Non del carattere usato, non
del suo ordine, non della sua somiglianza con l'originale, e **non del costo in token**:
la riga più efficace in assoluto costa meno token del testo che sta sostituendo.

| attacco | Δtoken | ASR | confidenza sul torto | OER | label distinte |
|---|---|---|---|---|---|
| E12n multicarattere (digrammi 2→1) | +2 | **0.0000** | — | 0 | 12 → 12 |
| E12b sinonimi rari (`commonest`) | ~0 | 0.0177 | 0.289 | 0 | — |
| E12c spazi in coda ×30 | +49 | 0.0177 | 0.182 | 0 | — |
| E12m swap di tasti adiacenti | +22 | 0.0177 | 0.410 | 0 | 12 → 12 |
| E12i `_` fra le parole | +36 | 0.0885 | 0.198 | 0 | 12 → 12 |
| E12c simbolo ×30 | +49 | 0.1681 | 0.221 | 0 | — |
| E12n bidi override RLO 100% | +80 | 0.0885 | 0.208 | 0 | 12 → 12 |
| E12l zero-width U+200B 50% | +232 | 0.6549 | 0.203 | 0 | 12 → 7 |
| E12e omoglifi UTS #39 100% | +286 | 0.6991 | 0.211 | 0 | 12 → 7 |
| E12l zero-width U+200D 50% | +359 | **0.6991** | 0.186 | 0 | 12 → 7 |
| E12d spazi distribuiti ~374 | −25 | 0.6726 | 0.208 | 0 | 12 → 6 |
| E04 filler in coda (riferimento) | +212 | 0.553 | 0.193 | ≤0.009 | — |

La tabella è **quasi** monotona, e le due righe fuori posto dicono una cosa precisa.
`E12c simbolo ×30` aggiunge 49 token e ottiene 0.1681, mentre `E12c spazi in coda ×30` ne
aggiunge gli stessi 49 e ottiene 0.0177: **stesso costo, dieci volte di differenza**. La ragione
è che i 30 caratteri sono in coda allo stato, mentre il Δtoken conta solo quando la corruzione
cade **dentro** una parola. Appendere a uno stato JSON non tocca il contenuto.

Quindi la legge ha due condizioni, non una: i token devono essere aggiunti **e** devono cadere
dentro le parole. La sezione sulla posizione sotto la quantifica.

> La seconda condizione è quella buona e regge da sola. La prima è quella che i dati
> smentiscono (E12d: Δtoken negativo, ASR 0.67).

## Cosa non conta, misurato

**Il carattere.** A parità di posizione e di costo, `_` dà 0.0885 e `φ` per `a` dà 0.4336: la
sostituzione non ha un carattere "più offensivo" di un altro ([[E12i-underscore]]).

**La somiglianza con l'originale.** I tre livelli di confusabilità visiva sono indistinguibili:
normativo (A) 0.6991, plausibile ma non normativo (B) 0.6814, controllo senza analogia latina
(C) 0.6372. Sono 79, 77 e 72 flip su 113: differenza di due flip ([[E12f-omoglifi-livelli]]).

**L'ordine dei caratteri.** Scambiare due lettere adiacenti su tastiera costa zero token e dà
0.0177. Il motivo è che i due caratteri adiacenti stanno già nello stesso token BPE, quindi lo
swap non cambia la tokenizzazione ([[E12m-swap-tasti]]).

**L'arità della sostituzione.** Un multicarattere 2→1 è una contrazione: toglie token invece di
aggiungerne, e dà 0.0000 ([[E12n-multicarattere-bidi]]).

**Il rendering.** Il bidi override (U+202E) dà 0.0885 con drop negativo. L'inversione sta nel
rendering, e un encoder non ha rendering: il codepoint letto è quello che c'è
([[E12n-multicarattere-bidi]]).

## Cosa conta oltre al volume

**La posizione, sempre.** Stesso carattere, due posizioni:
| | fra le parole o in coda | dentro le parole |
|---|---|---|
| spazi, 30 caratteri | 0.0177 (2/113) | **0.6726** (76/113) |
| zero-width U+200B | 0.0531 | **0.6549** |

Trentotto volte e dodici volte. La corruzione deve cadere **dentro** la parola, non accanto a
essa: appendere caratteri a uno stato JSON non tocca il contenuto, e il modello non se ne accorge.

**Il costo in byte, per la furtività.** L'efficienza per token è ~0.005 con caratteri a 2 byte e
~0.002 con caratteri a 3 byte, indipendentemente dalla somiglianza. Il set invisibile di UTS #39
è soprattutto cirillico a 3 byte, quindi pagare l'invisibilità costa circa il doppio. Limitando la
tabella alle 5 lettere greche a 2 byte (`c o p u v`) il costo scende da 286 a 100 token e
l'efficienza raddoppia, ma l'ASR assoluto cala a 0.5310 ([[E12e-omoglifi-uts39]]).

**La copertura, solo se le lettere sono frequenti.** Aggiungere `m`, `n`, `k` con sostituti
dichiarati porta 19 → 22 lettere e l'ASR da 0.5841 a 0.6726. Ma `m` e `n` sono frequenti nel
testo, `k` no. Confronto: 9 lettere eguagliavano 19 lettere in E12f, perché le 9 includevano
`a c i o s`. La regola è la frequenza nel corpus, non il numero ([[E12n-multicarattere-bidi]]).

**Un limite non aggirabile.** `f`, `g`, `k`, `m` non hanno confusibile in nessuno script, e
UTS #39 non contiene nessuna sequenza multi-carattere verso una lettera latina (zero righe).
Per queste quattro lettere l'attacco character-level non esiste: non è un problema di metodo, il
carattere non c'è.

## Quello che nessuno di questi attacchi fa

**OER 0 in tutti e dieci gli esperimenti.** La confidenza sul torto resta fra 0.13 e 0.51, e in
nessun caso raggiunge la soglia 0.75 dei .

Il dato strutturale è un altro: **il collasso del vocabolario**. Sotto attacco le 12 label
diventano 7 e sei spariscono del tutto (`3`, `continue`, `false`, `harmful`, `observe`,
`success`). Non è un modello che sceglie male, è un modello che **perde la capacità di esprimere
metà delle risposte** e ricade su un sottoinsieme. L'accuracy va da 0.753 a 0.27-0.29, contro
0.083 di caso per 12 classi.

Per questo E12g dà il verdetto: 2 delle 3 soglie del livello "Bucato" soddisfatte, la terza no.
Il drop è enorme (48.0pp) e l'ASR è 1.0000, cioè 79 flip su 79 decisioni corrette. Ma la
confidenza è 0.2112. L'attacco azzera il modello **in modo passivo**.

Su quindici righe di tabella, l'unica che ha OER 0.85 è E03, che scrive dentro la zona `[MASK]`
della sequenza, dove il modello decide.

## Il modello, in una frase

Non è sensibile a **cosa** le parole dicano. È sensibile a due sole cose: se il segnale che
determina la risposta è presente nel testo, e se lo stato contiene un'istruzione esplicita. Una
mutazione lessicale non tocca nessuna delle due, e un rumore che aggiunge token non ne introduce
nessuna.

## Due bug che avrebbero prodotto falsi risultati

Li metto perché il pattern si è ripetuto due volte e la regola che ne deriva è operativa.

1. **Indici spaCy contro `text.split()`** (E12b). Sullo stato reale spaCy dà 100 token e
   `split()` 37, per la punteggiatura JSON e i trattini. Indicizzando per posizione ogni
   sostituzione falliva in silenzio, e il risultato era "0 flip", che coincideva con il
   risultato atteso. Il primo run con la tabella normativa ha dato 5/30; quello sbagliato dava 0/30.
2. **Costanti Unicode scritte come letterali** (E12l). `"ZWSP U+200B"` conteneva uno spazio
   ASCII: il letterale a larghezza zero si è perso in scrittura. Il dato sembrava dimostrare che
   il tokenizer di laya elimini lo zero-width space. Non era mai stato testato.

**Regola:** ogni costante Unicode va scritta come `"\u200b"`, e ogni risultato nullo va
verificato con un test di identità, cioè rimuovendo la perturbazione il testo deve tornare
identico byte per byte. Entrambi i bug producevano uno zero che sembrava confermare
un'ipotesi già scritta, e in entrambi i casi non avevo motivo di sospettarli.

## Verdetto e spazio rimasto

La superficie character-level è **esaurita** per questa architettura: dieci esperimenti, una
legge che li spiega tutti, e un limite dimostrato (quattro lettere non attaccabili). Proseguire
su questa linea significa cambiare checkpoint o architettura, non cambiare operatore.

Quello che resta, in ordine di valore:
1. **Estendere E03**, che è l'unico canale con errori convinti: trasferimento cross-checkpoint,
   o ricerca automatica della parola al posto della scelta manuale.
2. **Glossario di dominio** per E12b, se si vuole sapere se la superficie lessica autentica è
   più alta di 0.02.
3. **Calibrazione**: tutte le confidenze di questa nota hanno l'asterisco delle temperature non
   valide del checkpoint. Senza rifittarle, l'OER è confrontabile solo fra noi.

## Note collegate
[[E12b-sinonimi-rari]] · [[E12c-simbolo]] · [[E12d-omoglifi]] · [[E12e-omoglifi-uts39]] · [[E12f-omoglifi-livelli]] · [[E12i-underscore]] · [[E12l-zerowidth]] · [[E12m-swap-tasti]] · [[E12n-multicarattere-bidi]] · [[E03-option-injection]] · [[E04-truncation]] ·  · [[attacchi-pubblicati]] · [[superficie-attacco]] · [[calibrazione-fratturata]] · [[esperimenti]]

## Output
Le singole note hanno i dettagli e i comandi. Gli script sono in `tools/`:
`e12e_confusables.py`, `e12e_homoglyph_uts.py`, `e12f_homoglyph_tiers.py`,
`e12f_greek_only.py`, `e12g_scorecard.py`, `e12i_underscore.py`, `e12l_zerowidth.py`,
`e12m_keyboard_swap.py`, `e12n_multichar_bidi.py`.
