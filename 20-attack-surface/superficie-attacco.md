---
tags: [modello, superficie-attacco, moc]
---

# Superficie d'attacco

MOC di tutte le leve disponibili su [[laya]]. Ogni leva → una nota → un esperimento in [[esperimenti]].

> ⚠️ **Prima di tutto:** leggi [[attacchi-pubblicati]]. Injection-in-state su Laya è già uscita (tanh, 22 set): il nostro spazio è altrove.

> Laya **non genera testo**: niente jailbreak classico, niente prompt injection contro un decoder. La superficie è quella di un **encoder + decision head con opzioni runtime**. Per analogie con gli encoder classifier, vedi [[prompt-injection-encoder]].

## 1. Strutturali (budget e troncamento)
| leva | nota | tipo |
|---|---|---|
| **Nominare un'opzione nello stato** | [[E03-option-injection]] | **black box, misurato (ASR 0.86)** |
| Options che condividono `head_max_len` | [[token-budget-head]] | gray/black (osservabile) |
| Stato troncato fuori budget | [[truncation-stato]] | black box |
| Iniezione nello spazio delle opzioni | [[option-marker-injection]] | black box |
| Forzare il checkpoint sbagliato | [[routing-script-bypass]] | black box |

### La leva piu' economica: l'opzione letta come istruzione
Le opzioni di un `choice` non sono solo etichette da classificare: sono **testo che il
modello legge**. Scrivere il nome di un'opzione sbagliata nello stato (le etichette e il
gold restano intatti, quindi è lecito) porta il modello a sceglierla, con l'errore
*diretto* sul target nel 100% dei casi in cui sbaglia.

MASSIVE it-IT, N=2974, curva completa K=4..40: **ASR 0.83-0.89** (soglia 0.50), drop fino
a 54.3pp. Non serve un payload: quattro parole. Funziona anche in coda, e **meglio**:
ASR 0.98 contro 0.90 in testa a K=4 ([[E03xE04-composizione]]).

Il dato che lo rende serio: **rompe il guardrail di confidenza**. Al pulito il modello
sbaglia con confidenza bassa (OER 0.02 a K=16), sotto attacco sbaglia *convinto* (OER
0.85): un filtro a 0.75 non solo non lo intercetta, è la condizione perché funzioni.
Dettagli in [[E03-option-injection]].

**La superficie lessicale è quasi interamente inerte.** Il primo operatore, la sostituzione di
bigrammi, produceva **zero flip** in 30 casi e 600 generazioni: il ramo è stato rimosso perché
non ha più niente da dire. Quello che resta, la sostituzione di sinonimi da WordNet
([[E12b-sinonimi-rari]]), dà **un flip su trenta** e un ASR di 0.009. Non serve una ricerca per
saperlo, ed è la ragione per cui [[E03-option-injection]] funziona: l'attacco non cerca una
parola, scrive un'istruzione.

Corollario misurato sull'etichettatura ([[E13-label-semantics]], N=1000): a parita' di K e di
stato, rinominare le opzioni da `calendar_query` a `I7` costa **25-28pp** (26.5pp a K=4),
costante su K=4..16. Il formato non c'entra: `I7: calendar_query` è indistinguibile dalle
etichette reali. E con etichette anonime l'accuracy scende **sotto il caso** (0.099 a K=8,
0.039 a K=16): informazione spuria, non assenza di informazione.

Le etichette sono quindi un canale di informazione reale, ma **non sono attaccabili**
(backend): serve misurarlo, non sfruttarlo. Il canale che l'attaccante controlla resta lo
stato, ed è l'unico su cui si puo' agire senza toccare il backend.

## 2. Bias strutturali noti (documentati upstream)
| leva | nota | issue |
|---|---|---|
| `noul` segue le label `false:`/`true:` | [[noul-label-bias]] | #156 |
| `score` non sceglie mai il primo livello | [[score-position-bias]] | #131 |

## 3. Calibrazione / confidence
| leva | nota |
|---|---|
| Over-confidence ereditata (temperature non rifitte) | [[calibrazione-fratturata]] |
| Confidence che non segnala OOD/lingua sbagliata | [[calibrazione-fratturata]] |

## 4. Semantici / contestuali
| leva | nota |
|---|---|
| Intent splitting / decomposition (multi-fatto distribuito) | [[prompt-injection-encoder]] |
| Polisemia e parole chiave fuori contesto | [[prompt-injection-encoder]] |
| Dominio out-of-distribution (zero-shot ~ chance) | [[laya]] |

## 5. Meta-ottimizzazione
| leva | nota |
|---|---|

## Priorità ipotizzate (da validare con [[E01-baseline]])
1. **[[token-budget-head]]** — degradazione misurata e documentata: attaccare le *descrizioni* delle opzioni per affamarle.
2. **[[noul-label-bias]]** — bug confermato, easy win se esteso.
3. **[[truncation-stato]]** — niente gradient, basta posizionare il segnale critico in coda.
4. **[[routing-script-bypass]]** — traslitterazione/script switch per far cadere sul checkpoint sbagliato.
5. Attacchi gradient-based (gray-box, pesi disponibili) con transfer → [[black-box-vs-gray-box]].

## Cosa NON è attaccabile
- Il formato della risposta (type-safe per costruzione)
- La generazione di testo (non c'è)
- La definizione della decisione (`questions`/`criteria`): toccarli = muovere il goalpost. Vedi .

## Note collegate
[[architettura-laya]] · [[primitive-typed]] · [[esperimenti]] · [[metriche-attacco]]
