---
tags: [paper, interpretabilita, interventi, 2022]
---

# Locating and Editing Factual Associations in GPT (ROME)

**Meng, Bau, Andonian & Belinkov** · NeurIPS 2022 · arXiv:2202.05262 · `paper/meng-bau-rome-locating-editing.pdf`

## Cosa fa

Due cose in un paper: un intervento diagnostico (**causal tracing**) e uno correttivo
(**ROME**).

Il causal tracing corre l'input, poi **ripristina gli stati interni puliti** in una
posizione (livello, token) alla volta, e osserva dove il ripristino recupera la risposta
corretta. Risultato: le associazioni fattuali passano per moduli feed-forward di **medio
livello**, nell'ultimo token del soggetto. ROME poi sfrutta quella localizzazione per
scrivere una nuova associazione con una modifica rank-one.

## Perche' ci riguarda

Il **ripristina un pezzo e guarda se torna** e' esattamente l'esperimento che manca a
[[E03-option-injection]]. L'equivalente per il nostro caso: far girare il caso pulito,
ripristinare nel caso attaccato lo stato al `[MASK]` dell'opzione giusta, livello per
livello, e misurare quanti flip si recuperano. Dove il recupero e' massimo, c'e' la
profondita' a cui l'attacco "si fissa". E' una misura singola, non 448.

L'idea dell'editing ci interessa meno, perche' richiede di decidere **cosa** scrivere, e
quello e' un passo successivo ([[letteratura-collaterale]], SHIFT).

L'avvertenza: la localizzazione trovata su GPT non si trasferisce a un encoder mascherato
con una testa decisionale. L'**indice** di livello non e' un risultato trasferibile, il
**metodo** si.

## Note collegate
[[circuiti-e-interventi]] · [[mediazione-causale]] · [[E23-meccanismo-injection]]
