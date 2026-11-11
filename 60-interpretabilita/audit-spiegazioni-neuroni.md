---
tags: [paper, interpretabilita, unita, 2023]
---

# Rigorously Assessing Natural Language Explanations of Neurons

**Huang, Geiger, D'Oosterlinck, Wu & Potts** · BlackboxNLP@EMNLP 2023 · arXiv:2309.10312 · `paper/huang-geiger-neuron-explanations-audit.pdf`

## Cosa fa

Audita il metodo di [[spiegare-i-neuroni]] con due criteri:

- **modo osservazionale**: la spiegazione, letta come predizione sulle attivazioni, e'
  precisa e completa? (una feature attiva su tutte e sole le stringhe di un concetto)
- **modo intervento**: la spiegazione sopravvive a un intervento causale? (il neurone e'
  un mediatore causale del concetto che dovrebbe rappresentare)

Applicato alle spiegazioni GPT-4 dei neuroni di GPT-2 XL: fra i neuroni che il metodo
stesso considera spiegati meglio (top 0.6%), la **precisione e' 0.64 e il recall 0.50**, e
l'efficacia causale e' pressocche' nulla. Gli autori chiudono chiedendosi se il linguaggio
naturale sia il mezzo giusto per le spiegazioni e se i neuroni siano l'unita' giusta,
propendendo per formalismi strutturati.

## Perche' ci riguarda

E' il paper piu' utile dei tre sull'automatizzazione, perche' e' l'unico che misura
l'affidabilita' invece di proporla. La lezione operativa e' diretta: un punteggio alto
prodotto dallo **stesso** sistema che genera la spiegazione non e' una verifica. Il
controllo deve essere esterno e causale, cioe' un intervento.

Per [[E23-meccanismo-injection]] fissare lo standard: qualsiasi affermazione del tipo
"questa testa/direzione codifica la nominazione" deve reggere un intervento
(l'ablazione o il patching di quella componente cambiano il flip?). Se non lo regge,
l'affermazione resta un'osservazione e va scritta come tale.

## Note collegate
[[unita-di-analisi]] · [[spiegare-i-neuroni]] · [[circuiti-e-interventi]]
