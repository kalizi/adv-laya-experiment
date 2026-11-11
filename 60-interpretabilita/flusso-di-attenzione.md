---
tags: [paper, interpretabilita, attenzione, 2020]
---

# Quantifying Attention Flow in Transformers

**Abnar & Zuidema** · ACL 2020 · arXiv:2005.00928 · `paper/abnar-zuidema-attention-flow.pdf`

## Cosa propone

Nei trasformatori i token perdono identita' salendo di livello, quindi chiedere a un
singolo strato "quanta attenzione va a quel token" e' malposto. Due modi per propagare
l'informazione attraverso i livelli:

- **Attention rollout**: moltiplica le matrici di attenzione lungo il percorso,
  incorporando le connessioni residue (`A = 0.5 W_att + 0.5 I`).
- **Attention flow**: tratta il grafo di attenzione come una rete di flusso e calcola il
   massimo flusso dalle sorgenti (embedding) ai pozzi (token di input).

## I risultati

Entrambi correlano meglio dei pesi grezzi con le misure di ablazione (blank-out) e con i
gradienti di input; il **flow** e' piu' affidabile del rollout, perche' distribuisce il
peso sull'insieme dei token piu' osservati invece di concentrarlo. C'e' anche
un'avvertenza pratica: nei decoder c'e' un bias verso i token iniziali, quindi il metodo
va normalizzato sul campo recettivo.

## Perche' ci riguarda

E' il metodo che E22 **non** ha usato, e la differenza e' significativa. E22 ha misurato
la massa per singolo livello (perche' la domanda era globale vs locale). Una misura di
flusso cumulato risponderebbe a una domanda diversa: quanta informazione dallo span
arriva allo stato finale del `[MASK]`, attraversando tutti i livelli. E' un candidato
naturale per [[E23-meccanismo-injection]] come **seconda lettura**, accanto alle ablazioni.

L'avvertenza sul campo recettivo per noi e' concreta: con 18 livelli locali su 28, una
propagazione che ignora le maschere locali promette un flusso che il modello non puo'
avere. Il flusso va calcolato **rispettando la maschera locale**, ed e' una differenza che
possiamo verificare direttamente.

## Note collegate
[[attenzione-come-spiegazione]] · [[E22-attenzione-laya]]
