---
tags: [metodologia, accesso]
---

# Black-box vs gray-box

Livelli di accesso ammessi. Va dichiarato in ogni esperimento, cambia il valore del risultato.

## L0 — Black box puro
- Solo l'API/CLI, nessun peso
- Jev hosted: questo è l'unico livello possibile → costo $0.042/1M token
- Su Laya self-hosted: possibile ma è una scelta (es. attaccare solo via `laya-serve`)
- **Vale di più per la narrative** della study (matcha l'esperienza reale di chi usa Jev)

## L1 — Black box con osservabilità
- Risposte complete: distribuzione, confidence, token counts, quale checkpoint ha risposto
- Già il default con Laya locale: abbiamo tutto `r["answers"]`
- Il transfer da L1 a L0 (Jev API) è il risultato più interessante

## L2 — Gray box (pesi leggibili, gradient)
- Pesi scaricati → gradient su embedding/logits (FGSM, HotFlip, PGD sull'input)
- **Solo su Laya**, mai su Jev
- Il transfer gray→black (attacco ottimizzato su `laya-typed-decisions`, valutato su `laya` o su Jev) è il livello 🥇 di 

## L3 — Full fine-tune access
- Addestrare un attaccante/prefix-tuning sui pesi
- Ammesso? La study dice "qualsiasi cosa", ma va **dichiarato**: è un attacco white-box puro e va separato dalle statistiche L0-L2

## Decisione da prendere (una volta)
- [ ] L0-L1 come categoria principale (tutti possono farla, anche senza GPU)
- [ ] L2 come categoria separata con classifica separata
- [ ] L3 o ammesso dichiarato o escluso

## Raccomandazione
Due classifiche: **"Open" (L0–L1)** e **"Gray" (L2–L3)**, con i premi sulla Open. Motivo: la Open è replicabile da chiunque abbia una CPU e i dataset, e rispetta meglio lo spirito "portfolio per tutti".

## Note collegate
 ·  · [[esperimenti]] · [[configurazione-ambiente]]
