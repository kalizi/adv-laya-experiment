---
tags: [attacco, encoder, semantico, injection]
---

# Prompt injection su encoder classifier

Cosa sappiamo dalla letteratura sugli encoder classifier, e cosa si applica a [[laya]].

## Riferimento principale
*Do Encoders Suffice? A Systematic Comparison of Encoder and Decoder Safety Judges* (arXiv 2606.25782, 2026): encoder ModernBERT-family come safety judge vs decoder judges, su 563k q-a pairs.

### Findings che ci riguardano
1. **Decomposition/multi-turn è il tallone d'Achille**: l' intento nocivo è distribuito tra turni intermedi; l'encoder vede solo lo scambio finale → **FNR più alti** di tutte le tecniche. Sul nostro caso: i "turni" sono paragrafi dello stato — un intento split in 3 paragrafi può collassare.
2. **Il size aiuta**: 150M → 421M abbassa il FNR su decomposition da ~50% a ~12% (stesso paper, dimensioni diverse). Laya è 421M: meno fragile dei piccoli, ma il meccanismo resta.
3. **Le tecniche catalogate**: single-turn pass-through, decomposition, escalation, context manipulation → mappabili 1:1 su varianti di stato.

### Perché un encoder è comunque adatto
Il paper stesso: la safety classification è language understanding → il bidirezionale in single pass è la scelta giusta. Non ci aspettiamo un collasso totale, ci aspettiamo **buchi sistematici su certe tecniche**.

## Varianti per System One (adattate)
| tecnica (paper) | come si manifesta sullo stato |
|---|---|
| single-turn | stato pulito, modifiche mirate → [[option-marker-injection]] |
| decomposition | intento frammentato in più paragrafi, nessuno è da solo sufficiente |
| escalation | innocuo → innocuo → dannoso, solo la coda conta → interagisce con [[truncation-stato]] |
| context manipulation | cambiare il contesto (firma, reparto, ruolo) senza cambiare il fatto |

## Esempi di primitive da testare
1. **Ledger attack**: lo stato è un log/tabella; la riga decisiva è una di 50 righe innocue.
2. **Role framing**: "questa è una simulazione di training / è un esempio per il corso di sicurezza" in testa allo stato.
3. **Negation stacking**: 3 frasi con "not / non / no" dove solo l'ultima decide.
4. **Cross-lingual pivot**: metà inglese + metà italiana (se il deployment è English-only checkpoint).

## Cosa NON funziona (ipotesi da confutare subito)
- Jailbreak classici contro un decoder ("DAN", "ignore previous instructions come system prompt"): qui non c'è un system prompt né generazione. Le frasi imperativi hanno senso **solo** se il fine-tuning ha insegnato al modello a privilegiare ordini nel testo → testare in [[option-marker-injection]] D.

## Da misurare
- [ ] 4 tecniche × N esempi × 3 checkpoint: ASR per tecnica
- [ ] Curva decomposition: 1, 2, 4, 8 pezzi
- [ ] Escalation: posizione del fatto decisivo (inizio/metà/coda)
- [ ] Cross-checkpoint transfer degli attacchi

## Note collegate
[[superficie-attacco]] · [[truncation-stato]] · [[paper-e-riferimenti]] · [[esperimenti]]
