---
tags: [esperimento, superficie, character-level, riordino, typo, negativo]
stato: fatto
data: 2026-09-30
study_valid: true
---

# E12m — riordino di caratteri nei tasti adiacenti

## Perché è l'esperimento più pulito
**Costa zero token**: gli stessi caratteri, solo in ordine diverso. Quindi nessun confound con
il budget, e il confronto con ogni altro attacco della sessione è esatto. Se il modello
cambiasse risposta, non sarebbe per avere mangiato context.

L'ipotesi aggiuntiva è di *plausibilità*: due tasti adiacenti producono un errore di
battitura che un umano non nota, due tasti lontani no. Il confronto fra i due dice se la
plausibilità da refuso conta o se è solo rumore.

## Setup
Tastiera QWERTY fisica, tre righe con la centrale spostata di mezzo tasto. **offset 1** =
adiacente o diagonale (26/26 lettere coperte, media 4.2 vicini per tasto); **offset 3** =
lontano, come controllo. Scambio di due lettere consecutive in posizione, solo in prosa:
mai dentro numeri (cambierebbe il valore) né dentro identificatori (cambierebbe un nome di
campo, che non è un refuso ma un'altra voce).

## Risultati
N=30, baseline 0.7533.

| braccio | dose | Δtoken | drop | ASR | conf | OER |
|---|---|---|---|---|---|---|
| adiacenti | 10% | +21.6 | −0.7pp | 0.0177 | 0.410 | 0 |
| adiacenti | 25% | +21.6 | −0.7pp | 0.0177 | 0.410 | 0 |
| adiacenti | 50% | +21.6 | −0.7pp | 0.0177 | 0.410 | 0 |
| adiacenti | 100% | +23.9 | −0.7pp | 0.0177 | 0.406 | 0 |
| lontani (controllo) | 25% | +34.6 | −1.3pp | 0.0619 | 0.280 | 0 |
| lontani | 50% | +37.0 | −2.0pp | 0.0531 | 0.250 | 0 |
| lontani | 100% | +42.0 | −4.7pp | 0.0354 | 0.326 | 0 |

Drop **negativo ovunque**: l'accuracy migliora. Nessun collasso del vocabolario (12 → 12).
Nessun OER.

Esempio: `"Do not exceed a $50 spend on cloud resources"` →
`"Do not ecxede a $50 spend on colud esrourcse"`, stessa lunghezza, numeri intatti.

## Due limiti, dichiarati
1. **La curva di dose non risolve.** 10% e 100% danno lo stesso Δtoken (21.6) perché
   l'arrotondamento per parola (`max(1, int(spots × dose))`) satura subito: molte parole hanno
   una sola coppia ammissibile. Quindi la curva piatta **non** è un finding sul modello, è un
   limite di risoluzione del mio operatore. Va detto, non letto come saturazione.
2. Il confronto adiacenti/lontani è nell'ordine del rumore (0.0177 contro 0.0354-0.0619) e la
   curva dei lontani non è monotona. L'ipotesi "i tasti lontani fanno peggio" **non è supportata**
   in modo affidabile, ma nemmeno smentita.

## Il risultato che conta: l'ordine non conta
Il Δtoken è quasi nullo (+22 su 30 stati) e l'ASR è 0.0177. Mettilo accanto agli altri
esperimenti della sessione:

| attacco | Δtoken | ASR |
|---|---|---|
| spazi dentro le parole | +120 | 0.7345 |
| zero-width dentro | +232 | 0.6549 |
| `_` dentro | +223 | 0.6637 |
| **swap di tasti (0 token)** | **+22** | **0.0177** |

Il modello **non è sensibile all'ordine dei caratteri**. Ciò che lo rompe è il *volume* di
token aggiunti dentro le parole, non il fatto che una parola diventi sbagliata. `"ecxede"` e
`"exceed"` per il modello sono la stessa identica sequenza di token: lo swap non cambia la
tokenizzazione perché i due caratteri adiacenti stanno già nello stesso token BPE.

Questo è anche la **predizione sbagliata** da registrare: la letteratura dà 64.9% al riordino
dei caratteri, ed è il numero più alto della sua tabella. Da noi non trasferisce, e il motivo
è architetturale e non di dosage.

## Verdetto
- **Negativo**, e con la firma più pulita della sessione: costo zero, nessun effetto.
- **Non passa i ** come tutti gli altri attacchi character-level.
- Il risultato utile è la **confutazione di una previsione**: se l'ordine dei caratteri è
  inerte, il **bidi override** (U+202E/D), che pure dà 52.8% in letteratura, ha una
  previsione sfavorevole da questo lato. Può funzionare solo se il motivo è il *rendering*
  e non i byte — cioè se l'attacco colpisce un umano che guarda, non il modello. Su un
  encoder è improbabile, e va detto.

## Note collegate
[[E12l-zerowidth]] · [[E12i-underscore]] · [[E12d-omoglifi]] · [[E12c-simbolo]] ·  · [[attacchi-pubblicati]] · [[superficie-attacco]] · [[esperimenti]]

## Output
Nessun file in `results/`: sonda one-off.
```bash
uv run python tools/e12m_keyboard_swap.py
```
