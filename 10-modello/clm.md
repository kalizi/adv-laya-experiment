---
tags: [modello, clm, systemone, hard-negatives, termometro]
---

# CLM (Contrastive Language Models)

Il modello più recente della famiglia, e l'unico che sia stato **addestrato esplicitamente
contro la confusione che E03 sfrutta**.

## Scheda
- **Casa:** Stanford (Kwok, Ré) + NVIDIA Research (Pavone, Mirhoseini)
- **Release:** blog 2026-09-19, repo 2026-09-23, HF 2026-09-21
- **Checkpoint:** `Contrastive-LM/CLM-v0.1-8B`, Apache 2.0 (codice e pesi)
- **Categoria:** [[system-one-model]], stessa API typed di [[laya]] e [[jev]]
  (`noul` / `choice` / `score`)
- **Maturità:** nessun arXiv, solo il blog Notion. `Development Status :: 3 - Alpha`,
  8 commit, ~700 like in 11 giorni. Pedigree serio, software alpha.

## Non compete con laya, compete con Jev

CLM dichiara di essere "on par with **Jev**" e misura contro Jev in ogni tabella.
Laya è il parente più vecchio della stessa famiglia, e [[jev]] lo abbiamo già come
backend (`--model deberta-onnx`). Quindi il delta vero è **CLM vs open-jev**, non CLM vs laya.

## Come funziona
- Backbone **Qwen3-8B congelato** + due teste di proiezione da 20M (state head, action head)
- Addestrato con **InfoNCE bidirezionale** su coppie stato-azione: `softmax(100 · cos(z_s, z_a))`
- **Last-token pooling**, nessun BOS, nessun EOS (addestrato su `add_special_tokens=False`)
- Risposta = softmax sugli score. Come in laya e Jev.

### Il meccanismo decisionale è identico, non diverso

`README` di CLM, verbatim: *"a softmax over CLM's scores **is** the answer distribution"*.

Quindi E03 **non** sta attaccando un meccanismo nuovo: sta attaccando lo stesso oggetto
(`dot product` stato-opzione) a circa 100 volte la scala. Le tre cose che sono davvero diverse:

1. **Scala**: 8B frozen contro encoder da ~100M che abbiamo noi (laya 3.5 MB, open-jev ONNX 8.5 MB)
2. **Segnale di addestramento**: 30M di **hard negatives** sintetici, "risposte semanticamente
   simili ma sbagliate", generati da Gemini 2.5 Flash-Lite
3. **Serving disaccoppiato**: stato e azioni passano da encoder separati, con vector cache.
   Le opzioni sono embeddate una volta e riusate, e non le tocca l'attacco

Il punto 2 è il motivo per cui questo modello è interessante e non un duplicato di [[jev]].
Un hard negative è, strutturalmente, **lo stesso oggetto dell'iniezione E03**: un'azione
plausibile ma sbagliata verso cui lo stato viene tirato. CLM ha visto 30M di quelli.
La domanda falsificabile è se questo chiude il buco, e non risulta pubblicata da nessuna parte.

## Il termometro: `logit_scale`

`exp(logit_scale)` nel checkpoint è **100.81**, clampato a 100 in `heads.py:95`.
Quindi la temperatura effettiva è **τ = 1/100.81 = 0.0099**.

Confronto con i nostri checkpoint:

| modello | τ | scala |
|---|---|---|
| `laya-multilingual` | 1.0000 | 1.0 |
| `laya-typed-decisions` (`choice:11+`) | 0.1006 | 9.94 |
| **CLM-v0.1-8B** | **0.0099** | **100.81** |

CLM è **10 volte più affilato** della temperatura che E16 aveva già identificato come
"troppo affilata", e 100 volte più affilato di `laya-multilingual`.

**Ma la scala non cambia l'argmax.** `argmax_k(100 · cos_k) = argmax_k(cos_k)`, quindi
l'ASR di E03 su CLM è governato dalla stessa geometria di laya, e non dalla temperatura.
La temperatura governa la **confidenza**, e quindi `OER`. Tenere separati i due effetti,
altrimenti si attribuisce al termometro un effetto che è dell'attacco o viceversa.

## Fattibilità su questa host

RTX 3080, **10 GB** VRAM. La model card di CLM dice letteralmente
*"Keep --max-model-len 2048 so vLLM fits on a 24 GB GPU"*. vLLM non è installato e
non sta in 10 GB con Qwen3-8B.

La via che funziona è `czl/CLM-v0.1-8B-GGUF` (Apache 2.0, non gated), che è **solo
l'encoder**: le teste le scarica `clm-serve` da `~/.cache/clm/`.

| file | size | verdetto di chi l'ha misurato |
|---|---|---|
| `Qwen3-8B-BF16.gguf` | 16.4 GB | non entra |
| `Qwen3-8B-Q8_0.gguf` | 8.71 GB | troppo stretto per i buffer |
| `Qwen3-8B-Q6_K.gguf` | 7.03 GB | "usable", −2.88 pt di planner accuracy |
| `Qwen3-8B-Q5_K_M.gguf` | 6.24 GB | "usable", −2.05 pt |
| `Qwen3-8B-Q4_K_M.gguf` | 5.49 GB | **sconsigliato**, −6.63 pt |
| `IQ4_XS` | | **rifiutato**, −27.99 pt |

**Sta in 10 GB**: `Q6_K` (7.03) o `Q5_K_M` (6.24) con `-ngl 36`, offload dei pesi **non
serve**. Le varianti `-outq2` (meno 457 MB ciascuna) mettono `lm_head` a `q2_K`, che è
**bit-identico** per il pooling perché la proiezione di uscita non è nel grafo
(misurato: max abs diff 0.0). Serve solo se la VRAM è stretta.

Comando:
```
llama-server -m Qwen3-8B-Q6_K.gguf --embedding --pooling last -c 16384 -np 8 -ngl 36 --port 8090
clm-serve --device cpu --action-cache 0 --port 8700 --no-ui
```

### Smoke test superato il 2026-10-04

Il gate era l'**argmax** sulle due ancore pubblicate, non le probabilità: la carta GGUF
stessa avverte che il numero sulla invoice non è confrontabile perché l'input esatto non è
pubblicato, e che il suo confronto è sempre contro un riferimento bf16 **dello stesso
runtime**.

| ancore | atteso | misurato (Q6_K, llama.cpp b11160) |
|---|---|---|
| tides, rank 1 | 0.99340 (rif. bf16) | **0.99341** |
| tides, rank 2 | 0.00003 | 0.00004 |
| invoice, `choice` | `billing` | **`billing`** ✓ |

Q6_K a 5 cifre sul riferimento bf16: sul caso singolo la quantizzazione è indistinguibile.
Sulla invoice a due opzioni la nostra `billing` è 0.98197 contro 0.98775 del riferimento
bf16: è il caso near-tie che la carta segnala come dove i flip si concentrano, e l'argmax
tiene. **Argmax su entrambe: PASS.**

### Tre fatti che il setup ha costretto, e che vanno dichiarati

1. **`Q6_K` con `-ngl 36` riempie il 94% della 3080**: 9658 MiB occupati, **269 MiB
   liberi**. Le teste **non possono stare in GPU** e la vector cache deve essere spenta
   (`--action-cache 0`). Non è un dettaglio: è la condizione che rende la head su CPU e
   che va nel `run_identity`.
2. **L'`Engine` in-process tira le teste in CUDA per default** e va in
   `torch.OutOfMemoryError` chiedendo 24 MiB: `--device` è del server, non dell'`Engine`,
   quindi l'`Engine` costruita in un processo separato non lo eredita. Il messaggio
   d'errore lo dice: l'encoder teneva 8.51 GiB e il processo delle teste si fermava a
   216 MiB. Quindi il backend **deve** passare dal client HTTP. Non è una scelta di
   architettura, è un vincolo di memoria, ma il risultato è lo stesso backend cliente.

### Costo reale misurato (non stimato)

`input_tokens` non nullo, encoder davvero in esecuzione, testo unico per call:

| K | ms/call (mediana) | N=2974 × 2 condizioni |
|---|---|---|
| 4 | 105.7 | 10.5 min |
| 16 | 161.5 | 16.0 min |
| 40 | 235.0 | 23.3 min |

Curva completa K=4..40 con **entrambi** i template e il pulito condiviso: **~2.6 ore**.
Quindi N pieno è praticabile, e la trappola di E17 (misurare a `--limit 300` e chiamarlo
un verdetto) è evitabile per costo, non per convinzione.

**La trappola del benchmark**: la prima misura che ho fatto dava 4.4 ms a K=4 e
`input_tokens=0`, perché il mio stato e le mie opzioni erano identici a ogni call e
l'embedder ha una LRU. Misurava la cache, non l'encoder, ed era 25x più veloce del reale.
Ogni benchmark di questo backend deve usare testo **mai** riusato, o non sta misurando
niente.

### La quantizzazione è una condizione sperimentale, non un rumore

Quella carta è misurata bene e la conclusione è la nostra: ogni confronto è fatto
contro un **riferimento bf16 dello stesso runtime**, mai contro i numeri vLLM della model
card, altrimenti un delta di runtime viene riportato come un risultato di quantizzazione.
Noi non possiamo girare bf16, quindi:

- il livello di quantizzazione va nel `run_id` **e** nella riga del risultato
- la baseline pulita va **rimisurata** sui nostri dataset, mai importata
- i numeri di CLM non si toccano: sono di un altro dataset e di un altro runtime

Questo è la classe di bug di E15 (`--quantize` accettato e poi ignorato) e di `--dataset`.
Se CLM entra, il backend deve **rifiutare** i selettori degli altri, come fa `--quantize`
contro deberta-onnx.

## Asimmetria train/serve, dichiarata upstream

La carta GGUF la riporta esplicitamente:

> CLM's **training** recipe keeps the **last** tokens of a state while the **serving** path
> truncates from the **head**.

Il servizio **scarta dalla testa**. Quindi la nostra iniezione in `prepend` è quella a
rischio di essere tagliata, mentre `append` è al sicuro. Su `laya-multilingual` il rapporto
coda/testa è già ~3x (ASR 0.9732 contro 0.3752): su CLM ci aspettiamo che il divario sia
**più largo**, perché lì il discard della testa è esplicito e non un effetto di budget.
È un difetto upstream, non nostro, e lo possiamo caratterizzare meglio di loro perché
abbiamo già la macchina (E03 `prepend`/`append`, curva di sopravvivenza, `head_budget_walls`).

## Il bug di metrica che questo modello ci fa trovare

`metrics.core.oer` usa la soglia **0.75** sulla probabilità della top-1. Con τ = 0.0099
la distribuzione di confidenza collassa su 1.0: **una soglia fissa a 0.75 smette di
discriminare** e `OER@0.75` diventa semplicemente "frazione di errori".

È la stessa classe dei due bug che ci sono già costati una correzione:
- ECE dichiarati invece di misurati (E15, 0.417 era di `typed-decisions`)
- device fuori dal `run_id`

Il rimedio è usare il **margin** (`top1 − p2`, che l'API di CLM espone come
`confidence`) o soglie derivate dalla temperatura dichiarata, non una costante ereditata
da un altro modello. Va deciso **prima** di girare, altrimenti la prima tabella CLM
pubblica un numero che sembra nostro e non lo è.

## Previste, da registrare prima di misurare

Motivazione per il vault: `` accetta il **trasferimento cross-checkpoint**
come estensione legittima di [[attacchi-pubblicati]], e quei numeri sono del 22 settembre,
**tre giorni prima** di CLM. Quindi CLM è la terza gamba del trasferimento, non un
doppione.

1. **Geometria (la previsione principale).** I 30M hard negatives non cambiano la geometria
   della scelta. E03 trasferisce con ASR ≥ 0.50 sulla curva K. **Se non trasferisce,
   l'hard-negative training ha comprato davvero robustezza all'iniezione lato stato, e
   quello è il risultato pubblicabile**: non un numero di benchmark ma una causalità sul
   segnale di addestramento. I due esiti sono mutuamente esclusivi ed entrambi pubblicabili,
   ed è per questo che l'esperimento vale.
2. **Termometro.** `OER@0.75` sotto attacco verso il soffitto. Ma la temperatura affilata
   gonfia anche l'**OER pulito**, quindi la *storia* del guardrail rotto trasferisce
   **meno** dell'ASR, non di più. Da dire, non da nascondere.
3. **Posizione.** `prepend` indebolito rispetto a laya, `append` a piena forza, divario
   coda/testa più largo di 3x per via del discard della testa.
4. **Precisione.** `Q5_K_M` contro `Q6_K` come due condizioni distinte: **eseguito**, e
   le due non si distinguono. Zero label diverse su 8 condizioni a N=2974, 64 confronti di
   metriche senza differenze, McNemar con $b=0$ e $c=0$, e 4 probabilità diverse su ~11.900
   con `max |Δp|` 0.024, nessuna delle quali cambia una decisione. Quindi la conclusione
   non sta su un filo di rasoio rispetto al livello di quantizzazione. **Non** sostituisce il
   confronto con bf16, che non sta in 10 GB → [[E21-clm-hard-negatives]]
   con baseline pulita rimisurata per ciascuna.

## Esito: E03 trasferisce (misurato il 2026-10-04)

Protocollo E03 identico, N=2974, `Q6_K`, curva K=4..40, entrambi i template.
**16 righe, `ASR >= 0.50` su 14** (`massive-it` 11/12, `massive-en` 3/4). Le due sotto
sono entrambe `append` a K=4.

Quindi **i 30M hard negatives non comprano nulla** contro l'iniezione lato stato: la
domanda che questo modello apriva ha una risposta, ed è negativa per la difesa e
positiva per la campagna (terza gamba del trasferimento).

Tre cose che il modello fa **diversamente** dai suoi parenti, e che vanno dette:

- il **guardrail si rompe 14--16 volte meno**: `OER@0.75` attaccato 0.0632 contro 0.8914--0.9832 dei
  tre checkpoint `laya` sulla stessa riga di protocollo, cioè 14--16 volte meno;
- **`prepend` e `append` sono quasi uguali** (testa leggermente più alta), dove su laya
  la coda era ~3x la testa. L'asimmetria train/serve è reale ma **inerte** a 30 token
  di stato contro una finestra di 2048;
- **la baseline è più bassa** su MASSIVE-it (0.3689 a K=4 contro 0.634 di laya), quindi
  il drop in punti percentuali non raggiunge i 20pp del criterio 🥉 e l'ASR è il
  denominatore giusto, come E03 aveva già scritto.

Il criterio di confidenza ≥0.75 **non è mai raggiunto** (max 0.6682) e decade con K: è una
media sui torti, e singole decisioni lo superano (l'`OER` attaccato a K=16 è 0.0632).
Non è robustezza: con K=40 opzioni servirebbe `p1 >= 0.756` perché `p1 - mean(p2..pk)` superi 0.75
per aritmetica della softmax. È la terza istanza dello stesso difetto di
[[metriche-attacco]] §3b: **una soglia fissa su una grandezza di confidenza non significa
la stessa cosa ovunque**, e qui la non-portabilità è su K, non su primitiva.

Dettaglio, teorie e limiti: [[E21-clm-hard-negatives]].

## Limiti dichiarati
- Non è peer-reviewed, e la carta GGUF è di un terzo, non del team CLM
- Nessun checkpoint più piccolo: la roadmap dice "larger backbones", quindi 8B è il minimo
- Niente pretraining su testo **adversariale**: la pipeline di dati è Q&A, hard negatives
  generati da un LLM e traiettorie agentiche. Nessuno di questi contiene un'istruzione
  malevola dentro lo stato. Quindi l'assenza di difesa contro E03 è **attesa a priori**,
  ed è quello che rende la misura informativa invece di scontata
- La latenza è un vicolo cieco: il 9x è su H100, un nostro numero sarebbe un aneddoto
  hardware. Saltare l'affermazione

## Note collegate
[[laya]] · [[jev]] · [[architettura-laya]] · [[system-one-model]] · [[E03-option-injection]] ·
[[E21-clm-hard-negatives]] ·
[[E15-quantizzazione]] · [[E16-calibrazione-difesa]] ·  · [[attacchi-pubblicati]]