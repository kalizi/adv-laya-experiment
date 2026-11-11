---
tags: [esperimenti, template]
---

# Template esperimento

Copia questa nota in `40-esperimenti/E##-nome.md`, riempi tutto. Campi obbligatori tra `<>`.

```markdown
---
tags: [esperimento, <categoria>]
stato: idea            # idea | in-corso | fatto | scartato
autore: <>
data: YYYY-MM-DD
---

# E## — <titolo, ipotesi in una riga>

## Ipotesi
<
Se X, allora <metrica> si muove di <direzione> perché <meccanismo>.
Meccanismo → [[<nota leva>]]
>

## Setup
- checkpoint: `convaiinnovations/<...>` (hash pesi: <>)
- dispositivo: <> (CPU / T4 / altro → [[configurazione-ambiente]])
- dataset: <> split <> N=<>
- questions: invariate (byte-identiche: sì/no)
- baseline riferimento: da [[E01-baseline]] → acc=<>

## Metodo
<
Passi esatti. Se usi LLM per generare gli input: dichiaralo (regola study).
Codice: modulo in `src/layaadv/experiments/` + subcommand in `src/layaadv/cli.py` (comando: `uv run adv e##`)
>

## Risultati
| metrica | baseline | avversario | delta |
|---|---|---|---|
| accuracy | | | |
| ASR | — | | |
| OER (conf ≥ 0.75 e torto) | | | |
| mean confidence (torto) | | | |
| ECE | | | |

N attaccati = <>, N già-sbagliati-esclusi = <>

## Output
- `results/E##_adv.jsonl`
- `results/E##_log.txt`

## Verdetto
- drop ≥ 20pp? sì/no
- ASR ≥ 50%? sì/no
- overconf ≥ 0.75? sì/no
- → livello : 🥉 / 🥈 / 🥇 / non conta

## Cosa è andato storto / repliche
<>

## Link
[[<leva>]] · [[<dataset>]] · [[esperimenti]]
```

## Regole di compilazione
- **Mai** lasciare `<...>` nel finale: se un campo non si applica, scrivi `N/A` con una riga di motivo.
- I numeri vanno **copiati dal log**, non a memoria.
- Se lo esegue un'LLM: `autore: <umano> + <modello>` nella frontmatter.
- Se salti un controllo (es. baseline), `stato` resta `in-corso`, mai `fatto`.
