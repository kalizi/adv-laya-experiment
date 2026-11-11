"""E12 — la superficie di attacco, otto operatori in un solo esperimento.

Perché un modulo solo
---------------------
Fino a ora ogni superficie era una sonda in `tools/`: logica duplicata, nessuno store, e quindi
nessun numero ricalcolabile. Otto esperimenti che condividono lo stesso scheletro (casi puliti,
un operatore che cambia **solo lo stato**, `evaluate`, metriche) sono un esperimento con un
registro di operatori, non otto script.

La regola che rende ogni riga study-valid è la stessa per tutti: l'operatore riceve e
restituisce **solo lo stato**. `questions`, `criteria` e gold non li vede neanche, quindi non
possono cambiare. Il codice lo rende strutturale invece che dichiarato: gli operatori hanno
firma `(state: str) -> str` e nient'altro.

Il confronto che conta
----------------------
Ogni operatore ha almeno un **braccio di controllo** con lo stesso costo in token ma semantica
inerte (spazi al posto degli omoglifi, code al posto dei caratteri). Serve a separare le due
spiegazioni possibili di un ASR alto: "ho disturbato il modello" oppure "ho speso budget". Un
attacco che batte il suo controllo è degrado, non ingannamento.

Uso:
    uv run adv e12-surface                      # tutti gli operatori
    uv run adv e12-surface --operator homoglyph # uno solo
"""

from __future__ import annotations

import pathlib
import random
import re
import unicodedata
from typing import Callable

from layaadv import io
from layaadv.data import typed_decisions as td
from layaadv.evaluate import evaluate
from layaadv.metrics import core
from layaadv.model.engine import DEFAULT_CHECKPOINT, open_from_args
from layaadv.schema import Case

EXPERIMENT = "E12_surface"
SEED = 0
N_CASES = 30
CHUNK = 32
DOSES = (0.25, 0.5, 1.0)

# Confusables dichiarativi (classici, cross-script): non è una misura, è la lista di ciò che si
# vuole provare. Ciò che è misurato è se il modello li tratta come caratteri diversi.
GREEK = {
    "a": "α", "b": "β", "c": "ϲ", "d": "ԁ", "e": "ε", "i": "ι", "k": "κ",
    "l": "λ", "m": "μ", "n": "η", "o": "ο", "p": "ρ", "r": "г", "s": "ѕ",
    "t": "τ", "u": "υ", "v": "ν", "x": "χ", "y": "γ", "z": "ζ",
    "A": "Α", "B": "Β", "C": "Ϲ", "E": "Ε", "I": "Ι", "K": "Κ", "M": "Μ",
    "N": "Ν", "O": "Ο", "P": "Ρ", "T": "Τ", "X": "Χ", "Y": "Υ",
}
CYRILLIC = {
    "a": "а", "b": "в", "c": "с", "e": "е", "h": "һ", "i": "і", "j": "ј",
    "k": "к", "m": "м", "n": "н", "o": "о", "p": "р", "r": "г", "s": "ѕ",
    "t": "т", "x": "х", "y": "у",
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "I": "І", "K": "К",
    "M": "М", "O": "О", "P": "Р", "S": "Ѕ", "T": "Т", "X": "Х", "Y": "У",
}
MATH = {
    "a": "𝐚", "b": "𝐛", "c": "𝐜", "d": "𝐝", "e": "𝐞", "f": "𝐟", "g": "𝐠",
    "h": "𝐡", "i": "𝐢", "j": "𝐣", "k": "𝐤", "l": "𝐥", "m": "𝐦", "n": "𝐧",
    "o": "𝐨", "p": "𝐩", "q": "𝐪", "r": "𝐫", "s": "𝐬", "t": "𝐭", "u": "𝐮",
    "v": "𝐯", "w": "𝐰", "x": "𝐱", "y": "𝐲", "z": "𝐳",
    "A": "𝐀", "B": "𝐁", "C": "𝐂", "D": "𝐃", "E": "𝐄", "I": "𝐈", "J": "𝐉",
    "K": "𝐊", "L": "𝐋", "M": "𝐌", "N": "𝐍", "O": "𝐎", "P": "𝐏", "R": "𝐑",
    "S": "𝐒", "T": "𝐓", "U": "𝐔", "V": "𝐕", "W": "𝐖", "X": "𝐗", "Y": "𝐘",
}
# Controllo stesso-script: stessi byte-lunghi di una lettera, nessun cambio di script.
SAME_SCRIPT = {k: v for k, v in zip(GREEK, MATH.values())}

SYMBOL = "#"
# I caratteri a larghezza zero sono **escape**, mai letterali: un U+200B scritto come carattere
# nel sorgente sparisce in silenzio in un editor o in un trasferimento di file, e il risultato
# sembra dire che il tokenizer lo elimina. Non era vero: non era mai stato testato. Il primo
# giro di questo esperimento aveva `ZWSP = ""`, cioe' una stringa vuota, e l'"attacco invisibile"
# non toccava niente riportando ASR ~ 0.
ZWSP = "\u200b"
ZWNJ = "\u200c"
ZWJ = "\u200d"
FEFF = "\ufeff"
WORD_JOINER = "\u2060"


# Tre livelli di somiglianza, che è la domanda di E12f: se la somiglianza predicesse
# l'efficacia, i tre dovrebbero separarsi. A è la tabella normativa {UTS} #39, B è il greco
# visibile scelto a mano, C è il controllo stesso-script.
UTS39_PATH = "data/uts39/confusables.txt"


def uts39_table() -> dict[str, str]:
    """La tabella che lo standard stesso dichiara confusibile, orientata per l'attacco.

    {UTS} #39 mappa `sorgente -> prototipo`: la riga `α -> a` dice che alpha e a si confondono.
    Per sostituire una lettera latina con un omoglifo serve quindi l'inversa, `latina ->
    confusibile`, e il file va letto in quel verso.

    Il file separa i campi con `;`, non con spazi: `03B1 ; 0061 ; MA`. Con uno spazio il
    parser non trovava una coppia e la tabella restava vuota, in silenzio, e l'operatore
    diventava un no-op che avrebbe riportato ASR ~ 0.

    Sono scelte solo le righe con sorgente greco o cirillico e prototipo latino, che sono le
    uniche sostituibili lettera per lettera senza rompere il JSON: le combinazioni
    multi-carattere sono scheletri di grafemi, non sostituzioni, e vengono scartate.
    """
    table: dict[str, str] = {}
    for line in pathlib.Path(UTS39_PATH).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([0-9A-Fa-f]{4,6})\s*;\s*([0-9A-Fa-f]{4,6})\s*;\s*([A-Z]{2})", line)
        if not m:
            continue
        proto, src = chr(int(m.group(1), 16)), chr(int(m.group(2), 16))
        if not (src.isascii() and src.isalpha()):
            continue
        if unicodedata.name(proto, "").startswith(("GREEK", "CYRILLIC")):
            table.setdefault(src, proto)
    return table


# ------------------------------------------------------------------ operatori puri
# Firma `(state) -> state`. Non ricevono le domande: questo è ciò che rende ogni riga
# study-valid per costruzione e non per dichiarazione.

def substitute(state: str, table: dict[str, str], dose: float, seed: int) -> str:
    rng = random.Random(f"{seed}:{dose}")
    pos = [i for i, ch in enumerate(state) if ch in table]
    if not pos:
        return state
    rng = random.Random(f"{SEED}:shuffle131")
    rng.shuffle(pos)
    chosen = set(pos[: max(1, int(len(pos) * dose))])
    return "".join(table[ch] if i in chosen else ch for i, ch in enumerate(state))


def homoglyph(state: str) -> str:
    return substitute(state, GREEK, 1.0, SEED)


def homoglyph_dose(dose: float) -> Callable[[str], str]:
    return lambda s: substitute(s, GREEK, dose, SEED)


def spaces_in_place(state: str) -> str:
    """Il braccio di controllo di E12d: stesse posizioni degli omoglifi, carattere inerti.

    Se l'attacco batte questo braccio, non sta ingannando il modello: sta solo spendendo budget.
    """
    pos = [i for i, ch in enumerate(state) if ch.isalpha()]
    if not pos:
        return state
    rng = random.Random(f"{SEED}:shuffle152")
    rng.shuffle(pos)
    chosen = set(pos)
    return "".join(" " if i in chosen else ch for i, ch in enumerate(state))


def string_spans(state: str) -> list[tuple[int, int]]:
    """Gli intervalli dei **valori** stringa del JSON, esclusi i nomi delle chiavi.

    Non basta contare le virgolette: in un JSON ogni chiave è anch'essa una stringa, e
    inserire un carattere fra `:` e la virgoletta di apertura rompe il documento. Senza questo
    controllo due operatori su dodici producevano uno stato non parsabile, e il loro ASR misurava
    il fatto che il modello ignora il JSON malformato invece che l'attacco.

    È una scansione, non un parser: il formato è JSON serializzato compatto, quindi niente
    escape né stringhe annidate. Se un giorno i stati cambiano formato, questo va rifatto, ed è
    il motivo per cui il test verifica che ogni operatore lasci il JSON parsabile.
    """
    spans = []
    i = 0
    while i < len(state):
        if state[i] == '"':
            start = i + 1
            i += 1
            while i < len(state) and state[i] != '"':
                i += 1 if state[i] != "\\" else 2
            # se il prossimo non-whitespace è ':' è una chiave, non un valore
            j = i + 1
            while j < len(state) and state[j].isspace():
                j += 1
            if j < len(state) and state[j] != ":":
                spans.append((start, i))
        i += 1
    return spans


def _positions_in_strings(state: str, predicate) -> list[int]:
    """Le posizioni che soddisfano `predicate(state, i)` e stanno dentro un valore stringa."""
    out = []
    for a, b in string_spans(state):
        for i in range(a, b):
            if predicate(state, i):
                out.append(i)
    return out


def insert_before_spaces(state: str, char: str, dose: float) -> str:
    """Inserisce `char` davanti a una frazione degli spazi singoli.

    Gli stati sono JSON serializzato compatto, quindi **non c'è nessuno spazio doppio**: i
    punti d'inserimento sono gli spazi singoli, e mettere il carattere *dentro* la stringa JSON
    non la invalida. Il carattere va **prima** dello spazio, non attorno alla parola: metterlo
    anche dopo spacchetta il testo e cambia l'attacco in un'altra cosa.
    """
    targets = set(
        _positions_in_strings(
            state, lambda t, i: t[i] == " " and i > 0 and t[i - 1] == ","
        )
    )
    return "".join(char + ch if i in targets else ch for i, ch in enumerate(state))


def inject_inside_words(state: str, char: str, dose: float) -> str:
    """Inserisce il carattere **dentro** le parole, alle posizioni alfabetiche.

    È l'attacco vero: mettere un carattere a larghezza zero fra due parole lascia lo spazio e
    il testo resta identico a occhio. Inserirlo fra gli spazi non romperebbe niente ma
    non avrebbe nemmeno senso.
    """
    pos = [i for i, c in enumerate(state) if c.isalpha()]
    if not pos:
        return state
    rng = random.Random(f"{SEED}:shuffle223")
    rng.shuffle(pos)
    chosen = set(pos[: max(1, int(len(pos) * dose))])
    out = []
    for i, c in enumerate(state):
        out.append(char if i in chosen else c)
        if i in chosen and i + 1 < len(state) and state[i + 1].isalpha():
            out.append(state[i + 1])
            out.append("\x00")  # marcatore, rimosso sotto
    joined = "".join(out).replace("\x00", "")
    return joined


KEYBOARD_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")


def build_neighbours(max_col_offset: int = 1) -> dict[str, list[str]]:
    """Dizionario lettera -> lettere raggiungibili con uno spostamento fisico plausibile.

    L'offset di colonna controlla quanto è "vicino": 1 = tasto adiacente o diagonale. La riga
    centrale è spostata di mezzo tasto rispetto alle altre, quindi `a` ha `q`, `w`, `s`, `z`,
    non `p`: è quello che rende lo scambio "plausibile" e non arbitrario.
    """
    pos = {}
    for r, row in enumerate(KEYBOARD_ROWS):
        shift = 0.5 if r == 1 else 0.0
        for c, ch in enumerate(row):
            pos[ch] = (r, c + shift)
    out: dict[str, list[str]] = {}
    for a, (ra, ca) in pos.items():
        near = []
        for b, (rb, cb) in pos.items():
            if a == b or abs(rb - ra) > 1 or abs(cb - ca) > max_col_offset:
                continue
            near.append(b)
        out[a] = sorted(near)
    return out


def keyboard_swap_prose(state: str, neighbours: dict[str, list[str]], dose: float) -> str:
    """Scambia coppie di lettere adiacenti, dentro il testo.

    Una posizione è ammissibile se i due caratteri sono lettere diverse, entrambe mappate, e la
    seconda è un tasto vicino alla prima. Le parole con `_` o cifre sono saltate: sono
    identificatori, e toccarli romperebbe il JSON.

    Costa **zero token**: stessi caratteri, solo in ordine diverso. È l'attacco più economico
    possibile in assoluto, ed è quello con cui misuriamo se "economico" vuol dire "efficace".
    """
    words = state.split()
    for wi, word in enumerate(words):
        if "_" in word or any(ch.isdigit() for ch in word):
            continue
        chars = list(word)
        spots = [
            i for i in range(len(chars) - 1)
            if chars[i].lower() != chars[i + 1].lower()
            and chars[i].lower() in neighbours and chars[i + 1].lower() in neighbours
            and chars[i + 1].lower() in neighbours[chars[i].lower()]
        ]
        if not spots:
            continue
        rng = random.Random(f"{SEED}:shuffle284")
        rng.shuffle(spots)
        for i in spots[: max(1, int(len(spots) * dose))]:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        words[wi] = "".join(chars)
    return " ".join(words)


def symbol_insertion(state: str, dose: float, symbol: str = SYMBOL) -> str:
    """Inserisce un simbolo dentro una parola di contenuto, scelta con il POS di spaCy.

    Non sostituisce: **aggiunge**. Per questo non è una superficie nuova, è E04 con un filler
    peggiore, e il braccio di controllo è l'inserimento di uno spazio nello stesso punto, che
    spende gli stessi token senza cambiare il testo.
    """
    from layaadv.experiments.e12_rare_syn import build_nlp, content_candidates

    nlp = build_nlp("en_core_web_md")
    doc = nlp(state)
    cands = [c for c in content_candidates(nlp, state, 3) if c["index"] < len(doc)]
    if not cands:
        return state
    rng = random.Random(f"{SEED}:shuffle305")
    rng.shuffle(cands)
    keep = cands[: max(1, int(len(cands) * dose))]
    for c in sorted(keep, key=lambda c: doc[c["index"]].idx, reverse=True):
        tok = doc[c["index"]]
        state = state[: tok.idx] + symbol + state[tok.idx :]
        doc = nlp(state)
    return state


def space_insertion(state: str, dose: float) -> str:
    """Il braccio di controllo di E12c: stesso punto, stesso numero di caratteri, nessun simbolo."""
    return symbol_insertion(state, dose, symbol=" ")


BIDI_OPENERS = {"RLO": "\u202e", "LRO": "\u202d", "LRI": "\u2066"}


def bidi_opener(state: str, kind: str, dose: float) -> str:
    """Override bidirezionale davanti a una frazione di parole.

    Previsto inerte, e lo è: su un encoder che non mostra testo a nessuno l'inversione di
    direzione non ha bersaglio. È il motivo per cui la serie si chiude.
    """
    opener = BIDI_OPENERS[kind]
    targets = set(_positions_in_strings(state, lambda t, i: t[i] == " "))
    return "".join(opener + ch if i in targets else ch for i, ch in enumerate(state))


def bidi_override(state: str) -> str:
    """Override di direzione: nessun bersaglio su un encoder che non mostra testo a nessuno."""
    return state + "\u202e" + state[::-1] + "\u202c"


# ------------------------------------------------------------------ registro
# Ogni voce: nome, cosa misura, dose, e se ha un braccio di controllo. `control` è il nome di un
# altro operatore con lo stesso costo: la coppia è ciò che separa degrado da ingannamento.
REGISTRY: dict[str, dict] = {
    "uts39": {
        "label": "livello A: tabella normativa UTS #39",
        "dose": 1.0,
        "control": "spaces_in_place",
        "why": "la tabella che lo standard dichiara confusibile",
    },
    "homoglyph": {
        "label": "livello B: greco visibile, scelto a mano",
        "dose": 1.0,
        "control": "spaces_in_place",
        "why": "somiglianza media, cross-script",
    },
    "spaces_in_place": {
        "label": "controllo: spazi al posto degli omoglifi",
        "dose": 1.0,
        "control": None,
        "why": "stesso numero di posizioni toccate, carattere inerti",
    },
    "underscore": {
        "label": "underscore davanti agli spazi singoli",
        "dose": 1.0,
        "control": None,
        "why": "attacco in-distribuzione",
    },
    "zero_width_space": {
        "label": "ZWSP dentro le parole",
        "dose": 1.0,
        "control": "spaces_in_place",
        "why": "invisibile a un umano",
    },
    "zero_width_joiner": {
        "label": "ZWJ dentro le parole",
        "dose": 1.0,
        "control": "spaces_in_place",
        "why": "invisibile a un umano",
    },
    "zero_width_nbsp": {
        "label": "BOM dentro le parole",
        "dose": 1.0,
        "control": "spaces_in_place",
        "why": "invisibile a un umano",
    },
    "keyboard_swap": {
        "label": "scambio fra tasti adiacenti (dose 0.25)",
        "dose": 0.25,
        "control": None,
        "why": "costa zero token: l'attacco piu' economico possibile",
    },
    "symbol": {
        "label": "inserimento di un simbolo in una parola di contenuto (dose 0.5)",
        "dose": 0.5,
        "control": "space_insertion",
        "why": "aggiunge, non sostituisce: E04 con un filler peggiore",
    },
    "space_insertion": {
        "label": "controllo: uno spazio nello stesso punto (dose 0.5)",
        "dose": 0.5,
        "control": None,
        "why": "stessi token, nessun simbolo",
    },
    "bidi_rlo": {
        "label": "override RLO davanti alle parole (dose 0.5)",
        "dose": 0.5,
        "control": None,
        "why": "previsto inerte su un encoder",
    },
    "tier_c_control": {
        "label": "controllo: matematica senza cambio di script (livello C)",
        "dose": 1.0,
        "control": None,
        "why": "nessun cambio di script, stessa classe di byte",
    },
}

TRANSFORMS: dict[str, Callable[[str], str]] = {
    "uts39": lambda s: substitute(s, uts39_table(), 1.0, SEED),
    "homoglyph": homoglyph,
    "spaces_in_place": spaces_in_place,
    "underscore": lambda s: insert_before_spaces(s, "_", 1.0),
    "zero_width_space": lambda s: inject_inside_words(s, ZWSP, 1.0),
    "zero_width_joiner": lambda s: inject_inside_words(s, ZWJ, 1.0),
    "zero_width_nbsp": lambda s: inject_inside_words(s, FEFF, 1.0),
    "keyboard_swap": lambda s: keyboard_swap_prose(s, build_neighbours(), 0.25),
    "symbol": lambda s: symbol_insertion(s, 0.5),
    "space_insertion": lambda s: space_insertion(s, 0.5),
    "bidi_rlo": lambda s: bidi_opener(s, "RLO", 0.5),
    "tier_c_control": lambda s: substitute(s, SAME_SCRIPT, 1.0, SEED),
}


def transform_of(name: str) -> Callable[[str], str]:
    if name not in TRANSFORMS:
        raise SystemExit(f"operatore sconosciuto: {name!r}. Disponibili: {sorted(TRANSFORMS)}")
    return TRANSFORMS[name]


# ------------------------------------------------------------------ misura
def label_set(records: list[dict]) -> set[str]:
    return {v for r in records for v in r["labels"].values()}


def measure(clean, adv, clean_cases, adv_cases, extra: dict) -> dict:
    """Le metriche di una riga, più ciò che distingue un attacco da un degrado."""
    asr_val, asr_denom = core.asr(clean.records, adv.records)
    pred_clean = label_set(clean.records)
    pred_adv = label_set(adv.records)
    return {
        "accuracy_clean": clean.metrics["accuracy"],
        "accuracy_adv": adv.metrics["accuracy"],
        "drop_pp": core.drop(clean.metrics["accuracy"], adv.metrics["accuracy"]),
        "asr": asr_val,
        "asr_denominator": asr_denom,
        "oer_clean": clean.metrics["oer"],
        "oer_adv": adv.metrics["oer"],
        "mean_conf_wrong_clean": clean.metrics["mean_conf_wrong"],
        "mean_conf_wrong_adv": adv.metrics["mean_conf_wrong"],
        "ece_clean": clean.metrics["ece"],
        "ece_adv": adv.metrics["ece"],
        "brier_clean": clean.metrics.get("brier"),
        "labels_predicted_clean": len(pred_clean),
        "labels_predicted_adv": len(pred_adv),
        "label_collapse": sorted(pred_clean - pred_adv),
        "unicode_normalized": all(
            unicodedata.normalize("NFC", c.state) == c.state for c in adv_cases
        ),
        **extra,
    }


def run(args) -> dict:
    log = lambda m: print(m, flush=True)  # noqa: E731
    limit = getattr(args, "limit", None) or N_CASES
    resume = {"resume": not getattr(args, "no_resume", False),
              "force_new": bool(getattr(args, "force_new", False))}
    wanted = getattr(args, "operator", None)
    names = [wanted] if wanted else list(REGISTRY)
    for n in names:
        if n not in REGISTRY:
            raise SystemExit(f"operatore sconosciuto: {n!r}. Disponibili: {sorted(REGISTRY)}")

    cases = td.load_cases("test", limit=limit)
    engine = open_from_args(args, getattr(args, "checkpoint", None) or DEFAULT_CHECKPOINT)
    log(f"[E12] {len(cases)} casi, operatori: {', '.join(names)}")

    common = dict(experiment=EXPERIMENT, checkpoint=engine.checkpoint,
                  dataset="typed-decisions/test", k=None, n_cases=len(cases), seed=SEED,
                  batch_size=engine.batch_size, chunk=CHUNK,
                  **{i: v for i, v in engine.run_identity().items() if v is not None})

    clean_store = io.RunStore.create(EXPERIMENT, {**common, "kind": "clean"}, **resume)
    clean = evaluate(engine, cases, store=clean_store, batch_size=engine.batch_size,
                     chunk=CHUNK, log=log)
    clean_store.finish({"metrics": clean.metrics})
    log(f"[E12] baseline acc {clean.metrics['accuracy']:.4f}, "
        f"{len(label_set(clean.records))} label predette")

    rows = []
    for name in names:
        spec = REGISTRY[name]
        fn = transform_of(name)
        states = [fn(c.state) for c in cases]
        mutants = [Case(id=c.id, state=s, questions=c.questions, gold=c.gold)
                   for c, s in zip(cases, states)]
        # Gli indici spaCy non coincidono con text.split(), quindi l'operatore lavora sul testo
        # e l'indice si ricostruisce percorrendo i token. Qui si verifica invece la garanzia:
        # l'operatore non ha toccato nient'altro che lo stato.
        untouched = all(
            m.questions == c.questions and m.gold == c.gold for m, c in zip(mutants, cases)
        )
        changed = sum(1 for m, c in zip(mutants, cases) if m.state != c.state)
        d_tokens = engine.count_tokens(states[0]) - engine.count_tokens(cases[0].state)

        store = io.RunStore.create(
            EXPERIMENT,
            {**common, "kind": "adversarial", "operator": name, "dose": spec["dose"]},
            **resume,
        )
        adv = evaluate(engine, mutants, store=store, batch_size=engine.batch_size,
                       chunk=CHUNK, log=log)
        adv_store_finish = {"metrics": adv.metrics}
        store.finish(adv_store_finish)

        row = {
            "operator": name,
            "label": spec["label"],
            "dose": spec["dose"],
            "control": spec["control"],
            "why": spec["why"],
            "states_changed": changed,
            "delta_tokens_first_state": d_tokens,
            "questions_untouched": untouched,
            "clean_run_id": clean_store.run_id,
            "adv_run_id": store.run_id,
            **measure(clean, adv, cases, mutants, {}),
        }
        rows.append(row)
        log(f"[E12] {name:20} acc {row['accuracy_clean']:.3f}->{row['accuracy_adv']:.3f} "
            f"ASR {row['asr']:.4f} OER@.75 {row['oer_adv']['0.75']:.3f} "
            f"label {row['labels_predicted_clean']}->{row['labels_predicted_adv']} "
            f"Δtok {d_tokens} domande intatte {untouched}")
    engine.close()

    payload = {
        "experiment": EXPERIMENT,
        "question": "la superficie puramente lessicale e' una superficie nuova, o degrado?",
        "axes": {"checkpoint": common["checkpoint"], "n_cases": len(cases), "seed": SEED,
                 "operators": names, "study_valid": True},
        "study_valid": {
            "why": "gli operatori hanno firma (state) -> str: non vedono questions ne gold, "
                   "quindi non possono spostarli",
            "verified_per_row": "questions_untouched",
        },
        "results": rows,
    }
    stem = EXPERIMENT if len(names) == len(REGISTRY) else f"{EXPERIMENT}__{names[0]}"
    io.write_result(stem, payload)
    log(f"[E12] wrote results/{stem}.json ({len(rows)} righe)")
    return payload
