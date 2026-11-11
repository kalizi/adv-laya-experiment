"""E13 pure-function tests: la rinomina delle opzioni.

Il perche' di questi test sta nel protocollo, non nella forma: E13 tocca i criteria,
quindi non e' study-valid, e il rischio reale e' scrivere un braccio che cambia
qualcosa oltre al testo dell'opzione (e renderebbe il confronto privo di senso).
"""

from __future__ import annotations

from layaadv.experiments.e13_label_semantics import apply_arm, rename_map
from layaadv.schema import Case

LABELS = ["alarm_query", "alarm_set", "cooking_query", "transport_query", "weather_query"]


def base_case() -> Case:
    crit = {lab: "" for lab in LABELS}
    return Case(
        id="0",
        state="svegliami alle cinque",
        questions={"intent": {"type": "choice", "instructions": "Qual e' l'intento?", "criteria": crit}},
        gold={"intent": "cooking_query"},
    )


def test_real_arm_is_the_identity():
    """Il braccio baseline deve lasciare i nomi intatti, altrimenti il confronto
    'short vs real' non ha una baseline."""
    assert rename_map(LABELS, "real") == {lab: lab for lab in LABELS}


def test_short_arm_produces_distinct_short_labels():
    """Se due opzioni ricevessero lo stesso nome la domanda sarebbe ambigua e
    l'accuracy crollerebbe per un motivo che non e' informazione."""
    m = rename_map(LABELS, "short")
    assert len(set(m.values())) == len(LABELS)
    assert all(len(v) <= 3 for v in m.values())


def test_prefix_arm_keeps_the_long_name_inside():
    """Il braccio prefix esiste per isolare la lunghezza dal significato: senza
    il nome lungo non distinguerebbe 'il formato conta' da 'il testo conta'."""
    m = rename_map(LABELS, "prefix")
    for original, renamed in m.items():
        assert original in renamed


def test_apply_arm_renames_gold_and_criteria_together():
    """Il gold deve seguire la rinomina: altrimenti la domanda non avrebbe una
    risposta valida e l'accuracy misurerebbe un dataset rotto."""
    mapping = rename_map(LABELS, "short")
    out = apply_arm(base_case(), mapping)
    assert out.gold["intent"] == mapping["cooking_query"]
    assert set(out.questions["intent"]["criteria"]) == set(mapping.values())


def test_apply_arm_preserves_state_and_option_order():
    """Stato invariato e stesso ordine delle opzioni: l'unica variabile e' il testo.
    Se l'ordine cambiasse, il confronto misurerebbe il posizionamento."""
    case = base_case()
    original_order = list(case.questions["intent"]["criteria"])
    mapping = rename_map(LABELS, "short")
    out = apply_arm(case, mapping)
    assert out.state == case.state
    assert list(out.questions["intent"]["criteria"]) == [mapping[lab] for lab in original_order]


def test_apply_arm_keeps_descriptions_empty():
    """Le descrizioni restano vuote: una descrizione identica al label fa pagare il
    testo due volte (vedi loaders.check_head_budget)."""
    out = apply_arm(base_case(), rename_map(LABELS, "short"))
    assert all(v == "" for v in out.questions["intent"]["criteria"].values())
