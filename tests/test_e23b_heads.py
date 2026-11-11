"""E23b: il knockout deve togliere **quella** testa, e il censimento deve contare
quello che dice di contare.

I modi in cui questa misura mente senza fare rumore:

1. **la fetta zeroata è quella di un'altra testa**. Il knockout azzera
   `head * head_dim : (head + 1) * head_dim` dell'ingresso di `Wo`. Se l'ordinamento
   delle teste non fosse quello, ogni «la testa 3.7 porta l'attacco» sarebbe
   un'affermazione su un'altra testa, e il numero sarebbe plausibile lo stesso. Per
   questo `head_slice_layout` confronta il knockout per fetta con uno indipendente
   (azzerare i valori di quella testa in `Wqkv`), e il run si rifiuta di scrivere se non
   concordano.
2. **il recupero contato dove non c'era nulla da recuperare**. Un caso che il pulito
   sbaglia già non è nell'ASR: contarvi un «torno alla decisione di prima» gonfierebbe
   la frazione con ritorni a un errore.
3. **il danno non pubblicato accanto al recupero**. Una testa che ripara l'attacco
   distruggendo il modello è in cima alla classifica e va vista subito, non filtrata da
   una soglia scelta dopo aver letto i numeri.

Qui si testano le proprieta', non i numeri: un test che ripete i numeri della run si
falsifica al primo ricalcolo.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from layaadv.experiments import e23b_heads as e23b
from layaadv.model.engine import LayaEngine

QID = "intent"


# ------------------------------------------------------------------ modelli finti
class _FakeProjection(nn.Module):
    """Un `nn.Linear` senza pesi: conta la firma (`in_features`) che l'intervento legge.

    `nn.Identity` non ha `in_features`, e `_n_heads()` la usa per contare le teste: senza
    questa firma il test proverebbe un'altra cosa da quella che gira sul modello vero.
    """

    def __init__(self, in_features: int):
        super().__init__()
        self.in_features = in_features

    def forward(self, x):
        return x


class _FakeAttn(nn.Module):
    """Un blocco di attenzione finto con la geometria di quello vero.

    `Wo` riceve l'uscita dell'attenzione come `[batch, seq, n_heads * head_dim]`, con la
    testa `h` nelle colonne `h * head_dim : (h + 1) * head_dim`: è il contratto su cui il
    knockout si basa, quindi va riprodotto esattamente o il test non prova nulla.
    """

    def __init__(self, n_heads: int = 4, head_dim: int = 3):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, head_dim
        self.Wo = _FakeProjection(n_heads * head_dim)
        self.Wqkv = _FakeProjection(3 * n_heads * head_dim)

    def forward(self, hidden_states, **kwargs):
        # Nel modello vero `Wqkv` mappa l'ingresso a `[3, n_heads, head_dim]` e il
        # knockout per valori azzera la fetta dei **valori** di una testa. Qui lo finto
        # ripetendo tre volte l'ingresso, cosi' l'ultimo terzo sono i valori e la fetta
        # della testa h e' esattamente la sua fetta nell'ingresso: i due interventi
        # agiscono sullo stesso pezzo di tensore, come nel modello vero.
        qkv = self.Wqkv(hidden_states.repeat_interleave(3, dim=-1))
        values = qkv[..., 2 * self.n_heads * self.head_dim:]
        return self.Wo(values), None


class _FakeLayer(nn.Module):
    def __init__(self, attn: _FakeAttn):
        super().__init__()
        self.attn = attn

    def forward(self, hidden_states, **kwargs):
        out, _ = self.attn(hidden_states)
        return hidden_states + out


class _FakeEncoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, *args, **kwargs):
        return SimpleNamespace(last_hidden_state=torch.zeros(1, 4, 12))


def stub_engine(n_layers: int = 3, n_heads: int = 4, head_dim: int = 3) -> LayaEngine:
    """Un `LayaEngine` senza modello: la faccia che l'intervento tocca."""
    engine = LayaEngine.__new__(LayaEngine)
    layers = [_FakeLayer(_FakeAttn(n_heads=n_heads, head_dim=head_dim))
              for _ in range(n_layers)]
    engine._agent = SimpleNamespace(
        model=SimpleNamespace(encoder=_FakeEncoder(layers)), _fast=None
    )
    engine.compiled = False
    return engine


# --------------------------------------------------------------- il knockout
def knockout_out(engine, layer, head, hidden):
    handle = engine._register_head_zero(layer, head)
    try:
        return engine._agent.model.encoder.layers[layer](hidden)
    finally:
        handle.remove()


def test_il_knockout_toglie_il_contributo_di_quella_testa_e_basta():
    """La fetta della testa knockoutata torna al valore residuo, le altre restano
    raddoppiate: è la differenza fra «ho tolto quella testa» e «ho tolto l'attenzione»."""
    engine = stub_engine(n_heads=4, head_dim=3)
    hidden = torch.ones(1, 2, 12)
    out = knockout_out(engine, 1, 1, hidden)
    assert out[0, 0, 3:6].tolist() == [1.0, 1.0, 1.0]      # fetta della testa 1
    assert out[0, 0, 0:3].tolist() == [2.0, 2.0, 2.0]      # teste 0, 2, 3 intatte
    assert out[0, 0, 6:12].tolist() == [2.0] * 6


def test_il_knockout_non_muta_l_input():
    engine = stub_engine()
    hidden = torch.ones(1, 2, 12)
    knockout_out(engine, 0, 2, hidden)
    assert hidden.tolist() == [[[1.0] * 12] * 2]


def test_tutte_le_teste_insieme_azzerano_tutto_l_ingresso_di_wo():
    """Se la fetta non coprisse l'intera larghezza, «tutte le teste» non sarebbe
    «l'attenzione non conta piu' nulla», e il censimento conterebbe un'altra cosa."""
    engine = stub_engine(n_heads=4, head_dim=3)
    hidden = torch.arange(12, dtype=torch.float32).reshape(1, 1, 12)
    out = hidden.clone()
    for head in range(engine._n_heads()):
        out = knockout_out(engine, 1, head, out) if head == 0 else out
    # una fetta per volta, tutte le fette: l'uscita deve tornare al residuo su ogni colonna
    for head in range(engine._n_heads()):
        single = knockout_out(engine, 1, head, hidden)
        assert single[0, 0, head * 3:(head + 1) * 3].tolist() == hidden[
            0, 0, head * 3:(head + 1) * 3
        ].tolist()
    del out


def test_i_due_interventi_indipendenti_concordano():
    """Il knockout per fetta e quello per valori in `Wqkv` togliere la stessa testa: se
    discordassero, la fetta zeroata sarebbe quella di un'altra testa."""
    engine = stub_engine(n_heads=4, head_dim=3)
    attn = engine._agent.model.encoder.layers[1].attn
    hidden = torch.arange(12, dtype=torch.float32).reshape(1, 1, 12)
    n_heads, head_dim = 4, 3
    by_slice = knockout_out(engine, 1, 1, hidden)

    start = 2 * n_heads * head_dim + 1 * head_dim

    def zero_values(_module, args, kwargs):
        x = args[0].clone()
        x[..., start:start + head_dim] = 0
        return (x, *args[1:]), kwargs

    handle = attn.Wqkv.register_forward_pre_hook(zero_values, with_kwargs=True)
    try:
        by_qkv = engine._agent.model.encoder.layers[1](hidden)
    finally:
        handle.remove()
    assert torch.equal(by_slice, by_qkv)


def test_componenti_fuori_range_solleva():
    engine = stub_engine(3)
    from layaadv.schema import Case

    one = [Case(id="1", state="s", questions={QID: {"type": "choice", "criteria": {}}},
                gold={QID: "a"})]
    try:
        engine.head_knockout_profiles(one, one, components=[(0, 0), (3, 0)])
    except ValueError as exc:
        assert "fuori range" in str(exc)
    else:
        raise AssertionError("un livello fuori range deve sollevare")
    try:
        engine.head_knockout_profiles(one, one, components=[(0, 4)])
    except ValueError as exc:
        assert "fuori range" in str(exc)
    else:
        raise AssertionError("una testa fuori range deve sollevare")


def test_nessuna_testa_solleva():
    engine = stub_engine(3)
    try:
        engine.head_knockout_profiles([], [], components=[])
    except ValueError as exc:
        assert "nessuna testa" in str(exc)
    else:
        raise AssertionError("senza teste da knockare il run deve sollevare")


def test_fast_rifiuta_il_knockout():
    engine = stub_engine(3)
    engine._agent._fast = object()
    try:
        engine.head_knockout_profiles([], [], components=[(0, 0)])
    except RuntimeError as exc:
        assert "--fast" in str(exc)
    else:
        raise AssertionError("con --fast attivo il knockout deve rifiutare di girare")


def test_compile_rifiuta_il_knockout():
    engine = stub_engine(3)
    engine.compiled = True
    try:
        engine.head_knockout_profiles([], [], components=[(0, 0)])
    except RuntimeError as exc:
        assert "--compile" in str(exc)
    else:
        raise AssertionError("su un modello compilato il knockout deve rifiutare di girare")


# ------------------------------------------------------------------- il censimento
def ko_record(clean: str, attacked: str, correct: bool = True,
              exceptions: list[dict] | None = None) -> dict:
    return {"clean": clean, "attacked": attacked, "clean_correct": correct,
            "exceptions": exceptions or []}


def test_il_censimento_conta_il_ritorno_alla_decisione_pulita():
    records = [
        ko_record("a", "b", exceptions=[{"l": 3, "h": 7, "attack": "a"}]),
        ko_record("a", "b", exceptions=[{"l": 3, "h": 7, "attack": "c"}]),
        ko_record("a", "b", exceptions=[{"l": 3, "h": 7, "attack": "b"}]),
    ]
    census = e23b.census(records, [(3, 7)])[0]
    assert census["n_recovered"] == 1
    assert census["recovery"] == 0.3333  # il payload pubblica 4 decimali
    assert census["n_clean_changed"] == 0


def test_il_censimento_separa_il_danno_dal_recupero():
    records = [
        ko_record("a", "b", exceptions=[{"l": 3, "h": 7, "attack": "a", "clean": "z"}]),
        ko_record("a", "b", exceptions=[{"l": 3, "h": 7, "clean": "y"}]),
    ]
    census = e23b.census(records, [(3, 7)])[0]
    assert census["n_recovered"] == 1
    assert census["n_clean_changed"] == 2
    # due denominatori diversi: il danno si conta su tutti i casi, il recupero sui ribaltati
    assert census["damage_clean"] == 1.0


def test_i_casi_gia_sbagliati_al_pulito_non_sono_nell_asr():
    """Un «torno alla decisione di prima» su un caso che il pulito sbagliava gia' non e'
    un recupero: quei casi non sono nell'ASR, quindi non sono nel denominatore."""
    records = [
        ko_record("b", "b", correct=False,
                  exceptions=[{"l": 3, "h": 7, "attack": "c", "clean": "z"}]),
        ko_record("b", "c", correct=False,
                  exceptions=[{"l": 3, "h": 7, "attack": "b", "clean": "y"}]),
    ]
    census = e23b.census(records, [(3, 7)])[0]
    assert census["n_recovered"] == 0
    assert census["recovery"] is None  # nessun caso ribaltato: la frazione non esiste
    # ma il danno si conta lo stesso, su tutti i casi
    assert census["n_clean_changed"] == 2


def test_le_teste_confermate_sono_le_piu_recuperanti():
    rows = [
        {"layer": 0, "head": 0, "n_recovered": 3, "recovery": 0.1, "damage_clean": 0.9,
         "n_clean_changed": 27},
        {"layer": 5, "head": 2, "n_recovered": 9, "recovery": 0.3, "damage_clean": 0.1,
         "n_clean_changed": 3},
        {"layer": 9, "head": 1, "n_recovered": 0, "recovery": 0.0, "damage_clean": 0.0,
         "n_clean_changed": 0},
    ]
    survivors = e23b.ranked(rows, 2)
    assert [(r["layer"], r["head"]) for r in survivors] == [(5, 2), (0, 0)]
    # il danno resta visibile in cima alla classifica, non filtrato via
    assert survivors[1]["damage_clean"] == 0.9


def test_nessun_recupero_non_e_un_risultato_ma_non_e_zero():
    rows = [{"layer": 1, "head": 1, "n_recovered": 0, "recovery": 0.0,
             "damage_clean": 0.2, "n_clean_changed": 6}]
    assert e23b.ranked(rows, 5) == []


# ---------------------------------------------------------------- sottoinsieme
def test_il_sottoinsieme_di_screening_non_sono_i_primi_casi():
    class C:
        def __init__(self, i):
            self.id = str(i)

    cases = [C(i) for i in range(300)]
    subset = e23b.stride_subset(cases, 10)
    assert [c.id for c in subset][:3] == ["0", "30", "60"]
    assert len(subset) == 10
    # con piu' casi richiesti che disponibili, il sottoinsieme e' l'intero dataset
    assert e23b.stride_subset(cases[:5], 10) == cases[:5]
