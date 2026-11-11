"""Il test decisivo: lo stato identico allo stato pulito sposta le previsioni?

Se clean-vs-clean da' 0 cambiamenti, i 2-4 cambi dei braccia precedenti sono
effetti reali della perturbazione. Se da' qualche cambio, allora è rumore
numerico e il "simbolo" non sposta niente piu' di quanto sposti il caso.
"""

from layaadv.data import typed_decisions as td
from layaadv.evaluate import evaluate
from layaadv.model.engine import DEFAULT_CHECKPOINT, LayaEngine

cases = td.load_cases("test", limit=30)
eng = LayaEngine(DEFAULT_CHECKPOINT, batch_size=16)

a = evaluate(eng, cases, batch_size=16, chunk=32)
b = evaluate(eng, cases, batch_size=16, chunk=32)

# stesso input, due valutazioni: il riferimento del rumore
same = sum(
    1
    for x, y in zip(a.records, b.records)
    for qid in x["labels"]
    if x["labels"][qid] != y["labels"][qid]
)
# e la confidenza: è lei a essere rumorosa?
dconf = [
    abs(x["confidence"][qid] - y["confidence"][qid])
    for x, y in zip(a.records, b.records)
    for qid in x["labels"]
]
tot = sum(len(r["labels"]) for r in a.records)
print(f"decisioni: {tot}")
print(f"clean vs clean, label diverse: {same}")
print(f"clean vs clean, differenza confidenza: max {max(dconf):.6f}, media {sum(dconf) / len(dconf):.6f}")

# e con chunk diverso, che cambia il batching in bf16
c = evaluate(eng, cases, batch_size=16, chunk=1)
same_chunk = sum(
    1
    for x, y in zip(a.records, c.records)
    for qid in x["labels"]
    if x["labels"][qid] != y["labels"][qid]
)
print(f"chunk 32 vs chunk 1, label diverse: {same_chunk}")

eng.close()
