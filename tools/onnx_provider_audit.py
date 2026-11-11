"""Dove girano davvero i nodi del grafo, provider per provider.

`session.get_providers()` restituisce l'**ordine di priorita'** della lista, non la
assegnazione dei nodi: una sessione con CUDA EP che non ha il kernel per un operatore
ricade su quei nodi sulla CPU e continua a dichiarare `CUDAExecutionProvider` in testa. Un
esperimento che registra il device da li' pubblicherebbe latenze di CPU etichettate come GPU.

Il profiler è l'unica API che dice la verità per nodo, quindi qui si chiede a lui e si
conta. Serve a due cose:

1. **Il 4 bit gira sulla GPU?** `MatMulNBits` è un operatore recente e la sua disponibilita'
   nel CUDA EP dipende dalla versione di onnxruntime. Se ricadesse tutto sulla CPU, la
   tabella "la quantizzazione non accelera" sarebbe un fatto di CPU pubblicato come se fosse
   di GPU.
2. **I nodi che restano sulla CPU sono gli stessi in ogni grafo?** Se il fallback riguardasse
   i pesi, il confronto fra precisioni sarebbe falsato da un vantaggio di una sola variante.

Uso:
    uv run python tools/onnx_provider_audit.py
    uv run python tools/onnx_provider_audit.py --graph model_q4.onnx
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import tempfile

import numpy as np
import onnxruntime as ort

DEFAULT_GRAPHS = ("model.onnx", "model_fp16.onnx", "model_q4.onnx", "model_q4f16.onnx")
EXPORT = "data/onnx/open-jev-deberta-v3-large/onnx"


def probe(graph: str, providers: list[str]) -> dict:
    """Una sessione, una forward, e il conto dei nodi per provider."""
    tmp = tempfile.mkdtemp(prefix="onnx-audit-")
    opts = ort.SessionOptions()
    opts.enable_profiling = True
    opts.profile_file_prefix = os.path.join(tmp, "prof")
    # log_severity_level=3 tiene via i "Memcpy nodes are added..." e i constant_folding:
    # sono rumore d'informazione, e in un audit contano solo i provider.
    opts.log_severity_level = 3
    sess = ort.InferenceSession(graph, opts, providers=providers)
    feed = {}
    for spec in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in spec.shape]
        feed[spec.name] = (
            np.ones(shape, dtype=np.int64)
            if spec.type == "tensor(int64)"
            else np.random.default_rng(0).random(shape, dtype=np.float32)
        )
    sess.run(None, feed)
    path = sess.end_profiling()
    events = json.loads(open(path).read())
    kernels = [
        e
        for e in events
        if e.get("cat") == "Node" and e["name"].endswith("_kernel_time")
    ]
    by_provider = collections.Counter(e["args"].get("provider", "?") for e in kernels)
    cpu_ops = collections.Counter(
        e["args"].get("op_name", "?") for e in kernels
        if e["args"].get("provider") == "CPUExecutionProvider"
    )
    out = {
        "graph": os.path.basename(graph),
        "nodes_by_provider": dict(by_provider),
        "cpu_ops": dict(cpu_ops),
    }
    os.remove(path)
    os.rmdir(tmp)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", default=EXPORT, help="directory dell'export open-jev")
    ap.add_argument("--graph", action="append", help="un solo grafo (ripetibile)")
    ap.add_argument("--cpu-only", action="store_true", help="forza il solo provider CPU")
    args = ap.parse_args()

    providers = (
        ["CPUExecutionProvider"]
        if args.cpu_only
        else [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
              if p in set(ort.get_available_providers())]
    )
    print(f"onnxruntime {ort.__version__}, provider richiesti: {providers}")
    graphs = args.graph or [os.path.join(args.export, g) for g in DEFAULT_GRAPHS]

    results = []
    for g in graphs:
        if not os.path.exists(g):
            raise SystemExit(f"graffo assente: {g}")
        results.append(probe(g, providers))
        r = results[-1]
        print(f"\n{r['graph']}: {r['nodes_by_provider']}")
        if r["cpu_ops"]:
            top = ", ".join(f"{k}x{v}" for k, v in sorted(r["cpu_ops"].items()))
            print(f"  ricadono su CPU: {top}")

    # Il confronto fra precisioni è pulito solo se i nodi che restano sulla CPU sono gli
    # stessi in ogni grafo: altrimenti una variante ha un vantaggio che non è di precisione.
    cpu_sets = {r["graph"]: frozenset(r["cpu_ops"]) for r in results}
    identical = len(set(cpu_sets.values())) == 1
    print(f"\noperatori CPU identici in tutti i grafi: {'SI' if identical else 'NO'}")
    if not identical:
        for name, ops in cpu_sets.items():
            print(f"  {name}: {sorted(ops)}")
    if args.cpu_only:
        return 0

    cuda = all("CUDAExecutionProvider" in r["nodes_by_provider"] for r in results)
    print(f"\nCUDA esegue almeno un nodo in ogni grafo: {'SI' if cuda else 'NO'}")

    # Il controllo che regge la tabella: nessuna moltiplicazione, pesi o meno, cade sulla CPU.
    # "Almeno un nodo su CUDA" sarebbe troppo debole, perche' un fallback dei soli pesi
    # lascia comunque migliaia di nodi CUDA e la sessione dichiara CUDA in testa.
    heavy = ("MatMulNBits", "MatMul", "Gemm", "Attention")
    on_cpu = sorted({op for ops in cpu_sets.values() for op in ops if op in heavy})
    print(f"moltiplicazioni su CPU in almeno un grafo: {on_cpu or 'nessuna'}")
    verdict = cuda and not on_cpu
    print(f"VERDETTO: {'i pesi girano sulla GPU' if verdict else 'FALLBACK PARZIALE, non confrontabile'}")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
