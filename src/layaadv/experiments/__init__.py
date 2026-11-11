"""Un modulo per esperimento (E00, E01, ...), ognuno con ``run(args) -> dict``.

I moduli orchestano e basta: predizioni via ``layaadv.evaluate``, scrittura via
``layaadv.io``. Niente logica di batching/metriche duplicata qui.
"""
