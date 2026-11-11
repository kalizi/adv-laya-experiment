"""Attacchi avversari: un modulo per famiglia (truncation, GA, ...).

Contratto: trasformano lo **stato** di un ``Case`` senza toccare ``gold`` e
``questions`` (goal shift = escluso da ASR, vedi metriche-attacco). Ogni
attacco è un ``run(args) -> dict`` su ``layaadv.evaluate``.
"""
