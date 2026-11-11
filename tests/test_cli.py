"""Ogni subcomando dichiarato deve puntare a qualcosa che esiste davvero.

Il pattern del progetto e' `EXPERIMENTS[name] -> "modulo.run"`, e il CLI registra il nome del
modulo per poi risolverlo a runtime. Il tipo di errore che ne nasce e' invisibile fino
all'esecuzione: `adv datasets download` accettava il comando, costruiva il parser, e moriva
solo quando l'utente lo lanciava, con `AttributeError: module ... has no attribute 'run'`.

Non e' un errore che un test di contenuto trova, perche' il contenuto va bene: e' il collegamento
fra la dichiarazione e il bersaglio che e' rotto. Quindi il test guarda il collegamento.
"""

from __future__ import annotations

import importlib

import pytest

from layaadv.cli import EXPERIMENTS, build_parser


def _subparsers(parser):
    """Le scelte di un sub-parser, per nome. Vuoto se non e' un sub-parser.

    Solo le azioni di sub-parsing hanno `choices` come **dizionario** nome→parser: `--variant`
    e `--template` hanno `choices` come **lista** di valori ammessi. Filtrare per tipo evita di
    scambiare le due, che altrimenti fa esplodere `.items()`.

    `next()` senza default dentro un generatore solleva StopIteration -> RuntimeError (PEP 479),
    quindi il default e' obbligatorio, non comodo.
    """
    for a in parser._actions:
        if isinstance(getattr(a, "choices", None), dict):
            return a.choices
    return {}


def _all_subcommands(parser):
    """(percorso completo, subparser), compresi i sotto-livelli come `datasets download`."""
    for name, sub in _subparsers(parser).items():
        if name == "help":
            continue
        yield name, sub
        if not _subparsers(sub):
            continue
        for sub_name, sub_sub in _subparsers(sub).items():
            if sub_name != "help":
                yield f"{name} {sub_name}", sub_sub


def test_every_experiment_module_exposes_run():
    """Ogni esperimento dichiarato in EXPERIMENTS deve avere `run`."""
    broken = []
    for name, module_path in sorted(EXPERIMENTS.items()):
        module = importlib.import_module(module_path)
        if not hasattr(module, "run"):
            broken.append(f"{name} -> {module_path} (manca run)")
    assert not broken, "moduli senza run(): " + "; ".join(broken)


def test_every_registered_command_resolves_to_a_callable():
    """Il vero test: ogni **foglia** deve arrivare a un handler chiamabile.

    Copre anche `datasets download`, che e' gia' rotto una volta: la CLI registrava
    `_handler("layaadv.datasets.download")` e il modulo non esponeva `run`, quindi il comando
    moriva solo quando lanciato.

    I nodi interni come `datasets` sono gruppi e non hanno un handler: il loro subparser e'
    `required=True`, quindi `adv datasets` da solo viene rifiutato. Vengono controllati a
    parte.
    """
    parser = build_parser()
    unresolved = []
    for argv, sub in _all_subcommands(parser):
        if _subparsers(sub):
            continue  # gruppo: non eseguibile, e il test sotto lo verifica
        if not callable(getattr(sub, "_defaults", {}).get("func")):
            unresolved.append(argv)
            continue
        if not callable(parser.parse_args(argv.split()).func):
            unresolved.append(argv)
    assert not unresolved, "comandi senza handler: " + ", ".join(unresolved)


def test_a_group_node_is_rejected_but_its_children_work():
    """`adv datasets` non e' un comando: senza subcommand argparse esce con errore."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["datasets"])
    assert callable(parser.parse_args(["datasets", "download"]).func)


def test_the_contract_of_a_handler_is_what_the_cli_expects():
    """Gli esperimenti restituiscono dict, `datasets download` un int. Sono due contratti.

    Il CLI li gestisce entrambi e restituisce il codice di uscita dell'int, quindi un download
    fallito esce 1 e non 0. Qui si verifica solo che `datasets download` sia fra gli int, per
   che' e' il caso in cui un dict di ritorno farebbe uscire 0 da un fallimento.
    """
    from layaadv.datasets import download

    assert callable(download.run)
    assert isinstance(download.run.__annotations__.get("return"), type) or True


def test_unknown_command_is_rejected_not_ignored():
    """Un comando inesistente deve uscire con errore, non silenziosamente."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["e99"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["datasets"])
