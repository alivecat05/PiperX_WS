"""EgoHand annotation, native smoothing, context-crop recovery and epose export."""

import argparse
import importlib
import sys

COMMANDS = {
    "label": (
        "humanlabel.workflows.pipeline",
        "EgoHand + native smoothing + context recovery + camera epose export",
    ),
    "export": (
        "humanlabel.export",
        "Re-export existing geometry as PiperCapture-format camera wrist eposes",
    ),
    "prepare-sources": (
        "humanlabel.utils.prepare_sources",
        "Clone pinned EgoHand upstream sources and apply compatibility patches",
    ),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in COMMANDS:
        module = importlib.import_module(COMMANDS[argv[0]][0])
        return module.main(argv[1:])
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Use python -m humanlabel COMMAND --help for workflow options.",
    )
    parser.add_argument(
        "command",
        choices=COMMANDS,
        help="; ".join(f"{k}: {v[1]}" for k, v in COMMANDS.items()),
    )
    parser.parse_args(argv)
