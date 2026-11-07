"""Public command dispatch; optional GPU models load only for their workflow."""

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
    "egohand": (
        "humanlabel.workflows.egohand",
        "Run only the native EgoHand inference stage",
    ),
    "recover": (
        "humanlabel.workflows.recover",
        "Recover missed crops from existing EgoHand results",
    ),
    "compare": (
        "humanlabel.visualization.runner",
        "Run optional baselines and produce full comparison videos",
    ),
    "compare-videos": (
        "humanlabel.visualization.compare",
        "Validate and compare already-completed model videos",
    ),
    "temporal": (
        "humanlabel.visualization.temporal",
        "Run the HaWoR, ACE or WiLoR comparison backend",
    ),
    "refine": (
        "humanlabel.visualization.refine",
        "Fit an existing MANO sequence on CUDA",
    ),
    "render": (
        "humanlabel.visualization.render",
        "Re-render saved predictions with the matching camera estimates",
    ),
    "prepare-sources": (
        "humanlabel.utils.prepare_sources",
        "Clone pinned optional upstream sources and apply compatibility patches",
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
