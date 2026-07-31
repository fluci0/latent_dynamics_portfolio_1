import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import ExperimentConfig
from .experiments import EXPERIMENTS, run_many


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train pendulum latent-dynamics portfolio models.")
    parser.add_argument("experiment", choices=(*EXPERIMENTS, "all"), help="Model family to train")
    parser.add_argument("--epochs", type=int, default=None, help="Override the 700-epoch maximum")
    parser.add_argument("--patience", type=int, default=None, help="Override early-stopping patience")
    parser.add_argument("--output-dir", type=Path, default=None, help="Artifact directory")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig()
    changes = {key: value for key, value in {
        "epochs": args.epochs,
        "patience": args.patience,
        "output_dir": args.output_dir,
    }.items() if value is not None}
    if changes:
        config = replace(config, **changes)
    names = EXPERIMENTS if args.experiment == "all" else (args.experiment,)
    print(json.dumps(run_many(names, config), indent=2))


if __name__ == "__main__":
    main()

