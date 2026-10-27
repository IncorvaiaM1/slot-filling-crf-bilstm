"""Kick off the Part C training runs (you choose when to launch this).

Default: the eight headline runs (CRF + BiLSTM-CRF, four fractions, seed 42),
written to checkpoints/{model}_{fraction}/ -- the layout plot_curve.py and
prepare_submission.py read.

Extra runs for the report (multiple seeds, softmax-only BiLSTM) go to
checkpoints/extra/ so they never overwrite the submitted checkpoints:
    checkpoints/extra/bilstm_{fraction}_s{seed}/          BiLSTM-CRF
    checkpoints/extra/bilstm-softmax_{fraction}_s{seed}/  BiLSTM + softmax

Usage (the BiLSTM recipe used for all reported runs is
--constant-lr --max-epochs 300 --patience 20; see README):
    python train_all.py --models crf
    python train_all.py --models bilstm --constant-lr --max-epochs 300 --patience 20
    python train_all.py --models bilstm --extra --seeds 1 2 --constant-lr ...
    python train_all.py --models bilstm --extra --no-crf --seeds 42 1 2 --constant-lr ...
"""

from __future__ import annotations
import argparse
import subprocess
import sys

FRACTIONS = (0.05, 0.10, 0.25, 1.00)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CRF/BiLSTM on Part C fractions.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["crf", "bilstm"],
        default=["crf", "bilstm"],
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=list(FRACTIONS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--no-crf",
        action="store_true",
        help="BiLSTM: train the plain softmax tagger (implies --extra).",
    )
    parser.add_argument(
        "--extra",
        action="store_true",
        help="Write to checkpoints/extra/<name>_s<seed> instead of the headline dirs.",
    )
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="BiLSTM: passed through to train.py.")
    parser.add_argument("--patience", type=int, default=None,
                        help="BiLSTM: passed through to train.py.")
    parser.add_argument("--constant-lr", action="store_true",
                        help="BiLSTM: passed through to train.py.")
    return parser.parse_args()


def fraction_tag(fraction: float) -> str:
    return f"{fraction:.2f}"


def run_name(model: str, tag: str, seed: int, no_crf: bool, extra: bool) -> str:
    if not extra:
        return f"{model}_{tag}"
    variant = "bilstm-softmax" if (model == "bilstm" and no_crf) else model
    return f"extra/{variant}_{tag}_s{seed}"


def main() -> None:
    args = parse_args()
    extra = args.extra or args.no_crf or args.seeds != [42]
    for model in args.models:
        for seed in args.seeds:
            for fraction in args.fractions:
                tag = fraction_tag(fraction)
                name = run_name(model, tag, seed, args.no_crf, extra)
                cmd = [
                    sys.executable,
                    "train.py",
                    "--model",
                    model,
                    "--train-fraction",
                    tag,
                    "--checkpoint-dir",
                    f"checkpoints/{name}",
                    "--seed",
                    str(seed),
                ]
                if model == "bilstm":
                    cmd.extend(["--tensorboard-logdir", f"runs/{name}"])
                    if args.no_crf:
                        cmd.append("--no-crf")
                    if args.max_epochs:
                        cmd.extend(["--max-epochs", str(args.max_epochs)])
                    if args.constant_lr:
                        cmd.append("--constant-lr")
                    if args.patience:
                        cmd.extend(["--patience", str(args.patience)])
                print(" ".join(cmd), flush=True)
                subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
