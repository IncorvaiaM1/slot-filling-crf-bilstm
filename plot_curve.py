"""Part C: slot F1 vs. training-set size for CRF and BiLSTM on one plot.

Scores the headline checkpoints (checkpoints/{crf,bilstm}_{fraction}/) and,
with --extras, the extra runs in checkpoints/extra/ written by
`train_all.py --extra` (more BiLSTM-CRF seeds, and the softmax-only BiLSTM).

The x-axis is the number of training sentences on a log scale (the four
fractions are not evenly spaced), and the y-axis is zoomed; its range is
printed on the axis. With --extras, BiLSTM curves show the mean over seeds
with a min-max band, and each seed as a faint dot. The CRF (L-BFGS) training
is deterministic, so it has no seed variance.

Usage (after training):
    python plot_curve.py --split test
    python plot_curve.py --split test --extras
    python plot_curve.py --from-json figures/data_efficiency.json
"""

from __future__ import annotations
import argparse
import json
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

import evaluate

FRACTIONS = (0.05, 0.10, 0.25, 1.00)
N_SENTENCES = {"0.05": 224, "0.10": 448, "0.25": 1120, "1.00": 4478}

# Categorical slots 1-3 of the reference palette (validated for CVD).
COLORS = {"crf": "#2a78d6", "bilstm": "#eb6834", "bilstm-softmax": "#1baf7a"}
LABELS = {
    "crf": "CRF (hand features)",
    "bilstm": "BiLSTM-CRF (Part B model)",
    "bilstm-softmax": "BiLSTM + softmax",
}
MARKERS = {"crf": "o", "bilstm": "s", "bilstm-softmax": "^"}
LINESTYLES = {"crf": "-", "bilstm": "-", "bilstm-softmax": "--"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def fraction_tag(fraction: float) -> str:
    return f"{fraction:.2f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Part C data-efficiency curves from trained checkpoints."
    )
    parser.add_argument("--checkpoint-root", type=str, default="checkpoints")
    parser.add_argument("--split", type=str, choices=["dev", "test"], default="test")
    parser.add_argument(
        "--out",
        type=str,
        default="figures/data_efficiency.png",
        help="Output image path (PNG + PDF). A JSON sidecar is written next to it.",
    )
    parser.add_argument(
        "--extras",
        action="store_true",
        help="Also score checkpoints/extra/* (extra seeds, softmax-only BiLSTM).",
    )
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument(
        "--from-json",
        type=str,
        default=None,
        help="Replot from an existing metrics JSON instead of re-scoring checkpoints.",
    )
    return parser.parse_args()


def score_checkpoint(model: str, ckpt_dir: Path, split: str) -> dict[str, float]:
    loaded = evaluate.load_model(model, str(ckpt_dir))
    sentences, gold = evaluate.load_eval_data(split)
    vocab = getattr(loaded, "vocab", None)
    predictions = loaded.predict(sentences, vocab)
    return evaluate.compute_metrics(predictions, gold)


def collect_curves(root: Path, split: str, skip_missing: bool) -> dict:
    """Headline runs: {model: {fraction_tag: {token_f1, span_f1}}}."""
    curves: dict = {"crf": {}, "bilstm": {}}
    missing: list[str] = []
    for model in ("crf", "bilstm"):
        for fraction in FRACTIONS:
            tag = fraction_tag(fraction)
            ckpt_dir = root / f"{model}_{tag}"
            expected = "crf_model.pkl" if model == "crf" else "bilstm_model.pt"
            if not (ckpt_dir / expected).is_file():
                missing.append(str(ckpt_dir / expected))
                continue
            print(f"Scoring {model} fraction={tag} on {split}...")
            curves[model][tag] = score_checkpoint(model, ckpt_dir, split)
            print(f"  span F1 = {curves[model][tag]['span_f1']:.4f}")
    if missing and not skip_missing:
        raise FileNotFoundError(
            "Missing checkpoints (train these fractions first):\n  " + "\n  ".join(missing)
        )
    return curves


def collect_extras(root: Path, split: str) -> dict:
    """Extra runs: {variant: {fraction_tag: {seed: span_f1}}}."""
    extras: dict = {}
    pattern = re.compile(r"^(bilstm|bilstm-softmax)_(\d\.\d\d)_s(\d+)$")
    for ckpt_dir in sorted((root / "extra").glob("*")):
        match = pattern.match(ckpt_dir.name)
        if not match or not (ckpt_dir / "bilstm_model.pt").is_file():
            continue
        variant, tag, seed = match.groups()
        print(f"Scoring extra {ckpt_dir.name} on {split}...")
        f1 = score_checkpoint("bilstm", ckpt_dir, split)["span_f1"]
        extras.setdefault(variant, {}).setdefault(tag, {})[seed] = f1
    return extras


def seed_table(curves: dict, extras: dict) -> dict:
    """{variant: {tag: [f1 per seed]}}; the headline BiLSTM run is seed 42."""
    table: dict = {"bilstm": {}, "bilstm-softmax": {}}
    for tag, metrics in curves.get("bilstm", {}).items():
        table["bilstm"].setdefault(tag, {})["42"] = metrics["span_f1"]
    for variant, by_tag in extras.items():
        for tag, by_seed in by_tag.items():
            table[variant].setdefault(tag, {}).update(by_seed)
    return table


def plot_curves(curves: dict, extras: dict, split: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    xs_all = [N_SENTENCES[fraction_tag(f)] for f in FRACTIONS]

    # CRF: deterministic, one point per fraction.
    tags = [t for t in N_SENTENCES if t in curves.get("crf", {})]
    ax.plot(
        [N_SENTENCES[t] for t in tags],
        [curves["crf"][t]["span_f1"] for t in tags],
        color=COLORS["crf"], marker=MARKERS["crf"], markersize=6,
        linewidth=2, linestyle=LINESTYLES["crf"], label=LABELS["crf"],
    )

    table = seed_table(curves, extras)
    for variant in ("bilstm", "bilstm-softmax"):
        by_tag = table[variant]
        tags = [t for t in N_SENTENCES if t in by_tag]
        if not tags:
            continue
        xs = [N_SENTENCES[t] for t in tags]
        vals = [list(by_tag[t].values()) for t in tags]
        counts = [len(v) for v in vals]
        n_seeds = max(counts)
        means = [statistics.mean(v) for v in vals]
        label = LABELS[variant]
        if n_seeds > 1:
            label += f", mean of {n_seeds} seeds"
            singles = [f"{int(float(t) * 100)}%" for t, c in zip(tags, counts) if c == 1]
            if singles:
                label += f" ({', '.join(singles)}: 1 seed)"
        if n_seeds > 1:
            ax.fill_between(
                xs, [min(v) for v in vals], [max(v) for v in vals],
                color=COLORS[variant], alpha=0.15, linewidth=0,
            )
            for x, v in zip(xs, vals):
                ax.scatter([x] * len(v), v, s=10, color=COLORS[variant], alpha=0.45,
                           linewidths=0, zorder=2)
        ax.plot(
            xs, means, color=COLORS[variant], marker=MARKERS[variant], markersize=6,
            linewidth=2, linestyle=LINESTYLES[variant], label=label, zorder=3,
        )

    ax.set_xscale("log")
    ax.set_xticks(xs_all)
    ax.set_xticklabels([f"{n}\n({int(float(t) * 100)}%)" for t, n in N_SENTENCES.items()])
    ax.minorticks_off()
    ax.set_xlabel("Training sentences (log scale)", color=MUTED)
    ax.set_ylabel(f"{split.capitalize()} span F1 (y-axis starts at 0.75)", color=MUTED)
    ax.set_ylim(0.75, 1.0)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)

    if args.from_json:
        payload = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
        split = payload.get("split", args.split)
        curves = payload["metrics"]
        extras = payload.get("extras", {})
    else:
        split = args.split
        root = Path(args.checkpoint_root)
        curves = collect_curves(root, split, args.skip_missing)
        extras = collect_extras(root, split) if args.extras else {}
        payload = {
            "split": split,
            "fractions": [fraction_tag(f) for f in FRACTIONS],
            "n_sentences": N_SENTENCES,
            "metrics": curves,
            "extras": extras,
        }
        sidecar = out_path.with_suffix(".json")
        sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {sidecar}")

    plot_curves(curves, extras, split, out_path)
    print(f"Wrote {out_path}")

    table = seed_table(curves, extras)
    print("\nspan F1 table (BiLSTM columns: mean [min-max] over seeds)")
    print(f"{'fraction':>8} {'n':>5}  {'CRF':>7}  {'BiLSTM-CRF':>24}  {'BiLSTM+softmax':>24}")
    for fraction in FRACTIONS:
        tag = fraction_tag(fraction)
        crf = curves.get("crf", {}).get(tag, {}).get("span_f1")
        cells = []
        for variant in ("bilstm", "bilstm-softmax"):
            v = list(table[variant].get(tag, {}).values())
            if not v:
                cells.append("n/a")
            elif len(v) == 1:
                cells.append(f"{v[0]:.4f}")
            else:
                cells.append(f"{statistics.mean(v):.4f} [{min(v):.4f}-{max(v):.4f}] n={len(v)}")
        crf_s = f"{crf:.4f}" if crf is not None else "n/a"
        print(f"{tag:>8} {N_SENTENCES[tag]:>5}  {crf_s:>7}  {cells[0]:>24}  {cells[1]:>24}")


if __name__ == "__main__":
    main()
