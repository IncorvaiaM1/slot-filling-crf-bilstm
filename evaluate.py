"""Course-provided evaluation script for A1a (Slot Filling on ATIS).

Model loading and data loading are wired up for you. The one thing left
to implement is `compute_metrics()` — scoring predictions against gold
labels via `scorer.py`. Look for the `# TODO: implement` block below.

Note: `load_model()` is used directly by Gradescope for reproducibility check —
loading a submitted checkpoint and re-running inference.
`load_eval_data()` remains local-dev-only — autograder's prediction-file scoring path —
`crf_predictions.jsonl` / `bilstm_predictions.jsonl` doesn't use it, since it operates on
already-generated predictions, not the underlying data split directly.

Usage:
    python evaluate.py --model crf    --checkpoint-dir checkpoints/crf_1.00    --split dev
    python evaluate.py --model bilstm --checkpoint-dir checkpoints/bilstm_1.00 --split test
"""

from __future__ import annotations
import argparse
import os
from types import SimpleNamespace
from typing import Any, Literal


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for an evaluation run."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CRF or BiLSTM slot-filling model on ATIS."
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["crf", "bilstm"],
        required=True,
        help="Which model type to evaluate.",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory containing the trained model to load.",
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=["dev", "test"],
        default="dev",
        help="Which split to evaluate on (default: dev).",
    )

    parser.add_argument(
        "--predictions-out",
        type=str,
        default=None,
        help="Optional path to write JSONL predictions (index/tokens/slots).",
    )

    return parser.parse_args()


def load_model(model_type: str, checkpoint_dir: str) -> Any:
    """Construct a model and load its trained weights/state from `checkpoint_dir`.

    Checkpoint format contract (what `train_loop()` in `train.py` must produce):
      - "crf":    `<checkpoint_dir>/crf_model.pkl`, a joblib-dumped
                  `sklearn_crfsuite.CRF` estimator.

      - "bilstm": `<checkpoint_dir>/bilstm_model.pt`, a torch-saved dict with
                  keys `state_dict`, `vocab_size`, `tagset_size`, and
                  `vocab` (itself a dict with `token_to_id` / `id_to_tag`,
                  matching the mapping `BiLSTMModel.predict()` expects).
    """
    model: Any

    if model_type == "crf":
        # --- Integration point: `models/crf.py` ------------------------------
        # Same dispatch pattern as `train.py`'s `build_model()`.
        from models.crf import CRFModel

        import joblib

        model = CRFModel()
        model.model = joblib.load(os.path.join(checkpoint_dir, "crf_model.pkl"))
        return model
        # ---------------------------------------------------------------------

    if model_type == "bilstm":
        # --- Integration point: `models/bilstm.py` ----------------------------
        # Same dispatch pattern as `train.py`'s `build_model()`.
        from models.bilstm import BiLSTMModel

        import torch

        checkpoint = torch.load(
            os.path.join(checkpoint_dir, "bilstm_model.pt"), map_location="cpu"
        )

        model = BiLSTMModel(
            vocab_size=checkpoint["vocab_size"],
            tagset_size=checkpoint["tagset_size"],
            # Absent in older checkpoints, which were all BiLSTM-CRF.
            use_crf=checkpoint.get("use_crf", True),
        )

        model.load_state_dict(checkpoint["state_dict"])
        model.vocab = SimpleNamespace(**checkpoint["vocab"])
        return model
        # ---------------------------------------------------------------------

    raise ValueError(f"Unknown model type: {model_type!r}")


def load_eval_data(
    split: Literal["dev", "test"],
) -> tuple[list[list[str]], list[list[str]]]:
    """Load tokenized sentences and gold BIO labels for the requested `split`.

    Returns:
        (`sentences`, `gold_labels`)
    """
    import data_loader

    examples = data_loader.load_split(split)
    sentences = [example.tokens for example in examples]
    gold_labels = [example.slots for example in examples]
    return sentences, gold_labels


def compute_metrics(
    predictions: list[list[str]], gold: list[list[str]]
) -> dict[str, float]:
    """Score predicted BIO tag sequences against gold BIO tag sequences.

    This is the part you implement.

    Args:
        predictions: Predicted BIO label sequences, one per sentence.
        gold: Gold BIO label sequences, one per sentence, aligned
            sentence-for-sentence (and token-for-token within each
            sentence) with `predictions`.

    Returns:
        A dict with at least the keys `"token_f1"` and `"span_f1"`.
    """
    import scorer

    return {
        "token_f1": float(scorer.token_scores(gold, predictions)["f1"]),
        "span_f1": float(scorer.span_scores(gold, predictions)["f1"]),
    }


def main() -> None:
    args = parse_args()

    model = load_model(args.model, args.checkpoint_dir)
    sentences, gold_labels = load_eval_data(args.split)

    # `CRFModel.predict()` ignores `vocab`; `BiLSTMModel.predict()` uses it.
    # `getattr()` covers `CRFModel` instances, which never set a `.vocab` attribute.
    vocab = getattr(model, "vocab", None)
    predictions = model.predict(sentences, vocab)

    metrics = compute_metrics(predictions, gold_labels)

    print(f"Evaluation results — model={args.model}, split={args.split}")
    print(f"  Token-level slot F1: {metrics['token_f1']:.4f}")
    print(f"  Span-level slot F1:  {metrics['span_f1']:.4f}")

    if args.predictions_out:
        write_predictions(args.predictions_out, sentences, predictions)
        print(f"  Wrote predictions:   {args.predictions_out}")


def write_predictions(
    path: str, sentences: list[list[str]], slots: list[list[str]]
) -> None:
    """Write test-set predictions in the Gradescope JSONL format."""
    import json

    with open(path, "w", encoding="utf-8") as handle:
        for index, (tokens, pred_slots) in enumerate(zip(sentences, slots)):
            handle.write(
                json.dumps({"index": index, "tokens": tokens, "slots": pred_slots})
                + "\n"
            )


if __name__ == "__main__":
    main()
