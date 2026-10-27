"""Course-provided skeleton for training Part A (CRF) and Part B (BiLSTM) models.

In this training script, CLI parsing, data loading, and model construction are wired up for you.
The only one thing left to implement is `train_loop()` — actual training loop,
for both Part A CRF and Part B BiLSTM. Look for the `# TODO: implement` block below.

Usage:
    python train.py --model crf    --train-fraction 1.00 --checkpoint-dir checkpoints/crf_1.00
    python train.py --model bilstm --train-fraction 0.25 --checkpoint-dir checkpoints/bilstm_0.25 \
        --tensorboard-logdir runs/bilstm_0.25
"""

from __future__ import annotations
import argparse
import random
from collections import Counter
from typing import Any
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a training run."""
    parser = argparse.ArgumentParser(
        description="Train a CRF or BiLSTM slot-filling model on ATIS."
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["crf", "bilstm"],
        required=True,
        help="Which model to train.",
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        required=True,
        help="Fraction of training set to use, e.g. 0.05 / 0.10 / 0.25 / 1.00.",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory to save the trained model to.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    parser.add_argument(
        "--tensorboard-logdir",
        type=str,
        default=None,
        help="TensorBoard log directory. Only used when --model BiLSTM; "
        "ignored for --model crf (CRF training doesn't need TensorBoard or a GPU).",
    )

    parser.add_argument(
        "--no-crf",
        action="store_true",
        help="BiLSTM only: train the plain softmax tagger (cross-entropy + argmax) "
        "instead of the default BiLSTM-CRF. Used for the Part C ablation curve.",
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="BiLSTM only: override the epoch budget (default: 80 if <500 "
        "training sentences, else 40).",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="BiLSTM only: override early-stopping patience (default: 10 if <500 "
        "training sentences, else 7).",
    )

    parser.add_argument(
        "--constant-lr",
        action="store_true",
        help="BiLSTM only: disable the ReduceLROnPlateau schedule (keep lr=1e-3).",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch — CPU + CUDA — for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(fraction: float, seed: int) -> tuple[Any, Any, Any, Any]:
    """Load the ATIS train/dev/test splits, subsampling the train split.

    Returns:
        (`train_data`, `dev_data`, `test_data`, `vocab`)

        `train_data`/`dev_data`/`test_data` are `list[data_loader.Example]`.
        `vocab` is the dict returned by `build_vocab()` (see below),
        augmented with `tagset_size` (via `data_loader.slot_label_set()`) —
        needed to construct `BiLSTMModel` in `build_model()`.
    """
    import data_loader

    all_splits = data_loader.load_atis()
    train_data = data_loader.subsample(
        all_splits["train"], fraction=fraction, seed=seed
    )

    dev_data = all_splits["dev"]
    test_data = all_splits["test"]

    train_sentences = [example.tokens for example in train_data]
    vocab = build_vocab(train_sentences)
    vocab["tagset_size"] = len(data_loader.slot_label_set(train_data))

    return train_data, dev_data, test_data, vocab


# Built here rather than in data_loader.py to avoid touching a
# teammate-owned file; if data_loader.py later grows its own vocab
# utility, this should be removed in favor of that to avoid two
# divergent implementations.
def build_vocab(train_sentences: list[list[str]]) -> dict[str, Any]:
    """Build a token vocabulary from tokenized training sentences.

    Fully implemented — this is infrastructure, not the student-fill part
    of the assignment.

    Reserves index 0 for "<PAD>" and index 1 for "<UNK>". Tokens are then
    ordered by descending frequency, ties broken alphabetically. This
    determinism is required: `token_to_id` must be identical across
    machines/Python versions/runs, or it risks breaking the ±0.3 F1
    reproducibility tolerance in the grading rubric. Do NOT rely on dict or
    set iteration order for this — the explicit `sorted()` key below is
    what makes it deterministic.

    Args:
        train_sentences: Tokenized training sentences.

    Returns:
        A dict with `token_to_id` (Dict[str, int]), `id_to_token`
        (Dict[int, str]), and `vocab_size` (int).
    """
    counts = Counter(token for sentence in train_sentences for token in sentence)
    ordered_tokens = sorted(counts, key=lambda token: (-counts[token], token))

    token_to_id = {"<PAD>": 0, "<UNK>": 1}
    for token in ordered_tokens:
        token_to_id[token] = len(token_to_id)
    id_to_token = {index: token for token, index in token_to_id.items()}

    return {
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "vocab_size": len(token_to_id),
    }


def build_model(
    model_type: str,
    vocab_size: int | None = None,
    tagset_size: int | None = None,
    use_crf: bool = True,
) -> Any:
    """Construct an untrained model instance for the given model type.

    `vocab_size`/`tagset_size` (from `load_data()`'s `vocab`) are required
    for `model_type == "bilstm"` and ignored for `model_type == "crf"` —
    `CRFModel` works on raw token features, not a fixed vocabulary, so
    there's nothing for it to size itself against. This is intentional,
    not a bug, same as `CRFModel.predict()`'s unused `vocab` parameter.
    """
    if model_type == "crf":
        from models.crf import CRFModel

        return CRFModel()

    if model_type == "bilstm":
        from models.bilstm import BiLSTMModel

        if vocab_size is None or tagset_size is None:
            raise ValueError(
                "vocab_size and tagset_size are required for model_type='bilstm'"
            )

        return BiLSTMModel(
            vocab_size=vocab_size, tagset_size=tagset_size, use_crf=use_crf
        )

    raise ValueError(f"Unknown model type: {model_type!r}")


def train_loop(
    model: Any,
    train_data: Any,
    dev_data: Any,
    args: argparse.Namespace,
) -> float:
    """Train `model` on `train_data`, validating against `dev_data`.

    This is the part you implement.

    Your implementation must:
      - Train `model` on `train_data`.

      - Use `dev_data` for validation (and early stopping, if you choose to implement it).

      - When `args.model == "BiLSTM"`, log training progress to
        TensorBoard via `args.tensorboard_logdir` (e.g. using
        `torch.utils.tensorboard.SummaryWriter`). Not applicable for `args.model == "crf"`.

      - Save the final trained model to `args.checkpoint_dir`. See
        `evaluate.py`'s `load_model()` docstring for the exact checkpoint
        format your saved model must match.

    Returns:
        Final dev-set slot F1 (float). `main()` will print what you return.
    """
    import os
    import joblib
    import data_loader
    import scorer
    from types import SimpleNamespace
    from torch.utils.data import DataLoader, Dataset

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_sentences = [example.tokens for example in train_data]
    train_labels = [example.slots for example in train_data]
    dev_sentences = [example.tokens for example in dev_data]
    dev_labels = [example.slots for example in dev_data]

    if args.model == "crf":
        print(f"Training CRF on {len(train_data)} utterances...")
        model.fit(train_sentences, train_labels)

        predictions = model.predict(dev_sentences)
        dev_f1 = float(scorer.span_scores(dev_labels, predictions)["f1"])
        print(f"Dev span F1: {dev_f1:.4f}")

        checkpoint_path = os.path.join(args.checkpoint_dir, "crf_model.pkl")
        joblib.dump(model.model, checkpoint_path)
        print(f"Saved CRF checkpoint to {checkpoint_path}")
        return dev_f1

    # --- BiLSTM -----------------------------------------------------------
    token_vocab = build_vocab(train_sentences)
    tag_names = sorted(data_loader.slot_label_set(train_data))
    tag_to_id = {tag: index for index, tag in enumerate(tag_names)}
    id_to_tag = {index: tag for tag, index in tag_to_id.items()}

    vocab_ns = SimpleNamespace(
        token_to_id=token_vocab["token_to_id"],
        id_to_tag=id_to_tag,
    )
    model.vocab = vocab_ns

    if getattr(model, "use_crf", False):
        model.apply_iob_constraints(id_to_tag)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training BiLSTM on {len(train_data)} utterances ({device})...")

    class _SlotDataset(Dataset):
        def __init__(self, examples: list) -> None:
            self.examples = examples
            self.unk_id = token_vocab["token_to_id"]["<UNK>"]

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
            example = self.examples[index]
            token_ids = [
                token_vocab["token_to_id"].get(token, self.unk_id)
                for token in example.tokens
            ]
            tag_ids = [tag_to_id[tag] for tag in example.slots]
            return token_ids, tag_ids

    def _collate(
        batch: list[tuple[list[int], list[int]]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lengths = torch.tensor([len(tokens) for tokens, _ in batch], dtype=torch.long)
        max_len = int(lengths.max().item())
        token_pad = torch.zeros(len(batch), max_len, dtype=torch.long)
        # Pad tags with 0 so CRF indexing is always valid; loss is masked by lengths.
        tag_pad = torch.zeros(len(batch), max_len, dtype=torch.long)
        for row, (tokens, tags) in enumerate(batch):
            length = len(tokens)
            token_pad[row, :length] = torch.tensor(tokens, dtype=torch.long)
            tag_pad[row, :length] = torch.tensor(tags, dtype=torch.long)
        return token_pad, tag_pad, lengths

    batch_size = 32
    # Small fractions see far fewer updates per epoch, so give them a longer
    # budget; full-data runs still early-stop (they already did around epoch 34).
    n_train = len(train_data)
    epochs = 80 if n_train < 500 else 40
    patience = 10 if n_train < 500 else 7
    if getattr(args, "max_epochs", None):
        epochs = args.max_epochs
    if getattr(args, "patience", None):
        patience = args.patience
    loader = DataLoader(
        _SlotDataset(train_data),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    writer = None
    if args.tensorboard_logdir:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=args.tensorboard_logdir)

    def _dev_span_f1() -> float:
        import warnings
        from sklearn.exceptions import UndefinedMetricWarning

        model.eval()
        with torch.no_grad():
            predictions = model.predict(dev_sentences, vocab_ns)
        # Epoch-1 models often predict all-O; seqeval then warns about 0 precision.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            return float(scorer.span_scores(dev_labels, predictions)["f1"])

    best_f1 = -1.0
    epochs_without_improvement = 0
    checkpoint_path = os.path.join(args.checkpoint_dir, "bilstm_model.pt")
    use_crf = getattr(model, "use_crf", False)

    def _save() -> None:
        torch.save(
            {
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "vocab_size": token_vocab["vocab_size"],
                "tagset_size": len(tag_to_id),
                "use_crf": bool(use_crf),
                "vocab": {
                    "token_to_id": token_vocab["token_to_id"],
                    "id_to_tag": id_to_tag,
                },
            },
            checkpoint_path,
        )

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for token_ids, tag_ids, lengths in loader:
            token_ids = token_ids.to(device)
            tag_ids = tag_ids.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            emissions = model(token_ids, lengths)
            if use_crf:
                loss = model.crf_nll(emissions, tag_ids, lengths)
            else:
                mask = (
                    torch.arange(token_ids.size(1), device=device).unsqueeze(0)
                    < lengths.unsqueeze(1)
                )
                loss = torch.nn.functional.cross_entropy(
                    emissions[mask], tag_ids[mask]
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)
        dev_f1 = _dev_span_f1()
        if not getattr(args, "constant_lr", False):
            scheduler.step(dev_f1)

        if writer is not None:
            writer.add_scalar("train/loss", avg_loss, epoch)
            writer.add_scalar("dev/span_f1", dev_f1, epoch)

        print(f"Epoch {epoch:02d}/{epochs}  loss={avg_loss:.4f}  dev_span_f1={dev_f1:.4f}")

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            epochs_without_improvement = 0
            _save()
            print(f"  new best checkpoint -> {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch} epochs (best dev F1={best_f1:.4f})")
                break

    if writer is not None:
        writer.close()

    if best_f1 < 0:
        _save()
        best_f1 = _dev_span_f1()

    print(f"Saved BiLSTM checkpoint to {checkpoint_path}")
    return best_f1


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_data, dev_data, test_data, vocab = load_data(args.train_fraction, args.seed)

    if args.model == "bilstm":
        model = build_model(
            args.model,
            vocab_size=vocab["vocab_size"],
            tagset_size=vocab["tagset_size"],
            use_crf=not args.no_crf,
        )

    else:
        model = build_model(args.model)

    dev_score = train_loop(model, train_data, dev_data, args)

    import os

    if not os.path.isdir(args.checkpoint_dir) or not os.listdir(args.checkpoint_dir):
        raise RuntimeError(
            f"Expected train_loop() to save a checkpoint to {args.checkpoint_dir!r}, "
            "but the directory is missing or empty."
        )

    print("Training complete.")
    print(f"  Model:          {args.model}")
    print(f"  Train fraction: {args.train_fraction}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")

    if dev_score is not None:
        print(f"  Final dev F1:   {dev_score:.4f}")


if __name__ == "__main__":
    main()
