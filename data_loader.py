"""Loads the ATIS slot-filling split into the format both the CRF (Part A)
and the BiLSTM (Part B) expect.

Expected on-disk layout (already populated under data/atis/):

    data/atis/{train,valid,test}/seq.in    whitespace-tokenized utterance, one per line
    data/atis/{train,valid,test}/seq.out   IOB2 slot tags, one per line, aligned to seq.in
    data/atis/{train,valid,test}/label     intent label, one per line

This is the standard ATIS release used across the slot-filling literature
(Goo et al. 2018 / MiuLab SlotGated-SLU): 4,478 train / 500 dev / 893 test,
120 slot labels, 21 intents.
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


DATA_DIR = Path(__file__).parent / "data" / "atis"
Split = Literal["train", "dev", "test"]
SPLIT_DIRS: dict[Split, str] = {"train": "train", "dev": "valid", "test": "test"}


@dataclass
class Example:
    tokens: list[str]
    slots: list[str]
    intent: str


def load_split(split: Split, data_dir: Path = DATA_DIR) -> list[Example]:
    """Reads one split (train/dev/test) into a list of Examples."""
    split_dir = data_dir / SPLIT_DIRS[split]
    tokens_lines = (split_dir / "seq.in").read_text(encoding="utf-8").splitlines()
    slots_lines = (split_dir / "seq.out").read_text(encoding="utf-8").splitlines()
    intent_lines = (split_dir / "label").read_text(encoding="utf-8").splitlines()

    if not (len(tokens_lines) == len(slots_lines) == len(intent_lines)):
        raise ValueError(
            f"{split}: seq.in ({len(tokens_lines)}), seq.out ({len(slots_lines)}), "
            f"label ({len(intent_lines)}) line counts disagree"
        )

    examples = []

    for tok_line, slot_line, intent in zip(tokens_lines, slots_lines, intent_lines):
        tokens = tok_line.split()
        slots = slot_line.split()

        if len(tokens) != len(slots):
            raise ValueError(
                f"{split}: token/slot length mismatch: {tok_line!r} vs {slot_line!r}"
            )

        examples.append(Example(tokens=tokens, slots=slots, intent=intent))

    return examples


def load_atis(data_dir: Path = DATA_DIR) -> dict[Split, list[Example]]:
    """Loads all three splits. Returns {"train": [...], "dev": [...], "test": [...]}."""
    return {split: load_split(split, data_dir) for split in SPLIT_DIRS}


def slot_label_set(examples: list[Example]) -> set[str]:
    return {tag for ex in examples for tag in ex.slots}


def intent_label_set(examples: list[Example]) -> set[str]:
    return {ex.intent for ex in examples}


# Not student-configurable: every student's Part-C subsamples must be
# identical, or data-efficiency curves aren't comparable across the class.
PART_C_SUBSAMPLE_SEED = 42


def subsample(
    examples: list[Example], fraction: float, seed: int = 42
) -> list[Example]:
    """Deterministic, fixed-seed subsample of the training set for the Part C
    data-efficiency curve (fraction in {0.05, 0.10, 0.25, 1.0}).

    Sampling without replacement, shuffled once per seed so that the 5% subset
    is contained in the 10% subset, etc. (a common, easy-to-defend choice —
    note this in your report if you rely on nesting).

    `seed` is intentionally NOT used for the subsampling itself — it always
    uses `PART_C_SUBSAMPLE_SEED`, so every student's data-efficiency curve is
    computed over identical subsamples; model training randomness (`--seed`
    in `train.py`) is unaffected and still student-controllable. The
    parameter is kept for other, non-Part-C callers that may need a caller-chosen seed.
    """
    if not 0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")

    n = round(len(examples) * fraction)
    rng = random.Random(PART_C_SUBSAMPLE_SEED)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    return shuffled[:n]


if __name__ == "__main__":
    data = load_atis()
    for split, examples in data.items():
        print(f"{split:5s}: {len(examples):5d} examples")

    print(f"slot labels: {len(slot_label_set(data['train']))}")
    print(f"intents:     {len(intent_label_set(data['train']))}")
    print(data["train"][0])
