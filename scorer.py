"""seqeval-based slot-F1 scorer, both token-level and span-level.

Span-level (a.k.a. entity-level) F1 is the standard metric for slot filling:
a predicted slot only counts as correct if its full B-I-I... span matches the
gold span exactly (right type, right boundaries). This is what seqeval's
precision_score/recall_score/f1_score compute by default, and it is what
Gradescope's autograder uses for the headline number.

Token-level F1 is a looser, diagnostic metric: it scores each token's tag
independently, ignoring span boundaries, so partial-span predictions get
partial credit. Useful for Part D error analysis (e.g. "CRF gets the right
type but the wrong boundary").
"""

from __future__ import annotations
import warnings
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from seqeval.scheme import IOB2
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import f1_score as sk_f1_score
from sklearn.metrics import precision_score as sk_precision_score
from sklearn.metrics import recall_score as sk_recall_score

SeqOfSeq = list[list[str]]


def span_scores(y_true: SeqOfSeq, y_pred: SeqOfSeq) -> dict[str, float]:
    """Entity/span-level precision, recall, F1 (seqeval, strict IOB2 scheme)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        return {
            "precision": precision_score(y_true, y_pred, mode="strict", scheme=IOB2),
            "recall": recall_score(y_true, y_pred, mode="strict", scheme=IOB2),
            "f1": f1_score(y_true, y_pred, mode="strict", scheme=IOB2),
        }


def token_scores(y_true: SeqOfSeq, y_pred: SeqOfSeq) -> dict[str, float]:
    """Per-token precision/recall/F1, flattened across all sequences.

    "O" tags are excluded from the label set so that the score reflects only
    how well slot tokens are tagged (matching the usual convention; including
    O would inflate the score since most tokens are O).
    """
    flat_true = [tag for seq in y_true for tag in seq]
    flat_pred = [tag for seq in y_pred for tag in seq]
    labels = sorted({t for t in flat_true if t != "O"})

    return {
        "precision": sk_precision_score(
            flat_true, flat_pred, labels=labels, average="micro", zero_division=0
        ),
        "recall": sk_recall_score(
            flat_true, flat_pred, labels=labels, average="micro", zero_division=0
        ),
        "f1": sk_f1_score(
            flat_true, flat_pred, labels=labels, average="micro", zero_division=0
        ),
    }


def full_report(y_true: SeqOfSeq, y_pred: SeqOfSeq) -> str:
    """Per-slot-type span-level precision/recall/F1 breakdown (seqeval)."""
    return classification_report(y_true, y_pred, mode="strict", scheme=IOB2, digits=4)


if __name__ == "__main__":
    y_true = [["O", "B-fromloc.city_name", "O", "B-toloc.city_name"]]
    y_pred = [["O", "B-fromloc.city_name", "O", "I-toloc.city_name"]]

    print("span :", span_scores(y_true, y_pred))
    print("token:", token_scores(y_true, y_pred))
    print(full_report(y_true, y_pred))
