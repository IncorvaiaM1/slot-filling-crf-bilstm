"""Local check of the Viterbi bonus, mirroring the four Gradescope cases.

No training needed. Usage:
    python test_viterbi.py
Part 1 builds a fresh BiLSTMModel (random transitions, no training-time IOB
penalties) and decodes hand-made emissions. Part 2 checks the submitted
checkpoint still decodes only legal IOB2 sequences and matches the saved
predictions.
"""
from __future__ import annotations
import json
from types import SimpleNamespace

import torch

from models.bilstm import BiLSTMModel

TAGS = ["O", "B-X", "I-X"]
BIG = 10.0


def legal(seq: list[str]) -> bool:
    prev = "O"
    for tag in seq:
        if tag not in TAGS and not tag[:2] in ("B-", "I-") and tag != "O":
            return False
        if tag.startswith("I-") and prev[2:] != tag[2:]:
            return False
        prev = tag
    return True


def entities(seq: list[str]) -> int:
    return sum(tag.startswith("B-") for tag in seq)


CASES = [
    ("single-token: I- cannot open a sequence", [[0, 0, BIG]],
     lambda o: legal(o) and o != ["I-X"]),
    ("all-O sequence decodes cleanly", [[BIG, 0, 0]] * 4,
     lambda o: o == ["O"] * 4),
    ("two adjacent same-type spans = 2 entities",
     [[0, BIG, 0], [0, 0, BIG], [0, BIG, 0], [0, 0, BIG]],
     lambda o: legal(o) and entities(o) == 2),
    ("argmax would emit O then I-; Viterbi keeps 1 legal entity",
     [[BIG, 0, 0], [0, 0.6 * BIG, BIG]],
     lambda o: legal(o) and entities(o) == 1),
]

VOCAB_SHAPES = {
    "object, int keys": lambda: SimpleNamespace(id_to_tag=dict(enumerate(TAGS))),
    "dict, str keys": lambda: {"id_to_tag": {str(i): t for i, t in enumerate(TAGS)}},
    "list": lambda: SimpleNamespace(id_to_tag=TAGS),
}


def part1() -> bool:
    print("Part 1: synthetic cases on a fresh (untrained) model")
    all_ok = True
    for shape, make_vocab in VOCAB_SHAPES.items():
        for seed in range(5):
            torch.manual_seed(seed)
            model = BiLSTMModel(vocab_size=10, tagset_size=len(TAGS))
            model.vocab = make_vocab()
            for name, em, check in CASES:
                out = model.viterbi_decode(torch.tensor([em]), torch.tensor([len(em)]))[0]
                if not check(out):
                    all_ok = False
                    print(f"  FAIL [{shape}, seed {seed}] {name}: {out}")
    for name, *_ in CASES:
        print(f"  {'PASS' if all_ok else '----'}  {name}")
    return all_ok


def part2() -> bool:
    print("Part 2: submitted checkpoint")
    import evaluate
    model = evaluate.load_model("bilstm", ".")
    sentences, _ = evaluate.load_eval_data("test")
    preds = model.predict(sentences, model.vocab)
    illegal = sum(not legal(p) for p in preds)
    saved = [json.loads(line)["slots"] for line in open("bilstm_predictions.jsonl")]
    same = sum(a == b for a, b in zip(preds, saved))
    print(f"  sequences with illegal IOB2: {illegal}")
    print(f"  identical to bilstm_predictions.jsonl: {same}/{len(saved)}")
    return illegal == 0 and same == len(saved)


if __name__ == "__main__":
    ok1 = part1()
    ok2 = part2()
    print("\nALL CHECKS PASS" if ok1 and ok2 else "\nSOME CHECKS FAILED")
