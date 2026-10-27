"""Part A feature log: dev F1 for each CRF feature set.

Trains the CRF on 100% of train with nested / leave-one-out subsets of the
feature template in `models/crf.py`, and scores each on the dev split (test F1
is recorded too, for reference only; model choices were made on dev).

Feature groups (keys produced by `CRFModel.extract_features()`):
  identity : word, word.lower                 (current token, or context token)
  affix    : word[:1..3], word[-1..-3:]
  ortho    : is*/len/shape/short_shape/has.*
  gaz      : gaz.* lexicon hits, looks.time regex, city/airline phrase pairs
  ctx1     : the groups above applied to the -1/+1 tokens, plus bigrams
  ctx2     : lower-cased word at -2/+2

Also records, on dev, how many predicted spans have exactly the gold boundaries
but the wrong label ("wrong_type"), and how many of those differ only in the
role prefix, e.g. toloc.city_name vs fromloc.city_name ("role_only").

Usage:
    python crf_ablation.py                       # all configs, ~30-60 min on CPU
    python crf_ablation.py --configs identity full
    python crf_ablation.py --fraction 0.05       # same configs on the 5% subset
Writes figures/crf_feature_ablation.json (or ..._0.05.json) and prints a table.
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Any

import data_loader
import scorer
from models.crf import CRFModel

ALL_GROUPS = {"identity", "affix", "ortho", "gaz", "ctx1", "ctx2"}

CONFIGS: dict[str, set[str]] = {
    # Cumulative build-up.
    "identity": {"identity"},
    "+affix": {"identity", "affix"},
    "+ortho": {"identity", "affix", "ortho"},
    "+ctx1": {"identity", "affix", "ortho", "ctx1"},
    "+ctx2": {"identity", "affix", "ortho", "ctx1", "ctx2"},
    "full": set(ALL_GROUPS),  # = +gazetteers, the submitted feature set
    # Leave-one-out from the full set.
    "full-gaz": ALL_GROUPS - {"gaz"},
    "full-affix": ALL_GROUPS - {"affix"},
    "full-ortho": ALL_GROUPS - {"ortho"},
    "full-ctx": ALL_GROUPS - {"ctx1", "ctx2"},
}


def key_groups(key: str) -> set[str]:
    """Groups a feature key needs enabled in order to be kept."""
    if key in ("bias", "BOS", "EOS"):
        return set()
    needed: set[str] = set()
    base = key
    for prefix, ctx in (("-1:", "ctx1"), ("+1:", "ctx1"), ("-2:", "ctx2"), ("+2:", "ctx2")):
        if key.startswith(prefix):
            needed.add(ctx)
            base = key[len(prefix):]
            break
    if base in ("BOS", "EOS"):
        return needed
    if base in ("word", "word.lower", "bigram"):
        needed.add("identity")
    elif base.startswith("word[") :
        needed.add("affix")
    elif base.startswith("gaz.") or base == "looks.time":
        needed.add("gaz")
    else:  # word.is*, len, shape, short_shape, has.*
        needed.add("ortho")
    return needed


def _spans(tags: list[str]) -> list[tuple[str, int, int]]:
    out, i = [], 0
    while i < len(tags):
        if tags[i].startswith("B-"):
            typ, j = tags[i][2:], i + 1
            while j < len(tags) and tags[j] == f"I-{typ}":
                j += 1
            out.append((typ, i, j))
            i = j
        else:
            i += 1
    return out


def role_confusions(gold: list[list[str]], pred: list[list[str]]) -> dict[str, int]:
    """Right boundaries, wrong label; and the subset that differs only in role."""
    wrong_type = role_only = 0
    for g, p in zip(gold, pred):
        by_span = {(a, b): t for t, a, b in _spans(p)}
        for typ, a, b in _spans(g):
            other = by_span.get((a, b))
            if other is None or other == typ:
                continue
            wrong_type += 1
            if typ.split(".")[-1] == other.split(".")[-1] and typ.split(".")[0] != other.split(".")[0]:
                role_only += 1
    return {"dev_wrong_type": wrong_type, "dev_role_only": role_only}


class AblatedCRF(CRFModel):
    def __init__(self, groups: set[str]) -> None:
        super().__init__()
        self.groups = groups

    def extract_features(self, sentence: list[str]) -> list[dict[str, Any]]:
        full = super().extract_features(sentence)
        return [
            {k: v for k, v in feats.items() if key_groups(k) <= self.groups}
            for feats in full
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Train on the fixed Part C subsample of this size.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.out is None:
        suffix = "" if args.fraction == 1.0 else f"_{args.fraction:.2f}"
        args.out = f"figures/crf_feature_ablation{suffix}.json"

    splits = data_loader.load_atis()
    train, dev, test = splits["train"], splits["dev"], splits["test"]
    train = data_loader.subsample(train, fraction=args.fraction)

    out_path = Path(args.out)
    results: dict[str, Any] = {}
    if out_path.is_file():
        results = json.loads(out_path.read_text(encoding="utf-8"))

    for name in args.configs:
        groups = CONFIGS[name]
        model = AblatedCRF(groups)
        start = time.time()
        model.fit([e.tokens for e in train], [e.slots for e in train])
        n_feats = len(model.model.state_features_)
        row: dict[str, Any] = {"groups": sorted(groups), "n_state_features": n_feats}
        for split_name, data in (("dev", dev), ("test", test)):
            pred = model.predict([e.tokens for e in data])
            gold = [e.slots for e in data]
            row[f"{split_name}_span_f1"] = float(scorer.span_scores(gold, pred)["f1"])
            row[f"{split_name}_token_f1"] = float(scorer.token_scores(gold, pred)["f1"])
            if split_name == "dev":
                row.update(role_confusions(gold, pred))
        row["n_train"] = len(train)
        row["seconds"] = round(time.time() - start, 1)
        results[name] = row
        print(f"{name:12s} dev span F1={row['dev_span_f1']:.4f}  "
              f"test span F1={row['test_span_f1']:.4f}  feats={n_feats}  ({row['seconds']}s)",
              flush=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n| Feature set | Groups | Dev span F1 | Test span F1 |")
    print("|---|---|---|---|")
    for name in CONFIGS:
        if name in results:
            r = results[name]
            print(f"| {name} | {', '.join(r['groups'])} | {r['dev_span_f1']:.4f} | {r['test_span_f1']:.4f} |")


if __name__ == "__main__":
    main()
