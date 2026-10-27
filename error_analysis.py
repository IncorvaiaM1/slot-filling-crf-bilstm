"""Part D helper: dump comparative test-set failures for the report.

Does not write the analysis — it finds CRF-only / BiLSTM-only / shared errors
and prints ten diverse examples (OOV tokens, city slots, span-boundary misses).

Usage (after training the 100% models):
    python error_analysis.py \
        --crf-dir checkpoints/crf_1.00 \
        --bilstm-dir checkpoints/bilstm_1.00
"""

from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import data_loader
import evaluate

N_EXAMPLES = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare CRF vs BiLSTM test errors for Part D."
    )
    parser.add_argument("--crf-dir", type=str, default="checkpoints/crf_1.00")
    parser.add_argument("--bilstm-dir", type=str, default="checkpoints/bilstm_1.00")
    parser.add_argument(
        "--out",
        type=str,
        default="figures/error_examples.json",
        help="JSON dump of selected failures plus summary counts.",
    )
    return parser.parse_args()


def iob_spans(tags: list[str]) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag.startswith("B-"):
            typ = tag[2:]
            j = i + 1
            while j < len(tags) and tags[j] == f"I-{typ}":
                j += 1
            spans.append((typ, i, j))
            i = j
        else:
            i += 1
    return spans


def format_spans(tokens: list[str], tags: list[str]) -> str:
    parts = []
    for typ, start, end in iob_spans(tags):
        filler = " ".join(tokens[start:end])
        parts.append(f"{typ}[{start}:{end}]={filler!r}")
    return "; ".join(parts) if parts else "(none)"


def train_vocab_and_cities() -> tuple[set[str], set[str]]:
    train = data_loader.load_atis()["train"]
    vocab = {token for ex in train for token in ex.tokens}
    cities: set[str] = set()
    for ex in train:
        for typ, start, end in iob_spans(ex.slots):
            if "city_name" in typ:
                cities.add(" ".join(ex.tokens[start:end]).lower())
    return vocab, cities


def bucket(gold: list[str], crf: list[str], lstm: list[str]) -> str:
    crf_ok = crf == gold
    lstm_ok = lstm == gold
    if crf_ok and lstm_ok:
        return "both_correct"
    if (not crf_ok) and lstm_ok:
        return "crf_wrong_only"
    if crf_ok and (not lstm_ok):
        return "bilstm_wrong_only"
    return "both_wrong"


def interestingness(
    tokens: list[str],
    gold: list[str],
    crf: list[str],
    lstm: list[str],
    train_vocab: set[str],
    train_cities: set[str],
) -> tuple[int, dict[str, Any]]:
    """Higher score => better Part D example. Also returns notes for the dump."""
    oov = [tok for tok in tokens if tok not in train_vocab]
    gold_spans = iob_spans(gold)
    crf_spans = set(iob_spans(crf))
    lstm_spans = set(iob_spans(lstm))
    gold_set = set(gold_spans)

    city_unseen = []
    for typ, start, end in gold_spans:
        if "city_name" not in typ:
            continue
        filler = " ".join(tokens[start:end]).lower()
        if filler not in train_cities:
            city_unseen.append(filler)

    boundary = False
    for typ, start, end in gold_spans:
        type_hits_crf = [s for s in crf_spans if s[0] == typ]
        type_hits_lstm = [s for s in lstm_spans if s[0] == typ]
        crf_type_only = any(s[0] == typ and s[1:] != (start, end) for s in type_hits_crf)
        lstm_type_only = any(s[0] == typ and s[1:] != (start, end) for s in type_hits_lstm)
        if crf_type_only or lstm_type_only:
            boundary = True

    score = 0
    score += 3 * len(oov)
    score += 4 * len(city_unseen)
    score += 2 if boundary else 0
    score += 1 if gold_set.symmetric_difference(crf_spans) else 0
    score += 1 if gold_set.symmetric_difference(lstm_spans) else 0
    # Prefer shorter utterances so they fit in a 3-page report.
    score += max(0, 8 - len(tokens) // 4)

    notes = {
        "oov_tokens": oov,
        "unseen_city_fillers": city_unseen,
        "span_boundary_error": boundary,
        "gold_spans": format_spans(tokens, gold),
        "crf_spans": format_spans(tokens, crf),
        "bilstm_spans": format_spans(tokens, lstm),
    }
    return score, notes


def pick_ten(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        if rec["bucket"] == "both_correct":
            continue
        by_bucket[rec["bucket"]].append(rec)

    for bucket_name in by_bucket:
        by_bucket[bucket_name].sort(key=lambda r: r["score"], reverse=True)

    quota = {"crf_wrong_only": 4, "bilstm_wrong_only": 4, "both_wrong": 2}
    chosen: list[dict[str, Any]] = []
    used: set[int] = set()
    for bucket_name, n in quota.items():
        for rec in by_bucket[bucket_name][:n]:
            chosen.append(rec)
            used.add(rec["index"])

    # Fill leftover slots from the global interestingness ranking.
    leftovers = [r for r in records if r["bucket"] != "both_correct" and r["index"] not in used]
    leftovers.sort(key=lambda r: r["score"], reverse=True)
    for rec in leftovers:
        if len(chosen) >= N_EXAMPLES:
            break
        chosen.append(rec)
    return chosen[:N_EXAMPLES]


def main() -> None:
    args = parse_args()
    sentences, gold = evaluate.load_eval_data("test")

    crf_model = evaluate.load_model("crf", args.crf_dir)
    lstm_model = evaluate.load_model("bilstm", args.bilstm_dir)
    crf_pred = crf_model.predict(sentences, getattr(crf_model, "vocab", None))
    lstm_pred = lstm_model.predict(sentences, getattr(lstm_model, "vocab", None))

    train_vocab, train_cities = train_vocab_and_cities()
    counts: Counter[str] = Counter()
    records: list[dict[str, Any]] = []

    for index, (tokens, y, c, b) in enumerate(zip(sentences, gold, crf_pred, lstm_pred)):
        kind = bucket(y, c, b)
        counts[kind] += 1
        score, notes = interestingness(tokens, y, c, b, train_vocab, train_cities)
        records.append(
            {
                "index": index,
                "bucket": kind,
                "score": score,
                "tokens": tokens,
                "gold": y,
                "crf": c,
                "bilstm": b,
                **notes,
            }
        )

    crf_metrics = evaluate.compute_metrics(crf_pred, gold)
    lstm_metrics = evaluate.compute_metrics(lstm_pred, gold)
    selected = pick_ten(records)

    print("Test-set comparison (sequence exact-match)")
    n = len(sentences)
    for key in ("both_correct", "crf_wrong_only", "bilstm_wrong_only", "both_wrong"):
        print(f"  {key:14s}: {counts[key]:4d}  ({counts[key] / n:.1%})")
    print(f"CRF    token F1={crf_metrics['token_f1']:.4f}  span F1={crf_metrics['span_f1']:.4f}")
    print(f"BiLSTM token F1={lstm_metrics['token_f1']:.4f}  span F1={lstm_metrics['span_f1']:.4f}")
    print()
    print(f"Ten report candidates (prefer CRF-wrong-only / BiLSTM-wrong-only / both-wrong):")
    for rec in selected:
        print("-" * 72)
        print(f"[{rec['index']}] bucket={rec['bucket']}  score={rec['score']}")
        print("  utt :", " ".join(rec["tokens"]))
        print("  gold:", rec["gold_spans"])
        print("  crf :", rec["crf_spans"])
        print("  lstm:", rec["bilstm_spans"])
        if rec["oov_tokens"]:
            print("  oov :", rec["oov_tokens"])
        if rec["unseen_city_fillers"]:
            print("  unseen city:", rec["unseen_city_fillers"])
        if rec["span_boundary_error"]:
            print("  note: type-right / boundary-wrong span")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_test": n,
        "exact_match_counts": dict(counts),
        "crf_metrics": crf_metrics,
        "bilstm_metrics": lstm_metrics,
        "examples": selected,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
