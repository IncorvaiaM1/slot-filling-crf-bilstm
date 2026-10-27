"""Pack the Gradescope coding zip from your 100% checkpoints.

Copies checkpoints to the zip-root names, writes test-set prediction JSONL,
and builds `submission.zip` with files at the archive root (no extra folder).

Usage:
    python prepare_submission.py \
        --crf-dir checkpoints/crf_1.00 \
        --bilstm-dir checkpoints/bilstm_1.00
"""

from __future__ import annotations
import argparse
import shutil
import zipfile
from pathlib import Path

import evaluate

ARCHIVE_MEMBERS = (
    "models/__init__.py",
    "models/crf.py",
    "models/bilstm.py",
    "crf_predictions.jsonl",
    "bilstm_predictions.jsonl",
    "crf_model.pkl",
    "bilstm_model.pt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the A1a coding zip.")
    parser.add_argument("--crf-dir", type=str, default="checkpoints/crf_1.00")
    parser.add_argument("--bilstm-dir", type=str, default="checkpoints/bilstm_1.00")
    parser.add_argument("--out-zip", type=str, default="submission.zip")
    return parser.parse_args()


def dump_predictions(model_type: str, checkpoint_dir: str, out_path: Path) -> None:
    model = evaluate.load_model(model_type, checkpoint_dir)
    sentences, gold = evaluate.load_eval_data("test")
    vocab = getattr(model, "vocab", None)
    predictions = model.predict(sentences, vocab)
    metrics = evaluate.compute_metrics(predictions, gold)
    evaluate.write_predictions(str(out_path), sentences, predictions)
    print(
        f"{model_type:6s} test token F1={metrics['token_f1']:.4f}  "
        f"span F1={metrics['span_f1']:.4f}  -> {out_path}"
    )


def main() -> None:
    args = parse_args()
    crf_src = Path(args.crf_dir) / "crf_model.pkl"
    lstm_src = Path(args.bilstm_dir) / "bilstm_model.pt"
    if not crf_src.is_file():
        raise FileNotFoundError(f"Missing CRF checkpoint: {crf_src}")
    if not lstm_src.is_file():
        raise FileNotFoundError(f"Missing BiLSTM checkpoint: {lstm_src}")

    shutil.copy2(crf_src, "crf_model.pkl")
    shutil.copy2(lstm_src, "bilstm_model.pt")
    print(f"Copied {crf_src} -> crf_model.pkl")
    print(f"Copied {lstm_src} -> bilstm_model.pt")

    dump_predictions("crf", args.crf_dir, Path("crf_predictions.jsonl"))
    dump_predictions("bilstm", args.bilstm_dir, Path("bilstm_predictions.jsonl"))

    out_zip = Path(args.out_zip)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member in ARCHIVE_MEMBERS:
            path = Path(member)
            if not path.is_file():
                raise FileNotFoundError(f"Refusing to zip; missing {member}")
            archive.write(path, arcname=member.replace("\\", "/"))

    print(f"Wrote {out_zip} with {len(ARCHIVE_MEMBERS)} entries at archive root:")
    with zipfile.ZipFile(out_zip) as archive:
        for info in archive.infolist():
            print(f"  {info.filename}")


if __name__ == "__main__":
    main()
