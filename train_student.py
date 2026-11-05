from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from student_tts import train_student


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the small phoneme-to-audio student model from Fish teacher speech."
    )
    parser.add_argument("--epochs", type=int, default=12, help="Number of training passes")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/train"))
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    try:
        summary = train_student(args.output_dir, epochs=args.epochs)
        text = "\n".join([
            "Student audio model training",
            f"  Teacher: {summary['teacher_model']}",
            f"  Aligned words: {summary['training_words']}",
            f"  Training frames: {summary['training_frames']}",
            f"  Validation frames: {summary['validation_frames']}",
            f"  Epochs: {summary['epochs']}",
            f"  Final training L1: {summary['final_train_l1']}",
            f"  Final validation L1: {summary['final_validation_l1']}",
            f"  Scope: {summary['scope']}",
        ]) + "\n"
        (args.output_dir / "metrics.txt").write_text(text, encoding="utf-8")
        print(text, end="")
        print(f"Training details: {args.output_dir / 'metrics.json'}")
        print(f"Model checkpoint: {args.output_dir / summary['checkpoint']}")
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"Student training failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
