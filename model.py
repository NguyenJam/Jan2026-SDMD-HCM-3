from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from student_tts import generate_student_audio, lookup_pronunciation, read_dictionary, split_words


ROOT = Path(__file__).resolve().parent
DEMO_TEXT = ROOT / "data" / "demo_manifesto.txt"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate speech with the trained phoneme-conditioned audio student."
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--text", help="English words for student speech generation")
    source_group.add_argument("--demo", action="store_true", help="Generate the manifesto with the trained student")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "artifacts" / "train" / "student_model.pt",
        help="Trained student checkpoint",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "demo")
    parser.add_argument("--validate-only", action="store_true", help="Check words against the bundled pronunciations")
    args = parser.parse_args()

    try:
        text = DEMO_TEXT.read_text(encoding="utf-8") if args.demo else (args.text or "Hello world")
        dictionary = read_dictionary()
        words = split_words(text)
        missing = sorted({word for word in words if not lookup_pronunciation(word, dictionary)})
        if missing:
            raise ValueError("No pronunciation in the bundled data for: " + ", ".join(missing))
        if args.validate_only:
            print(f"Words checked: {len(words)}")
            print("All words have bundled pronunciations.")
            return 0

        output_name = "manifesto_student.wav" if args.demo else "student_audio.wav"
        output_path = args.output_dir / output_name
        summary = generate_student_audio(text, args.model, output_path)
        report = {
            **summary,
            "input": "manifesto demo" if args.demo else "provided text",
            "teacher_model": "Fish Audio",
            "generation_model": "trained student model",
            "teacher_is_not_called_at_inference": True,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Student audio written to {output_path}")
        print(f"Run details written to {args.output_dir / 'metrics.json'}")
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"Student audio generation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
