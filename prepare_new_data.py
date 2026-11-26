from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PHONE_BASES = {
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER",
    "EY", "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG",
    "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W",
    "Y", "Z", "ZH",
}


def prepare(input_path: Path, output_path: Path, report_path: Path) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(f"Update file not found: {input_path}")

    accepted: dict[str, str] = {}
    rejected: list[dict[str, str | int]] = []
    with input_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"word", "pronunciation"}.issubset(reader.fieldnames):
            raise ValueError("Input CSV needs word and pronunciation columns.")
        for line_number, row in enumerate(reader, start=2):
            word = (row.get("word") or "").strip().lower()
            phones = " ".join((row.get("pronunciation") or "").upper().split())
            phone_tokens = phones.split()
            valid_phones = bool(phone_tokens) and all(
                re.fullmatch(r"[A-Z]+[012]?", token)
                and (token[:-1] if token[-1:].isdigit() else token) in PHONE_BASES
                for token in phone_tokens
            )
            if not re.fullmatch(r"[a-z']+", word) or len(word) > 24 or not valid_phones:
                rejected.append({"line": line_number, "word": word, "reason": "invalid word or ARPAbet sequence"})
                continue
            if len(phone_tokens) > 38:
                rejected.append({"line": line_number, "word": word, "reason": "pronunciation exceeds model length limit"})
                continue
            if word in accepted:
                rejected.append({"line": line_number, "word": word, "reason": "duplicate word"})
                continue
            accepted[word] = phones

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["word", "pronunciation"])
        writer.writerows(sorted(accepted.items()))

    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": input_path.name,
        "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "rows_accepted": len(accepted),
        "rows_rejected": len(rejected),
        "unique_words": len(accepted),
        "output_file": str(output_path.relative_to(ROOT)),
        "rejected_rows": rejected,
        "status": "pass" if accepted and not rejected else "fail",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if rejected:
        raise ValueError(f"Rejected {len(rejected)} update rows; resolve them before retraining.")
    if not accepted:
        raise ValueError("The update file contains no valid pronunciation rows.")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate incoming word-to-pronunciation data.")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "new_data.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "prepared" / "new_data.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "data" / "prepared" / "data_quality.json")
    args = parser.parse_args()
    try:
        prepare(args.input, args.output, args.report)
        return 0
    except (OSError, ValueError) as error:
        print(f"Data preparation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
