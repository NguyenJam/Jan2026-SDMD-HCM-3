from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BINS = ("1-4", "5-8", "9-12", "13+")
DRIFT_THRESHOLD = 0.20
MIN_TOKEN_ACCURACY = 0.80
MIN_EXACT_ACCURACY = 0.35


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def histogram(values: list[int]) -> dict[str, int]:
    counts = {key: 0 for key in BINS}
    for value in values:
        key = "1-4" if value <= 4 else "5-8" if value <= 8 else "9-12" if value <= 12 else "13+"
        counts[key] += 1
    return counts


def psi(reference: dict[str, int], current: dict[str, int]) -> float | None:
    reference_total = sum(reference.get(key, 0) for key in BINS)
    current_total = sum(current.get(key, 0) for key in BINS)
    if not reference_total or not current_total:
        return None
    value = 0.0
    for key in BINS:
        expected = (reference.get(key, 0) + 0.5) / (reference_total + 0.5 * len(BINS))
        observed = (current.get(key, 0) + 0.5) / (current_total + 0.5 * len(BINS))
        value += (observed - expected) * math.log(observed / expected)
    return round(value, 4)


def incoming_features(data_path: Path) -> tuple[int, dict[str, int], dict[str, int]]:
    word_lengths: list[int] = []
    phone_counts: list[int] = []
    if data_path.exists():
        with data_path.open("r", newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                word = (row.get("word") or "").strip()
                phones = (row.get("pronunciation") or "").split()
                if word and phones:
                    word_lengths.append(len(word))
                    phone_counts.append(len(phones))
    return len(word_lengths), histogram(word_lengths), histogram(phone_counts)


def write_preflight(data_quality_path: Path, update_path: Path, previous_run_path: Path,
                    report_path: Path, github_output: Path | None) -> dict:
    quality = read_json(data_quality_path)
    previous = read_json(previous_run_path)
    row_count, word_histogram, phone_histogram = incoming_features(update_path)
    old_word_hist = previous.get("baseline_word_length_histogram", {})
    old_phone_hist = previous.get("baseline_phone_count_histogram", {})
    word_psi = psi(old_word_hist, word_histogram)
    phone_psi = psi(old_phone_hist, phone_histogram)
    has_baseline = bool(old_word_hist and old_phone_hist)
    drift_detected = (
        not has_baseline
        or (word_psi is not None and word_psi >= DRIFT_THRESHOLD)
        or (phone_psi is not None and phone_psi >= DRIFT_THRESHOLD)
    )
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "pre-retraining",
        "incoming_rows": row_count,
        "data_quality_status": quality.get("status", "unavailable"),
        "distribution_drift": {
            "word_length_psi": word_psi,
            "phone_count_psi": phone_psi,
            "threshold": DRIFT_THRESHOLD,
            "detected": drift_detected,
        },
        "decision": "retrain" if drift_detected else "monitor_only",
        "reason": "No previous model baseline is available." if not has_baseline else None,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if github_output:
        github_output.parent.mkdir(parents=True, exist_ok=True)
        with github_output.open("a", encoding="utf-8") as stream:
            stream.write(f"drift_detected={str(drift_detected).lower()}\n")
            stream.write(f"incoming_rows={row_count}\n")
    print(json.dumps(report, indent=2))
    return report


def write_post_report(data_quality_path: Path, training_metrics_path: Path,
                      report_path: Path, alerts_path: Path) -> dict:
    quality = read_json(data_quality_path)
    metrics = read_json(training_metrics_path)
    word_psi = psi(metrics.get("baseline_word_length_histogram", {}),
                   metrics.get("incoming_word_length_histogram", {}))
    phone_psi = psi(metrics.get("baseline_phone_count_histogram", {}),
                    metrics.get("incoming_phone_count_histogram", {}))
    exact_accuracy = metrics.get("validation_exact_pronunciation_accuracy_sample")
    token_accuracy = metrics.get("validation_phone_token_accuracy")
    findings = []
    if quality.get("status") != "pass":
        findings.append("Incoming data quality checks did not pass.")
    if (word_psi is not None and word_psi >= DRIFT_THRESHOLD) or (
        phone_psi is not None and phone_psi >= DRIFT_THRESHOLD
    ):
        findings.append("The incoming pronunciation distribution differs from the baseline.")
    if token_accuracy is not None and token_accuracy < MIN_TOKEN_ACCURACY:
        findings.append("Validation phone-token accuracy is below the monitoring threshold.")
    if exact_accuracy is not None and exact_accuracy < MIN_EXACT_ACCURACY:
        findings.append("Validation exact-pronunciation accuracy is below the monitoring threshold.")

    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": metrics.get("model_version", "not recorded"),
        "data_version": metrics.get("data_version", "not recorded"),
        "data_quality": {
            "status": quality.get("status", "unavailable"),
            "accepted_rows": quality.get("rows_accepted", 0),
            "rejected_rows": quality.get("rows_rejected", 0),
            "sha256": quality.get("source_sha256"),
        },
        "validation": {
            "phone_token_accuracy": token_accuracy,
            "minimum_phone_token_accuracy": MIN_TOKEN_ACCURACY,
            "exact_pronunciation_accuracy": exact_accuracy,
            "minimum_exact_pronunciation_accuracy": MIN_EXACT_ACCURACY,
        },
        "drift": {
            "word_length_psi": word_psi,
            "phone_count_psi": phone_psi,
            "threshold": DRIFT_THRESHOLD,
            "detected": any(value is not None and value >= DRIFT_THRESHOLD for value in (word_psi, phone_psi)),
        },
        "status": "alert" if findings else "within_thresholds",
        "findings": findings,
        "recommended_action": "Review the input data and validation report before the next model promotion." if findings else "Continue scheduled monitoring.",
    }
    alerts = {
        "checked_at_utc": report["checked_at_utc"],
        "count": len(findings),
        "items": findings,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    alerts_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    alerts_path.write_text(json.dumps(alerts, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Check data drift and model quality for the retraining pipeline.")
    parser.add_argument("--mode", choices=("preflight", "post-train"), required=True)
    parser.add_argument("--data-quality", type=Path, default=ROOT / "data" / "prepared" / "data_quality.json")
    parser.add_argument("--update-data", type=Path, default=ROOT / "data" / "prepared" / "new_data.csv")
    parser.add_argument("--previous-run", type=Path, default=ROOT / "artifacts" / "run.json")
    parser.add_argument("--training-metrics", type=Path, default=ROOT / "artifacts" / "run.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "monitoring")
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "preflight":
            write_preflight(args.data_quality, args.update_data, args.previous_run,
                            args.output_dir / "reports" / "preflight.json", args.github_output)
        else:
            write_post_report(args.data_quality, args.training_metrics,
                              args.output_dir / "reports" / "monitoring.json",
                              args.output_dir / "alerts" / "alerts.json")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Monitoring failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
