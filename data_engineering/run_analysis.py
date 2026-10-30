"""Verify the pinned data and produce reproducible data-engineering evidence."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from cmudict_data import BASE_PHONES, LEGAL_SYMBOLS, Lexicon, VOWELS, split_phone
from fetch_data import ROOT, load_lock, verify_sources


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def analyze(root=ROOT):
    sources = verify_sources(root)
    lock = load_lock(root)
    path = root / "source" / "cmudict-0.7b"
    text = path.read_text(encoding=lock["dictionary_encoding"])
    lexicon = Lexicon.from_text(text)
    phone_rows = [line.split() for line in (root / "source" / "cmudict-0.7b.phones").read_text().splitlines()]
    phone_classes = dict(phone_rows)
    declared_symbols = set((root / "source" / "cmudict-0.7b.symbols").read_text().split())
    if set(phone_classes) != BASE_PHONES or declared_symbols != LEGAL_SYMBOLS:
        raise ValueError("Parser inventory differs from the locked companion tables")
    if {p for p, category in phone_classes.items() if category == "vowel"} != VOWELS:
        raise ValueError("Parser vowel classes differ from the locked phones table")

    phones = Counter(phone for entry in lexicon.entries for phone in entry.phones)
    base_phones = Counter()
    stress = Counter()
    unstressed_vowels = 0
    for symbol, count in phones.items():
        base, mark = split_phone(symbol)
        base_phones[base] += count
        if mark is not None:
            stress[str(mark)] += count
        elif base in VOWELS:
            unstressed_vowels += count
    distribution = Counter(len(entries) for entries in lexicon.by_word.values())
    non_ascii = [entry.raw_label for entry in lexicon.entries if not entry.raw_label.isascii()]
    duplicate_sequences = sum(
        len(entries) - len({entry.phones for entry in entries})
        for entries in lexicon.by_word.values()
    )
    profile = {
        "dataset": lock["dataset"],
        "svn_revision": lock["svn_revision"],
        "encoding": lock["dictionary_encoding"],
        "analyzed_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "counts": {
            "total_lines": len(text.splitlines()),
            "comment_lines": sum(line.strip().startswith(";;;") for line in text.splitlines()),
            "blank_lines": sum(not line.strip() for line in text.splitlines()),
            "pronunciation_entries": len(lexicon.entries),
            "base_spellings": len(lexicon.by_word),
            "words_with_multiple_entries": sum(count for n, count in distribution.items() if n > 1),
            "alternate_entries": sum(entry.variant_id != 0 for entry in lexicon.entries),
            "observed_base_phones": len(base_phones),
            "declared_base_phones": len(phone_classes),
            "observed_phone_symbols": len(phones),
            "declared_phone_symbols": len(declared_symbols),
            "audio_samples": 0,
            "minimum_phones_per_entry": min(len(entry.phones) for entry in lexicon.entries),
            "maximum_phones_per_entry": max(len(entry.phones) for entry in lexicon.entries),
            "maximum_entries_per_word": max(distribution),
            "non_ascii_labels": len(non_ascii),
        },
        "entry_count_distribution": {str(k): distribution[k] for k in sorted(distribution)},
        "stress_occurrences": {k: stress[k] for k in sorted(stress)},
        "quality_checks": {
            "malformed_rows": 0,
            "duplicate_word_variant_keys": 0,
            "unsupported_phone_symbols": sorted(set(phones) - declared_symbols),
            "duplicate_phone_sequences_within_word": duplicate_sequences,
            "vowel_tokens_without_stress_digit": unstressed_vowels,
            "all_source_checksums_verified": True,
            "companion_inventory_matches_parser": True,
        },
        "non_ascii_labels": non_ascii,
        "observed_base_phones": sorted(base_phones),
        "observed_phone_symbols": sorted(phones),
        "note": "Quality counts describe schema checks, not linguistic correctness or speech quality.",
    }
    return lexicon, profile, phone_classes, phones


def check_regressions(lexicon, root=ROOT):
    cases = json.loads((root / "regression_cases.json").read_text(encoding="utf-8"))
    evidence = []
    for case in cases:
        result = lexicon.lookup(case["query"], case.get("variant_id"))
        actual = {
            "status": result["status"],
            "selection_required": result["selection_required"],
            "variant_ids": [entry["variant_id"] for entry in result["pronunciations"]],
            "selected_phones": result["selected"]["phones"] if result["selected"] else None,
        }
        for key, expected in case["expected"].items():
            if actual[key] != expected:
                raise AssertionError(f"{case['id']}: {key}: expected {expected!r}, got {actual[key]!r}")
        evidence.append({"id": case["id"], "passed": True, "expected": case["expected"], "result": result})
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookup", help="Optional single dictionary token")
    parser.add_argument("--variant", type=int, help="Explicit source variant ID; 0 means no suffix")
    args = parser.parse_args()
    try:
        lexicon, profile, phone_classes, phones = analyze()
        evidence = check_regressions(lexicon)
    except (OSError, ValueError, AssertionError) as exc:
        parser.exit(1, f"Analysis failed: {exc}\n")

    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    source_text = (ROOT / "source" / "cmudict-0.7b").read_text(encoding="latin-1")
    header = []
    for line in source_text.splitlines():
        if not line.startswith(";;;"):
            break
        header.append(line)
    (ROOT / "LICENSE.CMUdict.txt").write_text("\n".join(header) + "\n", encoding="utf-8")
    write_json(output / "data_profile.json", profile)
    write_json(output / "lookup_evidence.json", evidence)
    with (output / "phone_inventory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["symbol", "base_phone", "stress", "phone_class", "occurrences"])
        for symbol in sorted(LEGAL_SYMBOLS):
            base, stress = split_phone(symbol)
            writer.writerow([symbol, base, "" if stress is None else stress, phone_classes[base], phones[symbol]])
    lines = [
        "# CMUdict verification results", "",
        f"Dataset: {profile['dataset']}, SVN revision {profile['svn_revision']}", "",
        "| Metric | Verified value |", "|---|---:|",
    ]
    lines += [f"| {name} | {value:,} |" for name, value in profile["counts"].items()]
    lines += ["", f"Lookup regression cases: **{len(evidence)}/{len(evidence)} passed**.", "",
              "All three source checksums and the parser/companion inventories matched.",
              "The full dictionary parsed without rejected rows or duplicate word/variant keys.",
              "These checks do not evaluate synthesized audio.", ""]
    (output / "verification_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(profile["counts"], indent=2))
    print(f"Lookup regression cases: {len(evidence)}/{len(evidence)} passed")
    if args.lookup is not None:
        print(json.dumps(lexicon.lookup(args.lookup, args.variant), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
