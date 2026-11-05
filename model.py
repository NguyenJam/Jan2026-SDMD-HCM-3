from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
import wave
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DICTIONARY = ROOT / "data" / "cmudict-0.7b"
FISH_API_URL = "https://api.fish.audio/v1/tts"
FISH_MODEL = "s2.1-pro-free"
FISH_API_KEY_DEFAULT = "sk-fish-1Zl7lVw8AA-mY2Sxs6N6oL8WRqZTc3vhSY9-wfNGzvI"

ARPABET_PHONES = frozenset({
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T",
    "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
})
VOWELS = frozenset({
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
    "IH", "IY", "OW", "OY", "UH", "UW",
})


def read_dictionary(path: Path = DICTIONARY) -> dict[str, list[list[str]]]:
    entries: dict[str, list[list[str]]] = defaultdict(list)
    with path.open(encoding="latin-1") as dictionary_file:
        for line in dictionary_file:
            if not line or line.startswith(";;;"):
                continue
            fields = line.strip().split()
            if len(fields) < 2:
                continue
            raw_word, *phones = fields
            word = re.sub(r"\(\d+\)$", "", raw_word).upper()
            entries[word].append(phones)
    return dict(entries)


def split_text(text: str) -> list[str]:
    normalized = text.replace("’", "'")
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", normalized)


def validate_phones(phones: list[str]) -> None:
    for phone in phones:
        match = re.fullmatch(r"([A-Z]+)([0-2])?", phone)
        if not match:
            raise ValueError(f"Unexpected dictionary phone: {phone}")
        symbol, stress_text = match.groups()
        if symbol not in ARPABET_PHONES:
            raise ValueError(f"Unknown CMU Arpabet phone: {symbol}")
        if symbol in VOWELS and stress_text is None:
            raise ValueError(f"Dictionary vowel {phone} has no stress marker")
        if symbol not in VOWELS and stress_text is not None:
            raise ValueError(f"Dictionary consonant {phone} has a stress marker")


def lookup_words(
    text: str,
    dictionary: dict[str, list[list[str]]],
    variant: int,
) -> list[dict]:
    words = split_text(text)
    if not words:
        raise ValueError("Enter at least one English word.")

    rendered_words = []
    for token in words:
        pronunciations = dictionary.get(token.upper())
        if not pronunciations:
            raise ValueError(f"'{token}' is not in the bundled pronunciation data.")
        if variant > len(pronunciations):
            raise ValueError(
                f"'{token}' has {len(pronunciations)} pronunciation(s); "
                f"variant {variant} was requested."
            )
        phones = pronunciations[variant - 1]
        validate_phones(phones)
        rendered_words.append({
            "word": token,
            "variant": variant,
            "source_phones": phones,
        })
    return rendered_words


def render_fish_text(
    text: str,
    rendered_words: list[dict],
) -> str:
    normalized = text.replace("’", "'")
    word_iter = iter(rendered_words)
    pattern = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

    def phoneme_tag(_: re.Match[str]) -> str:
        item = next(word_iter)
        phones = " ".join(item["source_phones"])
        return f"<|phoneme_start|>{phones}<|phoneme_end|>"

    return pattern.sub(phoneme_tag, normalized)


def dictionary_sha256(path: Path = DICTIONARY) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as dictionary_file:
        for block in iter(lambda: dictionary_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_dictionary(dictionary: dict[str, list[list[str]]]) -> tuple[int, int]:
    pronunciation_count = 0
    for pronunciations in dictionary.values():
        for phones in pronunciations:
            validate_phones(phones)
            pronunciation_count += 1
    return len(dictionary), pronunciation_count


def make_audio(
    text: str,
    output_dir: Path,
    variant: int,
    dictionary: dict[str, list[list[str]]],
) -> Path:
    words = lookup_words(text, dictionary, variant)
    fish_text = render_fish_text(text, words)
    api_key = os.environ.get("FISH_API_KEY") or FISH_API_KEY_DEFAULT
    payload = json.dumps({
        "text": fish_text,
        "format": "wav",
        "normalize": False,
        "temperature": 0.7,
        "top_p": 0.7,
        "latency": "normal",
    }).encode("utf-8")
    request = urllib.request.Request(
        FISH_API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "audio/wav",
            "model": FISH_MODEL,
        },
        method="POST",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / "audio_preview.wav"
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            wav_path.write_bytes(response.read())
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Fish Audio returned HTTP {error.code}: {details[:500]}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach Fish Audio: {error.reason}") from error

    with wave.open(str(wav_path), "rb") as wav_file:
        audio_info = {
            "channels": wav_file.getnchannels(),
            "sample_rate_hz": wav_file.getframerate(),
            "frames": wav_file.getnframes(),
            "duration_seconds": round(wav_file.getnframes() / wav_file.getframerate(), 3),
        }
    if audio_info["frames"] == 0:
        raise RuntimeError("Fish Audio returned an empty WAV file.")

    report = [
        "Pronunciation-to-audio demo",
        f"Input text: {text}",
        f"Pronunciation data: CMUdict 0.7b (SHA-256 {dictionary_sha256()})",
        "Voice model: Fish Audio",
        "Role: teacher model and final speech generator",
        "",
        "Pronunciations:",
    ]
    for item in words:
        report.append(f"  {item['word']}: {' '.join(item['source_phones'])}")
    report.extend([
        "",
        "WAV output:",
        f"  Channels: {audio_info['channels']}",
        f"  Sample rate: {audio_info['sample_rate_hz']} Hz",
        f"  Duration: {audio_info['duration_seconds']} seconds",
        f"  File: {wav_path.name}",
    ])
    (output_dir / "metrics.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    return wav_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Look up English pronunciations and generate speech with Fish Audio."
    )
    parser.add_argument("--text", default="Hello world", help="English words to pronounce")
    parser.add_argument("--variant", type=int, default=1, help="One-based pronunciation choice for each word")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts", help="Directory for WAV and metrics")
    parser.add_argument("--validate-only", action="store_true", help="Check the dictionary without generating audio")
    args = parser.parse_args()

    if args.variant < 1:
        parser.error("--variant must be 1 or greater")
    try:
        dictionary = read_dictionary()
        word_count, pronunciation_count = validate_dictionary(dictionary)
        if args.validate_only:
            lookup_words(args.text, dictionary, args.variant)
            print(f"Pronunciation entries loaded: {word_count:,} words, {pronunciation_count:,} pronunciations.")
            print("Dictionary phones and the sample input are valid.")
            return 0

        wav_path = make_audio(args.text, args.output_dir, args.variant, dictionary)
        print(f"Audio written to {wav_path}")
        print(f"Metrics written to {args.output_dir / 'metrics.txt'}")
        return 0
    except (OSError, RuntimeError, ValueError, wave.Error) as error:
        print(f"Audio generation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
