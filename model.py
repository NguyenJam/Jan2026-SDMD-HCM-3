"""Create a short WAV preview from dictionary pronunciations."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import os
import re
import shutil
import subprocess
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
FISH_FREE_THROUGH = date(2026, 11, 30)

# eSpeak NG's English phoneme names are not the same spelling as ARPAbet.
PHONE_MAP = {
    "AA": "A:", "AE": "a", "AH": "V", "AO": "O:",
    "AW": "aU", "AY": "aI", "EH": "E", "ER": "3",
    "EY": "eI", "IH": "I", "IY": "i:", "OW": "oU",
    "OY": "OI", "UH": "U", "UW": "u:",
    "B": "b", "CH": "tS", "D": "d", "DH": "D", "F": "f",
    "G": "g", "HH": "h", "JH": "dZ", "K": "k", "L": "l",
    "M": "m", "N": "n", "NG": "N", "P": "p", "R": "r",
    "S": "s", "SH": "S", "T": "t", "TH": "T", "V": "v",
    "W": "w", "Y": "j", "Z": "z", "ZH": "Z",
}
VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
    "IH", "IY", "OW", "OY", "UH", "UW",
}
STRESS_MARK = {0: "%", 1: "'", 2: ","}


def read_dictionary(path: Path = DICTIONARY) -> dict[str, list[list[str]]]:
    """Read pronunciations and keep alternate entries in order."""
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
    """Keep words and contractions; punctuation acts as a boundary."""
    normalized = text.replace("’", "'")
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", normalized)


def map_pronunciation(phones: list[str]) -> list[str]:
    """Convert one ARPAbet pronunciation to eSpeak NG phoneme symbols."""
    mapped = []
    for phone in phones:
        match = re.fullmatch(r"([A-Z]+)([0-2])?", phone)
        if not match:
            raise ValueError(f"Unexpected dictionary phone: {phone}")
        symbol, stress_text = match.groups()
        if symbol not in PHONE_MAP:
            raise ValueError(f"No eSpeak mapping is defined for dictionary phone {symbol}")
        stress = int(stress_text) if stress_text is not None else None
        if symbol in VOWELS and stress is None:
            raise ValueError(f"Dictionary vowel {phone} has no stress marker")

        # In American English, unstressed AH is schwa; stressed AH is /ʌ/.
        espeak_phone = "@" if symbol == "AH" and stress == 0 else PHONE_MAP[symbol]
        marker = STRESS_MARK[stress] if symbol in VOWELS else ""
        mapped.append(f"{marker}{espeak_phone}")
    return mapped


def render_text(text: str, dictionary: dict[str, list[list[str]]], variant: int) -> tuple[list[dict], str]:
    words = split_text(text)
    if not words:
        raise ValueError("Enter at least one English word.")

    rendered_words = []
    for token in words:
        key = token.upper()
        pronunciations = dictionary.get(key)
        if not pronunciations:
            raise ValueError(f"'{token}' is not in the bundled pronunciation data.")
        if variant > len(pronunciations):
            raise ValueError(
                f"'{token}' has {len(pronunciations)} pronunciation(s); "
                f"variant {variant} was requested."
            )
        phones = pronunciations[variant - 1]
        rendered_words.append({
            "word": token,
            "variant": variant,
            "source_phones": phones,
            "synthesis_phones": map_pronunciation(phones),
        })

    # eSpeak uses | to separate phone symbols and || to separate words.
    phonetic_text = "||".join("|".join(item["synthesis_phones"]) for item in rendered_words)
    return rendered_words, phonetic_text


def render_fish_text(text: str, dictionary: dict[str, list[list[str]]], variant: int) -> tuple[list[dict], str]:
    """Add Fish Audio CMU Arpabet tags while preserving punctuation."""
    words, _ = render_text(text, dictionary, variant)
    word_iter = iter(words)
    pattern = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
    normalized = text.replace("’", "'")

    def phoneme_tag(_: re.Match[str]) -> str:
        item = next(word_iter)
        return f"<|phoneme_start|>{' '.join(item['source_phones'])}<|phoneme_end|>"

    return words, pattern.sub(phoneme_tag, normalized)


def dictionary_sha256(path: Path = DICTIONARY) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as dictionary_file:
        for block in iter(lambda: dictionary_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_mapping(dictionary: dict[str, list[list[str]]]) -> tuple[int, int]:
    seen = set()
    entries = 0
    for pronunciations in dictionary.values():
        for phones in pronunciations:
            entries += 1
            for phone in phones:
                match = re.fullmatch(r"([A-Z]+)([0-2])?", phone)
                if not match:
                    raise ValueError(f"Unexpected dictionary phone: {phone}")
                seen.add(match.group(1))
    missing = sorted(seen - PHONE_MAP.keys())
    if missing:
        raise ValueError(f"Dictionary phones without an eSpeak mapping: {', '.join(missing)}")
    return len(dictionary), entries


def make_audio(text: str, output_dir: Path, variant: int, engine: str) -> Path:
    dictionary = read_dictionary()
    words, phonetic_text = render_text(text, dictionary, variant)
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / "audio_preview.wav"

    if engine == "espeak":
        executable = shutil.which("espeak-ng")
        if executable is None:
            raise RuntimeError("eSpeak NG is required. Install it before generating a WAV file.")
        command = [executable, "-v", "en-us", "-w", str(wav_path), f"[[{phonetic_text}]]"]
        subprocess.run(command, check=True, capture_output=True, text=True)
        version = subprocess.run([executable, "--version"], check=True, capture_output=True, text=True)
        synthesizer_label = version.stdout.strip() or version.stderr.strip()
        phone_label = "eSpeak phones"
    elif engine == "fish":
        if datetime.now(timezone.utc).date() > FISH_FREE_THROUGH:
            raise RuntimeError(
                "Fish Audio's listed free-model period has ended; no request was sent. "
                "The program will not switch to a paid model."
            )
        api_key = os.environ.get("FISH_API_KEY")
        if not api_key:
            raise RuntimeError("Set the FISH_API_KEY environment variable to use Fish Audio.")
        words, fish_text = render_fish_text(text, dictionary, variant)
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
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                wav_path.write_bytes(response.read())
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Fish Audio returned HTTP {error.code}: {details[:500]}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Could not reach Fish Audio: {error.reason}") from error
        synthesizer_label = f"Fish Audio {FISH_MODEL} API"
        phone_label = "Fish CMU Arpabet phones"
    else:
        raise ValueError(f"Unsupported synthesis engine: {engine}")

    with wave.open(str(wav_path), "rb") as wav_file:
        audio_info = {
            "channels": wav_file.getnchannels(),
            "sample_width_bytes": wav_file.getsampwidth(),
            "sample_rate_hz": wav_file.getframerate(),
            "frames": wav_file.getnframes(),
            "duration_seconds": round(wav_file.getnframes() / wav_file.getframerate(), 3),
        }
    if audio_info["frames"] == 0:
        raise RuntimeError("The synthesizer returned an empty WAV file.")

    report = [
        "Pronunciation-to-audio demo",
        f"Input text: {text}",
        f"Pronunciation data: CMUdict 0.7b (SHA-256 {dictionary_sha256()})",
        f"Synthesizer: {synthesizer_label}",
        "",
        "Pronunciations:",
    ]
    for item in words:
        report.append(f"  {item['word']}: {' '.join(item['source_phones'])}")
        phones = item["source_phones"] if engine == "fish" else item["synthesis_phones"]
        report.append(f"    {phone_label}: {' '.join(phones)}")
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
    parser = argparse.ArgumentParser(description="Look up pronunciations and synthesize a short WAV preview.")
    parser.add_argument("--text", default="Hello world", help="English words to pronounce")
    parser.add_argument("--variant", type=int, default=1, help="One-based pronunciation choice for each word")
    parser.add_argument("--engine", choices=("espeak", "fish"), default="espeak", help="Speech synthesizer")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts", help="Directory for WAV and metrics")
    parser.add_argument("--validate-only", action="store_true", help="Check the dictionary and phone mapping without audio")
    args = parser.parse_args()

    if args.variant < 1:
        parser.error("--variant must be 1 or greater")
    try:
        dictionary = read_dictionary()
        word_count, pronunciation_count = validate_mapping(dictionary)
        if args.validate_only:
            render_text(args.text, dictionary, args.variant)
            print(f"Pronunciation entries loaded: {word_count:,} words, {pronunciation_count:,} pronunciations.")
            print("All dictionary phones have a mapping; the sample input is present.")
            return 0

        wav_path = make_audio(args.text, args.output_dir, args.variant, args.engine)
        print(f"Audio written to {wav_path}")
        print(f"Metrics written to {args.output_dir / 'metrics.txt'}")
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Audio generation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
