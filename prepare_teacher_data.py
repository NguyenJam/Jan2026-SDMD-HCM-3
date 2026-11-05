from __future__ import annotations

import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEMO_TEXT = ROOT / "data" / "demo_manifesto.txt"
OUTPUT_DIR = ROOT / "data" / "teacher"
API_URL = "https://api.fish.audio/v1/tts/stream/with-timestamp"
MODEL = "s2.1-pro-free"


def make_teacher_data(api_key: str) -> tuple[Path, Path]:
    text = DEMO_TEXT.read_text(encoding="utf-8")
    payload = json.dumps({
        "text": text,
        "format": "wav",
        "sample_rate": 24000,
        "normalize": True,
        "temperature": 0.8,
        "top_p": 0.8,
        "prosody": {"speed": 0.94, "volume": 0, "normalize_loudness": True},
        "chunk_length": 300,
        "latency": "normal",
    }).encode("utf-8")
    request = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "model": MODEL,
        },
        method="POST",
    )

    audio_chunks: list[bytes] = []
    alignments: dict[int, dict] = {}
    offsets: dict[int, float] = {}
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                content = line[5:].strip()
                if not content or content == "[DONE]":
                    continue
                message = json.loads(content)
                if message.get("error"):
                    raise RuntimeError(str(message["error"]))
                audio_data = message.get("audio_base64")
                if audio_data:
                    audio_chunks.append(base64.b64decode(audio_data))
                sequence = message.get("chunk_seq")
                if sequence is not None:
                    offsets[int(sequence)] = float(message.get("chunk_audio_offset_sec", 0))
                    alignment = message.get("alignment")
                    if alignment is not None:
                        alignments[int(sequence)] = alignment
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Fish Audio returned HTTP {error.code}: {details[:500]}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach Fish Audio: {error.reason}") from error

    audio = b"".join(audio_chunks)
    if not audio.startswith(b"RIFF") or not audio_chunks:
        raise RuntimeError("Fish Audio did not return a WAV audio stream.")
    audio = bytearray(audio)
    data_marker = audio.find(b"data")
    if data_marker < 0 or data_marker + 8 > len(audio):
        raise RuntimeError("Fish Audio returned a malformed WAV header.")
    data_start = data_marker + 8
    audio[4:8] = (len(audio) - 8).to_bytes(4, "little")
    audio[data_marker + 4:data_marker + 8] = (len(audio) - data_start).to_bytes(4, "little")

    segments = []
    for sequence in sorted(alignments):
        offset = offsets.get(sequence, 0.0)
        for segment in alignments[sequence].get("segments", []):
            segments.append({
                "text": segment["text"],
                "start": round(offset + float(segment["start"]), 4),
                "end": round(offset + float(segment["end"]), 4),
            })
    if not segments:
        raise RuntimeError("Fish Audio returned no word alignment for the teacher audio.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audio_path = OUTPUT_DIR / "manifesto_teacher.wav"
    alignment_path = OUTPUT_DIR / "manifesto_alignment.json"
    audio_path.write_bytes(bytes(audio))
    alignment_path.write_text(
        json.dumps({
            "teacher": "Fish Audio",
            "model": MODEL,
            "sample_rate": 24000,
            "text_source": "../demo_manifesto.txt",
            "segments": segments,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return audio_path, alignment_path


def main() -> int:
    api_key = os.environ.get("FISH_API_KEY", "").strip()
    if not api_key:
        print("Set FISH_API_KEY to prepare the teacher sample.", file=sys.stderr)
        return 2
    try:
        audio_path, alignment_path = make_teacher_data(api_key)
        print(f"Teacher audio saved to {audio_path}")
        print(f"Word alignment saved to {alignment_path}")
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"Teacher-data preparation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
