from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from kokoro import KPipeline


ROOT = Path(__file__).resolve().parent
DEMO_TEXT = ROOT / "data" / "demo_text.txt"
SAMPLE_RATE = 24000


def generate_audio(text: str, output_path: Path, voice: str, speed: float) -> dict:
    torch.set_num_threads(2)
    pipeline = KPipeline(lang_code="a", device="cpu")

    # The short demo uses common English words. Disable eSpeak fallback so all
    # speech synthesis comes from Kokoro's neural model and its built-in G2P.
    pipeline.g2p.fallback = None

    audio_parts = []
    for _graphemes, _phonemes, audio in pipeline(text, voice=voice, speed=speed):
        if audio is not None and len(audio):
            audio_parts.append(np.asarray(audio, dtype=np.float32))

    if not audio_parts:
        raise RuntimeError("The speech model returned no audio.")

    waveform = np.concatenate(audio_parts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, waveform, SAMPLE_RATE, subtype="PCM_16")

    return {
        "model": "Kokoro-82M",
        "voice": voice,
        "text": text,
        "sample_rate_hz": SAMPLE_RATE,
        "channels": 1,
        "audio_format": "WAV PCM 16-bit",
        "duration_seconds": round(len(waveform) / SAMPLE_RATE, 2),
        "peak_amplitude": round(float(np.max(np.abs(waveform))), 4),
        "rms_amplitude": round(float(np.sqrt(np.mean(np.square(waveform)))), 4),
        "wav_file": output_path.name,
        "inference_device": "cpu",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the short project speech demo.")
    parser.add_argument("--text", help="Text to speak; defaults to the project introduction")
    parser.add_argument("--voice", default="af_heart", help="Kokoro English voice")
    parser.add_argument("--speed", type=float, default=0.96, help="Speech rate")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()

    try:
        if args.speed <= 0:
            raise ValueError("--speed must be greater than zero.")
        text = (args.text or DEMO_TEXT.read_text(encoding="utf-8")).strip()
        if not text:
            raise ValueError("The text to speak is empty.")

        output_path = args.output_dir / "audio_generation_demo.wav"
        details = generate_audio(text, output_path, args.voice, args.speed)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "run.json").write_text(
            json.dumps(details, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        metric_lines = [
            "Audio generation run",
            f"Model: {details['model']} (pretrained)",
            f"Voice: {details['voice']}",
            f"Device: {details['inference_device']}",
            f"Duration: {details['duration_seconds']} seconds",
            f"Sample rate: {details['sample_rate_hz']} Hz",
            f"Format: {details['audio_format']}, mono",
            f"Peak amplitude: {details['peak_amplitude']}",
            f"RMS amplitude: {details['rms_amplitude']}",
            f"Input text: {details['text']}",
        ]
        (args.output_dir / "metrics.txt").write_text(
            "\n".join(metric_lines) + "\n",
            encoding="utf-8",
        )
        print(f"Audio written to {output_path}")
        print(f"Metrics written to {args.output_dir / 'metrics.txt'}")
        print(f"Run details written to {args.output_dir / 'run.json'}")
        return 0
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"Audio generation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
