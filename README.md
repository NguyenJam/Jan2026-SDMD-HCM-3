# Pronunciation-to-audio demo

Group HCM-3 | COS40007 Artificial Intelligence Engineering

We use Fish Audio as the teacher model for this project and as the voice generator for the demo. CMUdict supplies the pronunciation and stress for words it contains. The program adds those pronunciations as phoneme tags, then Fish Audio generates the final WAV directly.

## Teacher model

Fish Audio receives the phrase with CMU Arpabet pronunciations for words found in the dictionary and produces speech while following those phone sequences. For README narration, names or other words missing from CMUdict are passed as ordinary text so Fish Audio can read them from spelling. Its generated audio is the demo output, so there is no separate synthetic baseline voice. This version does not train or fine-tune a second model from Fish Audio's output.

## GitHub Actions

The workflow runs on pushes to `main`, pull requests targeting `main`, and manual dispatch. Each run reads this README, turns its readable text into a Fish Audio WAV, and uploads the audio and pronunciation details as the `audio-demo` artifact. Headings and prose are included; code blocks, Markdown formatting, and link addresses are skipped. Download the artifact from the completed run under the repository's **Actions** tab.

## Run locally

With Python 3.11 or later, narrate the readable parts of this README with:

```bash
python model.py --readme --output-dir artifacts/fish
```

The WAV and pronunciation details are saved in `artifacts/fish`. To generate speech from your own phrase instead, use `--text`:

```bash
python model.py --text "Hello world" --output-dir artifacts/fish
```
