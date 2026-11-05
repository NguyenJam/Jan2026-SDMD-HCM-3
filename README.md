# Pronunciation-to-audio demo

Group HCM-3 | COS40007 Artificial Intelligence Engineering

We use Fish Audio as the teacher model for this project and as the voice generator for the demo. CMUdict supplies the pronunciation and stress for each word. The program adds those pronunciations as phoneme tags, then Fish Audio generates the final WAV directly.

## Teacher model

Fish Audio receives the phrase with its CMU Arpabet pronunciations and produces fluent speech while following those phone sequences. Its generated audio is the demo output, so there is no separate synthetic baseline voice. This version does not train or fine-tune a second model from Fish Audio's output.

## GitHub Actions

The workflow runs on pushes to `main`, pull requests targeting `main`, and manual dispatch. Each run generates a Fish Audio WAV and pronunciation details, then uploads them as the `audio-demo` artifact. Download the artifact from the completed run under the repository's **Actions** tab.

## Run locally

With Python 3.11 or later, generate a WAV with:

```bash
python model.py --text "Hello world" --output-dir artifacts/fish
```

To validate the pronunciation data and sample text without generating audio:

```bash
python model.py --validate-only --text "Hello world"
```
