# Pronunciation-to-audio demo

Group HCM-3 | COS40007 Artificial Intelligence Engineering

The workflow generates two WAV files from the same short English phrase. We use Fish Audio as the teacher model for this project. eSpeak NG produces the baseline for comparison. Both outputs and their run details are uploaded as the `audio-demo` artifact.

## Teacher and baseline

CMUdict supplies the word pronunciations and stress markers. The program inserts those pronunciations into Fish Audio's phoneme-control format, and Fish Audio generates a reference WAV from the tagged phrase. eSpeak NG synthesizes the same phrase as a baseline, so the two outputs can be compared.

The Fish output serves as the teacher reference for pronunciation and audio quality. The current code does not train or fine-tune a separate student voice model.

## Workflow outputs

The GitHub Actions workflow runs on pushes to `main`, pull requests targeting `main`, and manual dispatch. A push to `main` generates both the Fish teacher WAV and eSpeak baseline, then uploads them with pronunciation and audio details. Pull requests run the eSpeak baseline; Fish generation is skipped when repository secrets are unavailable to the pull-request workflow.

After a run completes, open it from the repository's **Actions** tab and download `audio-demo` to listen to and compare the WAV files.

## Local use

The eSpeak baseline runs with Python 3.11 or later and eSpeak NG installed:

```bash
python model.py --text "Hello world" --engine espeak --output-dir artifacts/espeak
```

The Fish teacher path is selected with `--engine fish` and reads the API credential from the `FISH_API_KEY` environment variable. The credential is not stored in the project files.
