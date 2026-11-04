# CMUdict audio workflow

Group HCM-3 | COS40007 Artificial Intelligence Engineering

This workflow adapts the Studio 7 GitHub Actions exercise to the planned pronunciation-to-audio project. It looks up each word in CMUdict 0.7b, maps the selected ARPAbet pronunciation to eSpeak NG symbols, creates a short WAV preview, and saves a run summary.

## Run locally

Requirements: Python 3.11 or later and eSpeak NG on your PATH. Python dependencies use the standard library only.

```bash
python -m pip install -r requirements.txt
python model.py --text "Hello world" --output-dir artifacts
```

The `artifacts` folder receives `audio_preview.wav` and `metrics.txt`. You can check a different phrase with `--text`. If a word has more than one dictionary pronunciation, `--variant 2` selects the second entry for every word in the input; the default is the first entry. The run stops with a clear error for an unsupported word or unavailable pronunciation choice.

To check the bundled dictionary and phone mapping without an audio engine:

```bash
python model.py --validate-only --text "Hello world"
```

## GitHub Actions trigger

The workflow at `.github/workflows/train.yml` runs on pushes to `main`, pull requests targeting `main`, and manual dispatch. It sets up Python 3.11, installs eSpeak NG, runs the audio smoke check, then uploads the WAV and run summary as the `cmudict-audio-preview` artifact. For manual runs, select the workflow in the repository's **Actions** tab and choose **Run workflow** after this file has been added to the default branch.

Add these files and folders to the root of the group repository. Push the workflow file to `main` to enable the push trigger; open a pull request to exercise the pull-request trigger.

## Project boundary

eSpeak NG is used here as a temporary CI synthesis baseline so the workflow can prove that phoneme lookup and mapping produce a non-empty WAV. It is not the project's chosen production TTS engine, and the voice or model still needs to be selected by the group. CMUdict provides pronunciations, not audio recordings or training data.

The `data` folder includes the CMUdict 0.7b source snapshot, its license, and a source record with the SHA-256 checksum. Preserve the license notice when redistributing the dictionary.
