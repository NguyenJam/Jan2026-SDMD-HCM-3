# Audio Generation Project: Retraining and Monitoring

This project updates a grapheme-to-phoneme model from pronunciation data and produces a short speech sample. A GitHub Actions workflow validates updates, checks for distribution drift, retrains the model when triggered, records evaluation results, and uploads the run outputs.

## Project files

- `model.py` trains the character-level LSTM pronunciation model and generates the audio sample.
- `prepare_new_data.py` validates the incoming pronunciation CSV.
- `monitor.py` checks data distributions and model metrics, then records monitoring results and alerts.
- `dvc.yaml` defines data preparation, training, and monitoring stages.
- `.github/workflows/train.yml` runs the pipeline on data/code pushes, weekly, or manually.
- `data/new_data.csv` is the current labelled update batch. Replace or extend this file when adding pronunciation examples.

## Update data format

Use a UTF-8 CSV with the following two columns. Pronunciations use CMU ARPAbet tokens separated by spaces.

```csv
word,pronunciation
example,IH0 G Z AE1 M P AH0 L
```

Push a valid update to `data/new_data.csv`. The workflow rejects malformed rows before training. New words are included in training; the existing validation split is retained for comparison.

## Workflow runs

The workflow starts on:

- A push to `main` that changes project code, dependencies, DVC configuration, or `data/new_data.csv`.
- Its weekly schedule.
- A manual run with either `retrain` or `monitor` selected.

A run checks word-length and phone-count distributions using the Population Stability Index. A score at or above `0.20` triggers retraining. Scheduled runs and the default manual option run the full pipeline. A manual monitoring-only run updates monitoring records when the preflight check does not request retraining.

## DVC storage setup

The DVC remote is configured for the DagsHub storage endpoint associated with this project. To enable remote pull and push, link the repository in DagsHub and add a repository Actions secret named `DAGSHUB_TOKEN`. The workflow uses this secret for the remote credentials; the token is not stored in the source files. Without the secret, the workflow still trains and uploads its run files as an Actions artifact, while DVC remote synchronization is skipped.

DVC records pipeline stage hashes in `dvc.lock`. Include that lock file in the repository after the first successful `dvc repro` so a later run can restore the matching cached model and monitoring outputs before processing new changes.

## Run locally

```bash
python -m pip install -r requirements.txt
python prepare_new_data.py
dvc repro
```

The default demo text is stored in `data/demo_text.txt`. The generated outputs are written to `artifacts/`, with monitoring reports and any alerts written under `monitoring/`.
