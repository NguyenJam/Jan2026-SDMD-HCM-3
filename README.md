# Audio generation model

This project trains a character-level LSTM that maps words to phonemes using CMUdict pronunciation labels, installed through the `cmudict` package. The model predicts IPA for the input sentence. Known words are checked against the dictionary to correct mismatched predictions; words missing from the dictionary use the model's prediction. The resulting IPA is passed to Kokoro-82M to synthesize the WAV file. Kokoro supplies acoustic speech synthesis, while the grapheme-to-phoneme model is trained by this project on each run.

## Run

Use Python 3.12:

```bash
python -m pip install -r requirements.txt
python model.py --output-dir artifacts
```

The default sentence is stored in `data/demo_text.txt`. To synthesize different text:

```bash
python model.py --text "A short sentence to speak." --output-dir artifacts
```

## Outputs

The run writes these files to the output directory:

- `audio_generation_demo.wav`: synthesized speech.
- `model_results.png`: training and validation loss by epoch.
- `metrics.txt`: dataset, training, validation and audio measurements.
- `run.json`: the same run details in structured form.
- `g2p_model.keras` and `g2p_vocab.json`: trained model and its vocabularies.

## Automated run

The workflow in `.github/workflows/train.yml` starts on pushes or pull requests to `main` and can also be started manually. It installs the dependencies, trains the grapheme-to-phoneme model, generates the speech sample, and uploads the output files as one artifact.

The voice synthesis step uses the published [Kokoro-82M model](https://huggingface.co/hexgrad/Kokoro-82M).
