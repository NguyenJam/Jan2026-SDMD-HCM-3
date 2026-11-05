# Audio generation demo

This project converts a short English introduction into a WAV clip with the Kokoro-82M text-to-speech model and the `af_heart` voice. Each run also creates a waveform image and audio metrics.

## GitHub Actions

The workflow runs when code is pushed to `main`, when a pull request targets `main`, or when started manually. It installs the project dependencies, generates the audio, and uploads the WAV, waveform, metrics, and run details as the `audio-generation-demo` artifact. Download the artifact from the completed Actions run.

## Run locally

Use Python 3.12:

```bash
python -m pip install -r requirements.txt
python model.py --output-dir artifacts
```

The default text is:

> Hello, this is group HCM-3's project on audio generation using artificial intelligence

To provide different text, pass it with `--text`:

```bash
python model.py --text "A short sentence to speak." --output-dir artifacts
```

## Model

The project uses the published Kokoro-82M speech model for text-to-speech inference. See the [Kokoro repository](https://github.com/hexgrad/kokoro) and [Kokoro-82M model card](https://huggingface.co/hexgrad/Kokoro-82M) for model details and license information.
