# Audio generation demo

This project turns a short text introduction into a WAV file. It uses Kokoro-82M, a pretrained text-to-speech model, to generate speech locally on the GitHub Actions runner. The demo does not use Fish Audio or an API key.

The default text is:

> Hello world. This is Group H C M Three's project on audio generation using artificial intelligence.

## Run in GitHub Actions

Every push to `main`, pull request targeting `main`, or manual workflow run generates `audio_generation_demo.wav`. Open the completed run in **Actions** and download the `audio-generation-demo` artifact.

## Run locally

Use Python 3.12 for the pinned inference dependencies.

```bash
python -m pip install -r requirements.txt
python model.py
```

To speak different text:

```bash
python model.py --text "Hello world. This is our audio generation demo."
```

The selected voice is `af_heart`; change it with `--voice`. Inference runs on CPU. The workflow downloads the model weights when it runs, then saves the WAV and a small `run.json` file in `artifacts/`.

## Model and license

Kokoro-82M is an open-weight pretrained speech model. This project uses the model for inference and does not train or fine-tune it. The source code and model weights are distributed under Apache 2.0 terms; see the [official model repository](https://github.com/hexgrad/kokoro) and [model card](https://huggingface.co/hexgrad/Kokoro-82M) for details.
