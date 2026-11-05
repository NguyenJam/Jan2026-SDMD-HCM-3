# Student speech-generation demo

Group HCM-3 | COS40007 Artificial Intelligence Engineering

This project trains a small audio student from a Fish Audio teacher recording, then uses the student to generate a spoken version of the manifesto. Fish Audio supplies the example speech and word-level timings. The training code looks up each word's phonemes and stress labels, then estimates phone-level durations by dividing each word's recorded duration among its phones. The student learns to predict log-mel spectrogram frames from that phoneme context. Griffin-Lim converts the predicted frames into a WAV file.

Fish Audio is used while preparing the teacher sample. The Actions workflow trains and runs the student without calling Fish Audio at inference.

## Model scope

The bundled teacher recording is one reading of the manifesto. Since Fish provides word-level, not phone-level, timings, the phone boundaries used for training are estimates. The student is a CPU-sized proof of concept trained on that single script, not a general-purpose TTS model. Its voice will be less natural than the teacher and may not generalize well to new text. Words need to be present in the bundled pronunciation dictionary.

The student predicts log-mel frames from phoneme context. The waveform reconstruction uses Griffin-Lim; there is no separately trained neural vocoder.

## GitHub Actions

Every push to `main`, pull request targeting `main`, and manual dispatch trains the student from the bundled Fish Audio sample, generates the manifesto WAV, and uploads the checkpoint, metrics, training curve, and audio as the `manifesto-student-audio` artifact. Find the artifact in the completed run under **Actions**.

## Run locally

Install dependencies and train the student:

```bash
python -m pip install -r requirements.txt
python train_student.py --epochs 12 --output-dir artifacts/train
```

Generate the manifesto with the trained student:

```bash
python model.py --demo --model artifacts/train/student_model.pt --output-dir artifacts/demo
```

To generate speech from your own text, use `--text` instead of `--demo`. Every word must have a pronunciation in the bundled dictionary.

The Fish-generated teacher WAV and its word timings are included under `data/teacher`. Regenerating them is optional and requires `FISH_API_KEY` to be set in the environment before running `python prepare_teacher_data.py`.
