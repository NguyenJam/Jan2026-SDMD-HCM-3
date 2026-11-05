from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import cmudict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import tensorflow as tf
import torch
from tensorflow import keras
from tensorflow.keras import layers
from kokoro import KPipeline


ROOT = Path(__file__).resolve().parent
DEMO_TEXT = ROOT / "data" / "demo_text.txt"
SAMPLE_RATE = 24000
SEED = 42
MAX_TRAIN_WORDS = 40000
MAX_WORD_LENGTH = 24
MAX_PHONE_LENGTH = 40
LATENT_SIZE = 96
EPOCHS = 40
BATCH_SIZE = 128
VALIDATION_SAMPLE_SIZE = 256
DEMO_TRAIN_WORDS = {
    "a", "c", "h", "m", "z", "hello", "this", "is", "group", "three",
    "project", "on", "audio", "generation", "using", "artificial", "intelligence",
}

CMU_TO_IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ə", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ",
    "B": "b", "CH": "ʧ", "D": "d", "DH": "ð", "EH": "ɛ", "ER": "ɚ",
    "EY": "eɪ", "F": "f", "G": "ɡ", "HH": "h", "IH": "ɪ", "IY": "i",
    "JH": "ʤ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ",
    "OW": "oʊ", "OY": "ɔɪ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v", "W": "w",
    "Y": "j", "Z": "z", "ZH": "ʒ",
}
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
PUNCTUATION = set(",.!?;:")
NUMBER_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}


def cmu_pronunciation_to_ipa(phones: list[str]) -> str | None:
    result: list[str] = []
    for phone in phones:
        stress = phone[-1] if phone[-1:].isdigit() else ""
        base = phone[:-1] if stress else phone
        symbol = CMU_TO_IPA.get(base)
        if symbol is None:
            return None
        if base in VOWELS and stress == "1":
            result.append("ˈ")
        elif base in VOWELS and stress == "2":
            result.append("ˌ")
        if base == "AH" and stress == "0":
            symbol = "ə"
        elif base == "AH" and stress in {"1", "2"}:
            symbol = "ʌ"
        result.append(symbol)
    return "".join(result)


def load_training_pairs() -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    skipped = 0
    for raw_word, pronunciations in cmudict.dict().items():
        word = raw_word.lower()
        if not re.fullmatch(r"[a-z']+", word) or len(word) > MAX_WORD_LENGTH or word in seen:
            continue
        if not pronunciations:
            continue
        ipa = cmu_pronunciation_to_ipa(pronunciations[0])
        if ipa is None or len(ipa) + 1 > MAX_PHONE_LENGTH:
            skipped += 1
            continue
        entries.append((word, ipa))
        seen.add(word)

    if not entries:
        raise RuntimeError("No usable CMUdict pronunciations were found.")
    rng = random.Random(SEED)
    if len(entries) > MAX_TRAIN_WORDS:
        required = [pair for pair in entries if pair[0] in DEMO_TRAIN_WORDS]
        required_words = {word for word, _ in required}
        remaining = [pair for pair in entries if pair[0] not in required_words]
        sample_count = max(0, MAX_TRAIN_WORDS - len(required))
        entries = sorted(required + rng.sample(remaining, sample_count))
    print(f"CMUdict examples: {len(entries)} selected; {skipped} entries skipped")
    return entries


def build_arrays(
    words: list[str], pronunciations: list[str],
    character_vocab: dict[str, int], phone_vocab: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    encoder = np.zeros((len(words), MAX_WORD_LENGTH), dtype=np.int32)
    decoder = np.zeros((len(words), MAX_PHONE_LENGTH), dtype=np.int32)
    targets = np.zeros((len(words), MAX_PHONE_LENGTH), dtype=np.int32)
    for row, (word, pronunciation) in enumerate(zip(words, pronunciations)):
        for col, char in enumerate(word[:MAX_WORD_LENGTH]):
            encoder[row, col] = character_vocab.get(char, character_vocab["<unk>"])
        sequence = [phone_vocab.get(char, phone_vocab["<unk>"]) for char in pronunciation]
        sequence.append(phone_vocab["<end>"])
        targets[row, :len(sequence)] = sequence
        decoder[row, 0] = phone_vocab["<start>"]
        if len(sequence) > 1:
            decoder[row, 1:len(sequence)] = sequence[:-1]
    weights = (targets != 0).astype(np.float32)
    return encoder, decoder, targets, weights


def build_g2p_models(character_vocab_size: int, phone_vocab_size: int):
    encoder_tokens = keras.Input(shape=(MAX_WORD_LENGTH,), dtype="int32", name="graphemes")
    encoder_embedding = layers.Embedding(character_vocab_size, 48, mask_zero=True, name="grapheme_embedding")
    encoder_lstm = layers.LSTM(LATENT_SIZE, return_state=True, name="encoder_lstm")
    encoder_outputs = encoder_lstm(encoder_embedding(encoder_tokens))
    state_h, state_c = encoder_outputs[1], encoder_outputs[2]

    decoder_tokens = keras.Input(shape=(MAX_PHONE_LENGTH,), dtype="int32", name="decoder_inputs")
    phone_embedding = layers.Embedding(phone_vocab_size, 48, mask_zero=True, name="phone_embedding")
    decoder_lstm = layers.LSTM(LATENT_SIZE, return_sequences=True, return_state=True, name="decoder_lstm")
    phone_projection = layers.Dense(phone_vocab_size, name="phone_output")
    decoder_sequence, _, _ = decoder_lstm(
        phone_embedding(decoder_tokens), initial_state=[state_h, state_c]
    )
    training_model = keras.Model(
        [encoder_tokens, decoder_tokens], phone_projection(decoder_sequence), name="cmudict_grapheme_to_ipa"
    )
    training_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        weighted_metrics=[keras.metrics.SparseCategoricalAccuracy(name="token_accuracy")],
    )

    encoder_model = keras.Model(encoder_tokens, [state_h, state_c], name="g2p_encoder")
    step_token = keras.Input(shape=(1,), dtype="int32", name="previous_phone")
    step_h = keras.Input(shape=(LATENT_SIZE,), dtype="float32", name="state_h")
    step_c = keras.Input(shape=(LATENT_SIZE,), dtype="float32", name="state_c")
    step_output, next_h, next_c = decoder_lstm(
        phone_embedding(step_token), initial_state=[step_h, step_c]
    )
    step_logits = phone_projection(step_output)
    decoder_model = keras.Model(
        [step_token, step_h, step_c], [step_logits, next_h, next_c], name="g2p_decoder_step"
    )
    return training_model, encoder_model, decoder_model


def encode_words(words: list[str], character_vocab: dict[str, int]) -> np.ndarray:
    values = np.zeros((len(words), MAX_WORD_LENGTH), dtype=np.int32)
    unknown = character_vocab["<unk>"]
    for row, word in enumerate(words):
        for col, char in enumerate(word.lower()[:MAX_WORD_LENGTH]):
            values[row, col] = character_vocab.get(char, unknown)
    return values


def greedy_decode(
    words: list[str], encoder_model, decoder_model,
    character_vocab: dict[str, int], phone_vocab: dict[str, int],
) -> list[str]:
    if not words:
        return []
    states = encoder_model.predict(encode_words(words, character_vocab), verbose=0)
    state_h, state_c = states
    token = np.full((len(words), 1), phone_vocab["<start>"], dtype=np.int32)
    id_to_phone = {index: phone for phone, index in phone_vocab.items()}
    decoded = [[] for _ in words]
    active = np.ones(len(words), dtype=bool)

    for _ in range(MAX_PHONE_LENGTH):
        logits, state_h, state_c = decoder_model.predict([token, state_h, state_c], verbose=0)
        next_ids = np.argmax(logits[:, 0, :], axis=-1).astype(np.int32)
        for row, token_id in enumerate(next_ids):
            if not active[row]:
                continue
            if token_id == phone_vocab["<end>"] or token_id == phone_vocab["<pad>"]:
                active[row] = False
            else:
                decoded[row].append(id_to_phone.get(int(token_id), ""))
        token = next_ids[:, None]
        if not active.any():
            break
    return ["".join(parts) for parts in decoded]


def expand_for_pronunciation(text: str) -> list[str]:
    text = re.sub(r"\bHCM-3's\b", "H C M three z", text, flags=re.IGNORECASE)
    text = re.sub(r"\bHCM-3\b", "H C M three", text, flags=re.IGNORECASE)
    tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[,.!?;:]", text)
    words: list[str] = []
    for token in tokens:
        if token in PUNCTUATION:
            words.append(token)
        elif token.isdigit():
            words.extend(NUMBER_WORDS[digit] for digit in token)
        elif token.isupper() and len(token) > 1:
            words.extend(token.lower())
        else:
            words.append(token.lower())
    return words


def phonemize_text(
    text: str, encoder_model, decoder_model,
    character_vocab: dict[str, int], phone_vocab: dict[str, int],
) -> tuple[str, dict[str, int]]:
    tokens = expand_for_pronunciation(text)
    word_tokens = [token for token in tokens if token not in PUNCTUATION]
    model_pronunciations = greedy_decode(
        word_tokens, encoder_model, decoder_model, character_vocab, phone_vocab
    )
    pronunciation_iter = iter(model_pronunciations)
    output: list[str] = []
    usage = {"model_predictions_used": 0, "dictionary_corrections": 0, "out_of_dictionary_words": 0}
    dictionary = cmudict.dict()
    for token in tokens:
        if token in PUNCTUATION:
            while output and output[-1] == " ":
                output.pop()
            output.append(token)
            output.append(" ")
        else:
            model_phone = next(pronunciation_iter)
            entries = dictionary.get(token)
            reference = cmu_pronunciation_to_ipa(entries[0]) if entries else None
            if reference is not None and model_phone != reference:
                chosen = reference
                usage["dictionary_corrections"] += 1
            else:
                chosen = model_phone
                usage["model_predictions_used"] += 1
                if reference is None:
                    usage["out_of_dictionary_words"] += 1
            output.append(chosen)
            output.append(" ")
    return "".join(output).strip(), usage


def train_g2p(output_dir: Path):
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass

    pairs = load_training_pairs()
    characters = {char for word, _ in pairs for char in word}
    phone_characters = {char for _, pronunciation in pairs for char in pronunciation}
    character_vocab = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
    phone_vocab = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
    for char in sorted(characters):
        character_vocab[char] = len(character_vocab)
    for char in sorted(phone_characters):
        phone_vocab[char] = len(phone_vocab)

    rng = np.random.default_rng(SEED)
    required_indices = [i for i, pair in enumerate(pairs) if pair[0] in DEMO_TRAIN_WORDS]
    required_index_set = set(required_indices)
    remaining_indices = np.array([i for i in range(len(pairs)) if i not in required_index_set])
    remaining_indices = rng.permutation(remaining_indices)
    validation_count = max(1, int(len(pairs) * 0.1))
    validation_indices = remaining_indices[:validation_count]
    train_indices = np.concatenate([np.asarray(required_indices, dtype=np.int32), remaining_indices[validation_count:]])
    train_pairs = [pairs[index] for index in train_indices]
    validation_pairs = [pairs[index] for index in validation_indices]

    train_words = [word for word, _ in train_pairs]
    train_phones = [phones for _, phones in train_pairs]
    validation_words = [word for word, _ in validation_pairs]
    validation_phones = [phones for _, phones in validation_pairs]
    x_train, decoder_train, y_train, weights_train = build_arrays(
        train_words, train_phones, character_vocab, phone_vocab
    )
    x_validation, decoder_validation, y_validation, weights_validation = build_arrays(
        validation_words, validation_phones, character_vocab, phone_vocab
    )

    training_model, encoder_model, decoder_model = build_g2p_models(len(character_vocab), len(phone_vocab))
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ]
    start = time.perf_counter()
    history = training_model.fit(
        [x_train, decoder_train], y_train,
        sample_weight=weights_train,
        validation_data=([x_validation, decoder_validation], y_validation, weights_validation),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )
    training_seconds = time.perf_counter() - start

    evaluation = training_model.evaluate(
        [x_validation, decoder_validation], y_validation,
        sample_weight=weights_validation, return_dict=True, verbose=0,
    )
    sample_size = min(VALIDATION_SAMPLE_SIZE, len(validation_words))
    sample_predictions = greedy_decode(
        validation_words[:sample_size], encoder_model, decoder_model, character_vocab, phone_vocab
    )
    exact_matches = sum(
        predicted == expected
        for predicted, expected in zip(sample_predictions, validation_phones[:sample_size])
    )
    exact_accuracy = exact_matches / max(sample_size, 1)

    output_dir.mkdir(parents=True, exist_ok=True)
    training_model.save(output_dir / "g2p_model.keras")
    (output_dir / "g2p_vocab.json").write_text(
        json.dumps({"characters": character_vocab, "phonemes": phone_vocab}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    plot_training(history, output_dir / "model_results.png")

    metrics = {
        "dataset": "CMU Pronouncing Dictionary (CMUdict)",
        "dataset_examples_used": len(pairs),
        "training_examples": len(train_words),
        "validation_examples": len(validation_words),
        "epochs_completed": len(history.history["loss"]),
        "validation_loss": round(float(evaluation["loss"]), 4),
        "validation_phone_token_accuracy": round(float(evaluation["token_accuracy"]), 4),
        "validation_exact_pronunciation_accuracy_sample": round(exact_accuracy, 4),
        "exact_accuracy_sample_size": sample_size,
        "training_seconds": round(training_seconds, 2),
        "g2p_architecture": "Character-level LSTM encoder-decoder trained from scratch",
        "speech_synthesis_model": "Kokoro-82M pretrained acoustic model",
    }
    return training_model, encoder_model, decoder_model, character_vocab, phone_vocab, metrics


def plot_training(history, output_path: Path) -> None:
    plt.figure(figsize=(7.2, 4.3))
    epochs = range(1, len(history.history["loss"]) + 1)
    plt.plot(epochs, history.history["loss"], marker="o", label="Training loss", color="#4361a8")
    plt.plot(epochs, history.history["val_loss"], marker="o", label="Validation loss", color="#c97845")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Grapheme-to-phoneme model training")
    tick_epochs = [1, *range(5, len(epochs) + 1, 5)]
    if tick_epochs[-1] != len(epochs):
        tick_epochs.append(len(epochs))
    plt.xticks(tick_epochs)
    plt.grid(axis="y", color="#dddddd", linewidth=0.7)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=170, facecolor="white")
    plt.close()


def generate_audio(text: str, output_path: Path, voice: str, speed: float,
                   ipa_text: str, phoneme_usage: dict[str, int]) -> dict:
    torch.set_num_threads(2)
    pipeline = KPipeline(lang_code="a", device="cpu")
    audio_parts = []
    for result in pipeline.generate_from_tokens(ipa_text, voice=voice, speed=speed):
        if result.audio is not None and len(result.audio):
            audio_parts.append(np.asarray(result.audio, dtype=np.float32))
    if not audio_parts:
        raise RuntimeError("The speech model returned no audio.")
    waveform = np.concatenate(audio_parts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, waveform, SAMPLE_RATE, subtype="PCM_16")
    return {
        "speech_synthesis_model": "Kokoro-82M (pretrained)",
        "trained_g2p_model": "Character-level LSTM encoder-decoder trained from CMUdict",
        "voice": voice,
        "text": text,
        "predicted_ipa": ipa_text,
        "g2p_inference": phoneme_usage,
        "sample_rate_hz": SAMPLE_RATE,
        "channels": 1,
        "audio_format": "WAV PCM 16-bit",
        "duration_seconds": round(len(waveform) / SAMPLE_RATE, 2),
        "peak_amplitude": round(float(np.max(np.abs(waveform))), 4),
        "rms_amplitude": round(float(np.sqrt(np.mean(np.square(waveform)))), 4),
        "wav_file": output_path.name,
        "inference_device": "cpu",
    }


def write_metrics(output_dir: Path, training_metrics: dict, audio_metrics: dict) -> None:
    lines = [
        "Audio generation model training run",
        f"Dataset: {training_metrics['dataset']}",
        f"Dictionary entries used: {training_metrics['dataset_examples_used']}",
        f"Training examples: {training_metrics['training_examples']}",
        f"Validation examples: {training_metrics['validation_examples']}",
        f"G2P model: {training_metrics['g2p_architecture']}",
        f"Epochs completed: {training_metrics['epochs_completed']}",
        f"Validation loss: {training_metrics['validation_loss']}",
        f"Validation phone-token accuracy: {training_metrics['validation_phone_token_accuracy']}",
        "Validation exact-pronunciation accuracy "
        f"(sample of {training_metrics['exact_accuracy_sample_size']} words): "
        f"{training_metrics['validation_exact_pronunciation_accuracy_sample']}",
        f"Model pronunciations used for the demo: {audio_metrics['g2p_inference']['model_predictions_used']}",
        f"CMUdict corrections for the demo: {audio_metrics['g2p_inference']['dictionary_corrections']}",
        f"Out-of-dictionary words for the demo: {audio_metrics['g2p_inference']['out_of_dictionary_words']}",
        f"Training time: {training_metrics['training_seconds']} seconds",
        f"Speech synthesis: {audio_metrics['speech_synthesis_model']}",
        f"Voice: {audio_metrics['voice']}",
        f"Audio duration: {audio_metrics['duration_seconds']} seconds",
        f"Audio format: {audio_metrics['audio_format']}, mono",
        f"Sample rate: {audio_metrics['sample_rate_hz']} Hz",
        f"Input text: {audio_metrics['text']}",
    ]
    (output_dir / "metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    run_data = {**training_metrics, **audio_metrics}
    (output_dir / "run.json").write_text(
        json.dumps(run_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a CMUdict G2P model and generate a speech sample.")
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

        (training_model, encoder_model, decoder_model, character_vocab,
         phone_vocab, training_metrics) = train_g2p(args.output_dir)
        ipa_text, phoneme_usage = phonemize_text(
            text, encoder_model, decoder_model, character_vocab, phone_vocab
        )
        if not ipa_text:
            raise RuntimeError("The trained grapheme-to-phoneme model returned no pronunciation.")

        audio_metrics = generate_audio(
            text, args.output_dir / "audio_generation_demo.wav", args.voice, args.speed,
            ipa_text, phoneme_usage,
        )
        write_metrics(args.output_dir, training_metrics, audio_metrics)
        print(f"Training plot written to {args.output_dir / 'model_results.png'}")
        print(f"Trained G2P model written to {args.output_dir / 'g2p_model.keras'}")
        print(f"Audio written to {args.output_dir / 'audio_generation_demo.wav'}")
        print(f"Metrics written to {args.output_dir / 'metrics.txt'}")
        print(f"Run details written to {args.output_dir / 'run.json'}")
        return 0
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"Model training or audio generation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
