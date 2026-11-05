from __future__ import annotations

import difflib
import json
import math
import re
import struct
import wave
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
DICTIONARY = ROOT / "data" / "cmudict-0.7b"
TEACHER_AUDIO = ROOT / "data" / "teacher" / "manifesto_teacher.wav"
TEACHER_ALIGNMENT = ROOT / "data" / "teacher" / "manifesto_alignment.json"
SAMPLE_RATE = 24000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
PHONE_PATTERN = re.compile(r"([A-Z]+)([0-2])?$")
TOKEN_PATTERN = re.compile(r"\[[^\]]+\]|(?<![A-Za-z0-9'])[A-Za-z]+(?:'[A-Za-z]+)?|[.,!?;:…]+")
PRONUNCIATION_OVERRIDES = {
    "ADDICT'S": ["AH0", "D", "IH1", "K", "T", "S"],
    "HUNGERED": ["HH", "AH1", "NG", "G", "ER0", "D"],
    "INCOMPETENCIES": ["IH2", "N", "K", "AA1", "M", "P", "AH0", "T", "AH0", "N", "S", "IY0", "Z"],
    "SADISTS": ["S", "EY1", "D", "AH0", "S", "T", "S"],
    "TECHNOBRAIN": ["T", "EH1", "K", "N", "OW0", "B", "R", "EY1", "N"],
}


def read_dictionary(path: Path = DICTIONARY) -> dict[str, list[list[str]]]:
    entries: dict[str, list[list[str]]] = defaultdict(list)
    with path.open(encoding="latin-1") as dictionary_file:
        for line in dictionary_file:
            if not line or line.startswith(";;;"):
                continue
            fields = line.strip().split()
            if len(fields) < 2:
                continue
            raw_word, *phones = fields
            word = re.sub(r"\(\d+\)$", "", raw_word).upper()
            entries[word].append(phones)
    return dict(entries)


def lookup_pronunciation(word: str, dictionary: dict[str, list[list[str]]]) -> list[str] | None:
    pronunciations = dictionary.get(word.upper())
    if pronunciations:
        return pronunciations[0]
    return PRONUNCIATION_OVERRIDES.get(word.upper())


def split_words(text: str) -> list[str]:
    normalized = text.replace("’", "'")
    without_cues = re.sub(r"\[[^\]]+\]", " ", normalized)
    return re.findall(r"(?<![A-Za-z0-9'])[A-Za-z]+(?:'[A-Za-z]+)?", without_cues)


def phone_parts(phone: str) -> tuple[str, int]:
    match = PHONE_PATTERN.fullmatch(phone)
    if not match:
        raise ValueError(f"Unexpected pronunciation phone: {phone}")
    base, stress = match.groups()
    return base, int(stress) if stress is not None else -1


def phone_weight(phone: str) -> float:
    base, stress = phone_parts(phone)
    if base in {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}:
        return {0: 1.8, 1: 2.5, 2: 2.1}.get(stress, 2.0)
    if base in {"P", "T", "K", "B", "D", "G", "CH", "JH"}:
        return 0.8
    return 1.0


def mel_filterbank(sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    def hz_to_mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_low = hz_to_mel(0.0)
    mel_high = hz_to_mel(sample_rate / 2)
    points = [mel_to_hz(mel_low + (mel_high - mel_low) * i / (N_MELS + 1)) for i in range(N_MELS + 2)]
    bins = torch.linspace(0, sample_rate / 2, N_FFT // 2 + 1)
    filters = torch.zeros(N_MELS, N_FFT // 2 + 1)
    for band in range(N_MELS):
        left, center, right = points[band:band + 3]
        filters[band] = torch.clamp(torch.minimum(
            (bins - left) / max(center - left, 1e-8),
            (right - bins) / max(right - center, 1e-8),
        ), min=0.0)
    return filters


def read_teacher_wave(path: Path = TEACHER_AUDIO) -> torch.Tensor:
    with wave.open(str(path), "rb") as audio_file:
        channels = audio_file.getnchannels()
        sample_width = audio_file.getsampwidth()
        sample_rate = audio_file.getframerate()
        frame_count = audio_file.getnframes()
        raw = audio_file.readframes(frame_count)
    if sample_width != 2:
        raise ValueError("Teacher WAV must use 16-bit PCM samples.")
    audio = torch.tensor(struct.unpack("<" + "h" * (len(raw) // 2), raw), dtype=torch.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(dim=1)
    if sample_rate != SAMPLE_RATE:
        if sample_rate == 48000:
            audio = audio.reshape(-1, 2).mean(dim=1)
            sample_rate = 24000
        elif sample_rate == 44100:
            audio = audio[::2]
            sample_rate = 22050
        else:
            new_length = round(audio.numel() * SAMPLE_RATE / sample_rate)
            audio = torch.nn.functional.interpolate(
                audio.view(1, 1, -1), size=new_length, mode="linear", align_corners=False
            ).view(-1)
            sample_rate = SAMPLE_RATE
    if sample_rate != SAMPLE_RATE:
        new_length = round(audio.numel() * SAMPLE_RATE / sample_rate)
        audio = torch.nn.functional.interpolate(
            audio.view(1, 1, -1), size=new_length, mode="linear", align_corners=False
        ).view(-1)
    return audio.contiguous()


def audio_log_mel(audio: torch.Tensor) -> torch.Tensor:
    window = torch.hann_window(N_FFT)
    spectrum = torch.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window=window,
        center=True,
        return_complex=True,
    ).abs()
    mel = mel_filterbank() @ spectrum
    return torch.log1p(mel * 1000.0).transpose(0, 1).contiguous()


def normalise_alignment_word(word: str) -> str:
    return re.sub(r"[^a-z]", "", word.lower().replace("’", "'"))


def align_teacher_words(
    text_words: list[str],
    segments: list[dict],
) -> dict[int, dict]:
    aligned_words = [str(segment.get("text", "")) for segment in segments]
    source_normal = [normalise_alignment_word(word) for word in text_words]
    align_normal = [normalise_alignment_word(word) for word in aligned_words]
    matcher = difflib.SequenceMatcher(a=source_normal, b=align_normal, autojunk=False)
    mapping: dict[int, dict] = {}
    for tag, source_start, source_end, align_start, align_end in matcher.get_opcodes():
        source_indices = list(range(source_start, source_end))
        align_indices = list(range(align_start, align_end))
        if tag == "equal":
            pairs = zip(source_indices, align_indices)
        elif tag == "replace" and source_indices and align_indices:
            pairs = zip(source_indices, align_indices) if len(source_indices) == len(align_indices) else []
        else:
            continue
        for source_index, align_index in pairs:
            mapping[source_index] = segments[align_index]
    return mapping


def distributed_phone_frames(phones: list[str], frame_count: int) -> list[int]:
    if frame_count < len(phones):
        return [0] * len(phones)
    weights = [phone_weight(phone) for phone in phones]
    total = sum(weights)
    exact = [frame_count * weight / total for weight in weights]
    counts = [max(1, int(value)) for value in exact]
    while sum(counts) > frame_count:
        candidates = [index for index, count in enumerate(counts) if count > 1]
        if not candidates:
            break
        chosen = min(candidates, key=lambda index: exact[index] - counts[index])
        counts[chosen] -= 1
    while sum(counts) < frame_count:
        chosen = max(range(len(counts)), key=lambda index: exact[index] - counts[index])
        counts[chosen] += 1
        exact[chosen] -= 1
    return counts


def acoustic_features(
    phones: list[str],
    durations: list[int],
    phone_to_id: dict[str, int],
) -> list[list[float]]:
    features: list[list[float]] = []
    for index, (phone, duration) in enumerate(zip(phones, durations)):
        base, stress = phone_parts(phone)
        current_id = phone_to_id[base]
        previous_id = phone_to_id[phone_parts(phones[index - 1])[0]] if index else 0
        next_id = phone_to_id[phone_parts(phones[index + 1])[0]] if index + 1 < len(phones) else 0
        stress_value = stress / 2.0 if stress >= 0 else -0.5
        position = index / max(1, len(phones) - 1)
        word_size = min(len(phones) / 16.0, 1.0)
        duration_value = min(math.log1p(duration) / 4.0, 1.0)
        for frame in range(duration):
            phase = frame / max(1, duration - 1)
            features.append([
                float(previous_id), float(current_id), float(next_id), stress_value,
                phase, position, word_size, duration_value,
            ])
    return features


class AcousticStudent(nn.Module):
    def __init__(self, phone_count: int, mel_count: int = N_MELS):
        super().__init__()
        self.phone_embedding = nn.Embedding(phone_count + 1, 24, padding_idx=0)
        self.layers = nn.Sequential(
            nn.Linear(24 * 3 + 5, 192),
            nn.ReLU(),
            nn.LayerNorm(192),
            nn.Linear(192, 192),
            nn.ReLU(),
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, mel_count),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        phone_ids = features[:, :3].long()
        continuous = features[:, 3:]
        embedded = self.phone_embedding(phone_ids).flatten(1)
        return self.layers(torch.cat((embedded, continuous), dim=1))


def build_teacher_examples(
    text: str,
    segments: list[dict],
    mel: torch.Tensor,
    dictionary: dict[str, list[list[str]]],
    phone_to_id: dict[str, int],
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int], dict[str, float], int]:
    words = split_words(text)
    mapping = align_teacher_words(words, segments)
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    validation_flags: list[int] = []
    duration_observations: dict[str, list[int]] = defaultdict(list)
    used_words = 0

    for word_index, word in enumerate(words):
        segment = mapping.get(word_index)
        phones = lookup_pronunciation(word, dictionary)
        if not segment or not phones:
            continue
        start_frame = max(0, round(float(segment["start"]) * SAMPLE_RATE / HOP_LENGTH))
        end_frame = min(mel.shape[0], round(float(segment["end"]) * SAMPLE_RATE / HOP_LENGTH))
        frame_count = end_frame - start_frame
        if frame_count < len(phones) or start_frame >= end_frame:
            continue
        phone_durations = distributed_phone_frames(phones, frame_count)
        features = acoustic_features(phones, phone_durations, phone_to_id)
        if len(features) != frame_count:
            continue
        xs.append(torch.tensor(features, dtype=torch.float32))
        ys.append(mel[start_frame:end_frame])
        validation_flags.extend([1 if word_index % 5 == 0 else 0] * frame_count)
        for phone, duration in zip(phones, phone_durations):
            base, stress = phone_parts(phone)
            duration_observations[f"{base}:{stress}"].append(duration)
        used_words += 1

    if not xs:
        raise ValueError("No aligned word frames could be built from the teacher sample.")
    duration_means = {
        key: sum(values) / len(values)
        for key, values in duration_observations.items()
        if values
    }
    return xs, ys, validation_flags, duration_means, used_words


def save_training_plot(train_losses: list[float], validation_losses: list[float], path: Path) -> None:
    width, height, margin = 760, 360, 42
    values = train_losses + validation_losses
    upper = max(values) if values else 1.0
    upper = max(upper, 0.01)

    def line_points(series: list[float]) -> str:
        points = []
        for index, value in enumerate(series):
            x = margin + index * (width - 2 * margin) / max(1, len(series) - 1)
            y = height - margin - (value / upper) * (height - 2 * margin)
            points.append(f"{x:.1f},{y:.1f}")
        return " ".join(points)

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="24" text-anchor="middle" font-family="Arial" font-size="18">Student training loss</text>
<path d="M {margin} {margin} V {height - margin} H {width - margin}" fill="none" stroke="#555" stroke-width="1"/>
<polyline points="{line_points(train_losses)}" fill="none" stroke="#3678b8" stroke-width="2"/>
<polyline points="{line_points(validation_losses)}" fill="none" stroke="#df7b32" stroke-width="2"/>
<text x="{margin}" y="{height - 12}" font-family="Arial" font-size="12">Epoch</text>
<text x="{margin + 6}" y="{margin + 14}" font-family="Arial" font-size="12" fill="#3678b8">Train L1</text>
<text x="{margin + 84}" y="{margin + 14}" font-family="Arial" font-size="12" fill="#df7b32">Validation L1</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def train_student(
    output_dir: Path,
    epochs: int = 12,
    seed: int = 23,
) -> dict:
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    dictionary = read_dictionary()
    bases = sorted({phone_parts(phone)[0] for pronunciations in dictionary.values() for pron in pronunciations for phone in pron})
    phone_to_id = {phone: index + 1 for index, phone in enumerate(bases)}

    text = (ROOT / "data" / "demo_manifesto.txt").read_text(encoding="utf-8")
    teacher = json.loads(TEACHER_ALIGNMENT.read_text(encoding="utf-8"))
    audio = read_teacher_wave()
    mel = audio_log_mel(audio)
    xs, ys, validation_flags, duration_means, used_words = build_teacher_examples(
        text, teacher["segments"], mel, dictionary, phone_to_id
    )
    x = torch.cat(xs)
    y = torch.cat(ys)
    valid_mask = torch.tensor(validation_flags, dtype=torch.bool)
    train_mask = ~valid_mask
    if int(train_mask.sum()) < 64 or int(valid_mask.sum()) < 64:
        raise ValueError("The teacher sample is too small for the train/validation split.")

    mean = y[train_mask].mean(dim=0)
    std = y[train_mask].std(dim=0).clamp_min(0.05)
    y_scaled = (y - mean) / std
    train_loader = DataLoader(
        TensorDataset(x[train_mask], y_scaled[train_mask]),
        batch_size=512,
        shuffle=True,
    )
    valid_loader = DataLoader(
        TensorDataset(x[valid_mask], y_scaled[valid_mask]),
        batch_size=1024,
        shuffle=False,
    )
    model = AcousticStudent(len(phone_to_id))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    losses = {"train_l1": [], "validation_l1": []}

    for _epoch in range(epochs):
        model.train()
        train_total = 0.0
        train_count = 0
        for features, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features)
            loss = torch.nn.functional.l1_loss(prediction, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total += float(loss.detach()) * features.shape[0]
            train_count += features.shape[0]

        model.eval()
        validation_total = 0.0
        validation_count = 0
        with torch.no_grad():
            for features, targets in valid_loader:
                prediction = model(features)
                loss = torch.nn.functional.l1_loss(prediction, targets)
                validation_total += float(loss) * features.shape[0]
                validation_count += features.shape[0]
        losses["train_l1"].append(train_total / max(1, train_count))
        losses["validation_l1"].append(validation_total / max(1, validation_count))

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "student_model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "phone_to_id": phone_to_id,
        "mel_mean": mean,
        "mel_std": std,
        "duration_means": duration_means,
        "sample_rate": SAMPLE_RATE,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "n_mels": N_MELS,
        "teacher": "Fish Audio",
        "training_words": used_words,
    }, checkpoint_path)
    save_training_plot(losses["train_l1"], losses["validation_l1"], output_dir / "training_curve.svg")
    summary = {
        "student_model": "phoneme-conditioned acoustic MLP",
        "teacher_model": "Fish Audio",
        "training_audio": str(TEACHER_AUDIO.relative_to(ROOT)),
        "training_words": used_words,
        "training_frames": int(train_mask.sum()),
        "validation_frames": int(valid_mask.sum()),
        "epochs": epochs,
        "final_train_l1": round(losses["train_l1"][-1], 5),
        "final_validation_l1": round(losses["validation_l1"][-1], 5),
        "sample_rate_hz": SAMPLE_RATE,
        "training_curve": "training_curve.svg",
        "checkpoint": checkpoint_path.name,
        "scope": "single-script CPU proof of concept; not a general-purpose TTS model",
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    text_summary = "\n".join(f"{key}: {value}" for key, value in summary.items()) + "\n"
    (output_dir / "metrics.txt").write_text(text_summary, encoding="utf-8")
    return summary


def add_phone_features(
    phones: list[str],
    duration_means: dict[str, float],
    phone_to_id: dict[str, int],
) -> list[list[float]]:
    weights = []
    for phone in phones:
        base, stress = phone_parts(phone)
        observed = duration_means.get(f"{base}:{stress}")
        if observed is None:
            observed = duration_means.get(f"{base}:-1", 4.0)
        weights.append(max(1.0, observed))
    total_frames = max(len(phones), round(sum(weights)))
    durations = [max(1, round(weight)) for weight in weights]
    while sum(durations) > total_frames:
        candidates = [index for index, count in enumerate(durations) if count > 1]
        if not candidates:
            break
        durations[max(candidates, key=lambda index: durations[index] - weights[index])] -= 1
    while sum(durations) < total_frames:
        durations[max(range(len(durations)), key=lambda index: weights[index] - durations[index])] += 1
    return acoustic_features(phones, durations, phone_to_id)


def text_to_features(
    text: str,
    dictionary: dict[str, list[list[str]]],
    checkpoint: dict,
) -> torch.Tensor:
    phone_to_id = checkpoint["phone_to_id"]
    duration_means = checkpoint["duration_means"]
    all_features: list[list[float]] = []
    pause_frames = {",": 5, ";": 7, ":": 7, ".": 12, "?": 12, "!": 12, "…": 18}
    for match in TOKEN_PATTERN.finditer(text.replace("’", "'")):
        token = match.group()
        if token.startswith("["):
            cue = token[1:-1].strip().lower()
            if cue == "break":
                all_features.extend([[0.0] * 8 for _ in range(18)])
            elif cue == "long-break":
                all_features.extend([[0.0] * 8 for _ in range(36)])
            continue
        if token in pause_frames:
            all_features.extend([[0.0] * 8 for _ in range(pause_frames[token])])
            continue
        if not token[0].isalpha():
            continue
        phones = lookup_pronunciation(token, dictionary)
        if not phones:
            raise ValueError(f"The student model has no dictionary pronunciation for '{token}'.")
        all_features.extend(add_phone_features(phones, duration_means, phone_to_id))
    if not all_features:
        raise ValueError("Enter at least one word found in the pronunciation dictionary.")
    return torch.tensor(all_features, dtype=torch.float32)


def generate_student_audio(
    text: str,
    checkpoint_path: Path,
    output_path: Path,
    griffin_lim_iterations: int = 8,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = AcousticStudent(len(checkpoint["phone_to_id"]), checkpoint["n_mels"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    dictionary = read_dictionary()
    features = text_to_features(text, dictionary, checkpoint)
    with torch.no_grad():
        predicted = model(features)
    mel_mean = checkpoint["mel_mean"]
    mel_std = checkpoint["mel_std"]
    predicted_log_mel = predicted * mel_std + mel_mean
    pause_mask = features[:, 1] == 0
    predicted_log_mel[pause_mask] = 0.0
    predicted_mel = torch.expm1(predicted_log_mel).clamp_min(0.0) / 1000.0

    filters = mel_filterbank(checkpoint["sample_rate"])
    inverse_filters = torch.linalg.pinv(filters)
    magnitude = (inverse_filters @ predicted_mel.T).clamp_min(1e-6)
    magnitude = magnitude.clamp_max(magnitude.quantile(0.999))
    window = torch.hann_window(checkpoint["n_fft"])
    phase = torch.rand_like(magnitude) * (2.0 * math.pi)
    complex_spectrum = torch.polar(magnitude, phase)
    desired_length = max(HOP_LENGTH, (predicted_mel.shape[0] - 1) * checkpoint["hop_length"])
    waveform = torch.zeros(desired_length)
    for _ in range(max(1, griffin_lim_iterations)):
        waveform = torch.istft(
            complex_spectrum,
            n_fft=checkpoint["n_fft"],
            hop_length=checkpoint["hop_length"],
            win_length=checkpoint["n_fft"],
            window=window,
            center=True,
            length=desired_length,
        )
        estimate = torch.stft(
            waveform,
            n_fft=checkpoint["n_fft"],
            hop_length=checkpoint["hop_length"],
            win_length=checkpoint["n_fft"],
            window=window,
            center=True,
            return_complex=True,
        )
        phase = estimate / estimate.abs().clamp_min(1e-8)
        complex_spectrum = magnitude * phase
    waveform = waveform / waveform.abs().max().clamp_min(1e-6) * 0.95
    pcm = (waveform.clamp(-1, 1) * 32767).round().to(torch.int16).tolist()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(checkpoint["sample_rate"])
        audio_file.writeframes(struct.pack("<" + "h" * len(pcm), *pcm))
    return {
        "student_model": "phoneme-conditioned acoustic MLP",
        "speech_source": "trained student model",
        "sample_rate_hz": checkpoint["sample_rate"],
        "duration_seconds": round(len(pcm) / checkpoint["sample_rate"], 2),
        "wav_file": output_path.name,
    }
