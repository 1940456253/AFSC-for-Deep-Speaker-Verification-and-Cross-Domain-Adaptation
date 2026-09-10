"""Audio manifests and utterance-based random training crops."""
import csv
import random
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


def read_manifest(path, require_speakers=False):
    path = Path(path).resolve()
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'ID', 'path'} | ({'spk'} if require_speakers else set())
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: required columns {sorted(required)}')
        rows = list(reader)
    seen = set()
    for row in rows:
        if not row['ID'] or row['ID'] in seen or any(c.isspace() for c in row['ID']):
            raise ValueError(f"Invalid/duplicate utterance ID: {row['ID']!r}")
        seen.add(row['ID'])
        audio_path = Path(row['path'])
        if not audio_path.is_absolute():
            audio_path = path.parent / audio_path
        row['path'] = str(audio_path.resolve())
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        if require_speakers and not row.get('spk'):
            raise ValueError(f"Missing speaker: {row['ID']}")
    if not rows:
        raise ValueError('Empty manifest')
    return rows


def load_audio(row, sample_rate=16000):
    """Use channel zero, enforce sample rate, optionally honor CSV start/stop seconds."""
    wav, sr = sf.read(row['path'], dtype='float32', always_2d=True)
    if sr != sample_rate:
        raise ValueError(f"{row['path']}: {sr} Hz; expected {sample_rate}. Resample first.")
    wav = wav[:, 0].copy()
    start = float(row.get('start') or 0)
    stop = float(row.get('stop') or len(wav) / sr)
    if not (0 <= start < stop <= len(wav) / sr + 1 / sr):
        raise ValueError(f"Invalid segment for {row['ID']}: {start}, {stop}")
    wav = wav[round(start * sr):round(stop * sr)]
    if not len(wav) or not np.isfinite(wav).all():
        raise ValueError(f"Empty or non-finite audio: {row['ID']}")
    return torch.from_numpy(wav)


class SpeakerDataset(Dataset):
    def __init__(self, manifest, extractor, duration=3.0, sample_rate=16000):
        self.rows = read_manifest(manifest, require_speakers=True)
        self.speakers = {s: i for i, s in enumerate(sorted({r['spk'] for r in self.rows}))}
        self.extractor = extractor
        self.sample_rate = sample_rate
        self.samples = round(duration * sample_rate)
        if self.samples < 400:
            raise ValueError('Training crop must be at least 25 ms')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        wav = load_audio(row, self.sample_rate)
        if wav.numel() > self.samples:
            start = random.randint(0, wav.numel() - self.samples)
            wav = wav[start:start + self.samples]
        elif wav.numel() < self.samples:
            wav = torch.nn.functional.pad(wav, (0, self.samples - wav.numel()))
        return self.extractor(wav), self.speakers[row['spk']]
