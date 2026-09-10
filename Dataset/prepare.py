"""Convert wav.scp + utt2spk to one CSV row per utterance (no repeated chunks)."""
import argparse
import csv
from pathlib import Path
import soundfile as sf


def read_mapping(path):
    values = {}
    with Path(path).open(encoding='utf-8-sig') as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            fields = line.strip().split(maxsplit=1)
            if len(fields) != 2 or fields[0] in values:
                raise ValueError(f'{path}:{number}: invalid or duplicate entry')
            values[fields[0]] = fields[1]
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wav-scp', required=True)
    p.add_argument('--utt2spk', help='Required for training, optional for evaluation')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    wavs = read_mapping(args.wav_scp)
    speakers = read_mapping(args.utt2spk) if args.utt2spk else {}
    if speakers and set(speakers) != set(wavs):
        raise ValueError('wav.scp and utt2spk IDs must match exactly')
    if not wavs:
        raise ValueError('No utterances')
    rows = []
    for key, value in sorted(wavs.items()):
        path = Path(value)
        if not path.is_absolute():
            path = Path(args.wav_scp).resolve().parent / path
        info = sf.info(path)
        if info.samplerate != 16000 or info.frames == 0:
            raise ValueError(f'{path}: expected nonempty 16 kHz audio')
        rows.append({'ID': key, 'path': str(path.resolve()), 'spk': speakers.get(key, ''),
                     'dur': info.frames / info.samplerate})
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['ID', 'path', 'spk', 'dur'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} utterances to {out}')


if __name__ == '__main__':
    main()
