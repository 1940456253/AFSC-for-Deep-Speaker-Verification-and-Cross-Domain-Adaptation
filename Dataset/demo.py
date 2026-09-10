"""Generate reproducible synthetic tones for software tests, not speaker research."""
import argparse
import csv
from pathlib import Path
import numpy as np
import soundfile as sf


def generate(output):
    out = Path(output)
    (out / 'audio').mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1234)
    t = np.arange(16000, dtype=np.float64) / 16000
    manifests = {}
    for split, identities in [('train', range(4)), ('eval', range(4, 7))]:
        rows = []
        for speaker in identities:
            for utterance in range(3):
                key = f's{speaker}_u{utterance}'
                base = 160 + 45 * speaker
                signal = sum(0.1 / k * np.sin(2 * np.pi * base * k * t + rng.uniform(0, 1)) for k in range(1, 5))
                signal += rng.normal(0, 0.004, len(t))
                path = out / 'audio' / f'{key}.wav'
                sf.write(path, signal, 16000, subtype='PCM_16')
                rows.append({'ID': key, 'path': f'audio/{key}.wav', 'spk': f's{speaker}'})
        with (out / f'{split}.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['ID', 'path', 'spk']); w.writeheader(); w.writerows(rows)
        manifests[split] = rows
    trials = []
    rows = manifests['eval']
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            trials.append(f"{a['ID']} {b['ID']} {int(a['spk'] == b['spk'])}")
    (out / 'trials.txt').write_text('\n'.join(trials) + '\n')
    print(f'Synthetic demo written to {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='demo_data')
    generate(p.parse_args().output)
