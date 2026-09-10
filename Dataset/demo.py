"""Build train/eval manifests and trials from a directory of speaker folders.

Each subdirectory under the source directory is treated as one speaker;
the audio files inside it are that speaker's utterances. This does not
generate synthetic audio and does not copy the source files.

Default layout (relative to the project root):

    Speech_example/
        speech_demo1/  a1.wav a2.wav ...   (one speaker)
        speech_demo2/  b1.wav b2.wav ...
        speech_demo3/  c1.wav c2.wav ...
"""
import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SOURCE = PROJECT_ROOT / 'Speech_example'
DEFAULT_OUTPUT = PROJECT_ROOT / 'demo_data'

AUDIO_EXTENSIONS = {'.wav', '.flac', '.ogg', '.aiff', '.aif', '.au', '.sf'}


def list_audio(directory):
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)


def generate(source, output, eval_per_speaker):
    source = Path(source).expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(f'{source} is not a directory. '
                                 f'Place speech_demo1/speech_demo2/speech_demo3 under {DEFAULT_SOURCE}, '
                                 f'or pass --source /path/to/Speech_example.')

    out = Path(output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    speaker_dirs = sorted(p for p in source.iterdir() if p.is_dir())
    if len(speaker_dirs) < 2:
        raise ValueError(f'{source}: need at least two speaker subdirectories')

    train_rows, eval_rows = [], []
    for speaker_dir in speaker_dirs:
        spk_id = speaker_dir.name
        files = list_audio(speaker_dir)
        if not files:
            raise ValueError(f'{speaker_dir}: no audio files found')
        if len(files) <= eval_per_speaker:
            raise ValueError(f'{speaker_dir}: need more than {eval_per_speaker} audio files, '
                             f'found {len(files)}')

        split = len(files) - eval_per_speaker
        for i, path in enumerate(files):
            row = {'ID': f'{spk_id}_{i:02d}', 'path': str(path), 'spk': spk_id}
            (train_rows if i < split else eval_rows).append(row)

    for name, rows in (('train.csv', train_rows), ('eval.csv', eval_rows)):
        with (out / name).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['ID', 'path', 'spk'])
            writer.writeheader()
            writer.writerows(rows)

    trials = []
    for i, a in enumerate(eval_rows):
        for b in eval_rows[i + 1:]:
            trials.append(f"{a['ID']} {b['ID']} {int(a['spk'] == b['spk'])}")
    (out / 'trials.txt').write_text('\n'.join(trials) + '\n')

    n_target = sum(1 for t in trials if t.endswith(' 1'))
    n_speakers_train = len({r['spk'] for r in train_rows})
    n_speakers_eval = len({r['spk'] for r in eval_rows})
    print(f'Train:  {len(train_rows)} utterances from {n_speakers_train} speakers')
    print(f'Eval:   {len(eval_rows)} utterances from {n_speakers_eval} speakers')
    print(f'Trials: {len(trials)} ({n_target} target, {len(trials) - n_target} nontarget)')
    print(f'Manifests written to {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default=str(DEFAULT_SOURCE), help=f'Directory containing one subdirectory per speaker '
                                                                 f'(default: {DEFAULT_SOURCE})')
    p.add_argument('--output', default=str(DEFAULT_OUTPUT), help=f'Output directory (default: {DEFAULT_OUTPUT})')
    p.add_argument('--eval-per-speaker', type=int, default=5, help='Utterances per speaker held out for evaluation (default: 5)')
    args = p.parse_args()
    generate(args.source, args.output, args.eval_per_speaker)
