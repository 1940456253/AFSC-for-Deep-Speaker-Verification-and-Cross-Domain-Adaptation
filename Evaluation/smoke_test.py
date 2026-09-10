"""End-to-end CPU check with synthetic audio and a tiny ECAPA.
Run from the repository root: python -m Evaluation.smoke_test
"""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import torch
import yaml


def main():
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    def run(module, *args):
        subprocess.run([sys.executable, '-m', module, *map(str, args)], cwd=root, env=env, check=True)
    with tempfile.TemporaryDirectory(prefix='afsc_smoke_') as folder:
        temp = Path(folder)
        data = temp / 'data'
        run('Dataset.demo', '--output', data)
        rows = list(csv.DictReader((data / 'train.csv').open()))
        (data / 'wav.scp').write_text(''.join(f"{r['ID']} {r['path']}\n" for r in rows))
        (data / 'utt2spk').write_text(''.join(f"{r['ID']} {r['spk']}\n" for r in rows))
        run('Dataset.prepare', '--wav-scp', data / 'wav.scp', '--utt2spk', data / 'utt2spk', '--output', data / 'prepared.csv')
        common = ['--config', root / 'Model/demo.yaml', '--train-csv', data / 'prepared.csv', '--device', 'cpu']
        uninterrupted, resumed = temp / 'full', temp / 'resumed'
        run('Model.train', *common, '--output', uninterrupted)
        run('Model.train', *common, '--output', resumed, '--stop-after-epoch', 1)
        run('Model.train', *common, '--output', resumed, '--resume', resumed / 'last.pt')
        a = torch.load(uninterrupted / 'last.pt', weights_only=True)
        b = torch.load(resumed / 'last.pt', weights_only=True)
        assert all(torch.equal(a['model'][k], b['model'][k]) for k in a['model']), 'Resume model mismatch'
        assert all(torch.equal(a['classifier'][k], b['classifier'][k]) for k in a['classifier']), 'Resume classifier mismatch'
        first = torch.load(uninterrupted / 'epoch_001.pt', weights_only=True)
        assert torch.equal(first['model']['frontend.raw_gaps'], a['model']['frontend.raw_gaps']), 'Frozen AFSC changed'
        run('Inference.extract', '--checkpoint', uninterrupted / 'last.pt', '--csv', data / 'eval.csv', '--output', temp / 'embeddings.npz', '--device', 'cpu')
        run('Evaluation.score', '--enrol', temp / 'embeddings.npz', '--trials', data / 'trials.txt', '--output', temp / 'scores')
        metrics = json.loads((temp / 'scores/metrics.json').read_text())
        assert metrics['target_trials'] == 9 and metrics['nontarget_trials'] == 27
        run('Inference.verify', '--checkpoint', uninterrupted / 'last.pt', '--enrol', data / 'audio/s4_u0.wav', '--test', data / 'audio/s4_u1.wav')
        run('Features.export', '--checkpoint', uninterrupted / 'last.pt', '--output', temp / 'filters')
        for feature in ['fbank', 'mfcc']:
            output = temp / feature
            run('Model.train', *common, '--feature', feature, '--output', output)
            run('Inference.extract', '--checkpoint', output / 'last.pt', '--csv', data / 'eval.csv', '--output', temp / f'{feature}.npz', '--device', 'cpu')
        cfg = yaml.safe_load((root / 'Model/demo.yaml').read_text())
        cfg['reset_optimizer_at_freeze'] = False
        config = temp / 'preserve.yaml'; config.write_text(yaml.safe_dump(cfg))
        common2 = ['--config', config, '--train-csv', data / 'prepared.csv', '--device', 'cpu']
        run('Model.train', *common2, '--output', temp / 'preserve')
        run('Model.train', *common2, '--output', temp / 'preserve', '--resume', temp / 'preserve/last.pt')
    print('PASS: prepare, AFSC train/freeze, exact epoch resume, extraction, scoring, pair inference, filter export, MFCC and FBank.')


if __name__ == '__main__':
    main()
