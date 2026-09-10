"""Compare two waveforms. A same-speaker decision requires a calibrated threshold."""
import argparse
import json
import torch
from Dataset.data import load_audio
from .extract import load_system, embed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--enrol', required=True)
    p.add_argument('--test', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threshold', type=float, help='Choose on separate development trials')
    args = p.parse_args()
    device = torch.device(args.device)
    model, extractor, _ = load_system(args.checkpoint, device)
    a = embed(load_audio({'ID': 'enrol', 'path': args.enrol}), model, extractor, device)
    b = embed(load_audio({'ID': 'test', 'path': args.test}), model, extractor, device)
    score = float(a @ b)
    result = {'cosine_similarity': score}
    if args.threshold is not None:
        result.update(threshold=args.threshold, same_speaker=score >= args.threshold)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
