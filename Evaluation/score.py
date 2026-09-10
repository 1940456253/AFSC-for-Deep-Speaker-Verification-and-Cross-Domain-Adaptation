"""Score explicit enrol-ID test-ID label trials from extracted NPZ embeddings."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .metrics import compute_metrics


def load_embeddings(path):
    with np.load(path, allow_pickle=False) as archive:
        ids = archive['ids'].astype(str)
        vectors = archive['embeddings'].astype(np.float64)
    if vectors.ndim != 2 or vectors.shape[0] != len(ids) or len(set(ids)) != len(ids):
        raise ValueError('Invalid embedding matrix or duplicate IDs')
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or (norms < 1e-12).any():
        raise ValueError('Non-finite or zero embeddings')
    return dict(zip(ids, vectors / norms))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--enrol', required=True)
    p.add_argument('--test', help='Defaults to the enrol NPZ')
    p.add_argument('--trials', required=True)
    p.add_argument('--output', required=True, help='Output directory')
    p.add_argument('--p-target', type=float, default=0.01)
    p.add_argument('--c-miss', type=float, default=1.0)
    p.add_argument('--c-fa', type=float, default=1.0)
    args = p.parse_args()
    enrol = load_embeddings(args.enrol)
    test = load_embeddings(args.test) if args.test else enrol
    records, scores, labels = [], [], []
    mapping = {'1': 1, 'target': 1, '0': 0, 'nontarget': 0}
    with open(args.trials, encoding='utf-8') as f:
        for number, line in enumerate(f, 1):
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            fields = line.split()
            if len(fields) != 3 or fields[2] not in mapping:
                raise ValueError(f'Trial line {number}: expected enrol_id test_id 0/1/target/nontarget')
            a, b, label = fields
            if a not in enrol or b not in test:
                raise KeyError(f'Trial line {number}: missing embedding for {a} or {b}')
            score = float(np.clip(enrol[a] @ test[b], -1, 1))
            scores.append(score); labels.append(mapping[label])
            records.append([a, b, mapping[label], score])
    metrics = compute_metrics(scores, labels, args.p_target, args.c_miss, args.c_fa)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'scores.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['enrol_id', 'test_id', 'label', 'score'])
        writer.writerows(records)
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
