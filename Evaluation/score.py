"""对显式的 enrol-ID / test-ID / label 三元组评分。"""
import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from Evaluation.metrics import compute_metrics


def default_paths():
    """选择默认的 enrol NPZ、trials 文件和输出目录。

      (A) 用户自己训练的模型  ——  当前默认启用
      (B) 论文预训练模型      ——  已注释，需要时取消注释
    """
    # ========== (A) 用户自己训练的模型（当前默认） ==========
    return (
        PROJECT_ROOT / 'runs' / 'demo' / 'embeddings.npz',    # enrol NPZ
        PROJECT_ROOT / 'runs' / 'demo' / 'scores',            # 输出目录
    )

    # ========== (B) 论文预训练模型（需要时启用） ==========
    # pretrained = PROJECT_ROOT / 'runs' / 'pretrained_eval.npz'
    # if pretrained.is_file():
    #     return pretrained, PROJECT_ROOT / 'runs' / 'pretrained_scores'
    # return PROJECT_ROOT / 'runs' / 'demo' / 'embeddings.npz', PROJECT_ROOT / 'runs' / 'demo' / 'scores'


def load_embeddings(path):
    """读 NPZ 并返回 {id: 归一化后的向量}。"""
    with np.load(path, allow_pickle=False) as archive:
        ids = archive['ids'].astype(str)
        vectors = archive['embeddings'].astype(np.float64)
    if vectors.ndim != 2 or vectors.shape[0] != len(ids) or len(set(ids)) != len(ids):
        raise ValueError('嵌入矩阵无效或 ID 重复')
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or (norms < 1e-12).any():
        raise ValueError('嵌入包含非有限值或零向量')
    return dict(zip(ids, vectors / norms))


def main():
    default_enrol, default_out = default_paths()
    default_trials = PROJECT_ROOT / 'demo_data' / 'trials.txt'

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--enrol', default=str(default_enrol),
                   help=f'enrol 嵌入 NPZ（默认: {default_enrol}）')
    p.add_argument('--test', help='test 嵌入 NPZ；默认与 enrol 相同')
    p.add_argument('--trials', default=str(default_trials),
                   help=f'trials 文件（默认: {default_trials}）')
    p.add_argument('--output', default=str(default_out),
                   help=f'输出目录（默认: {default_out}）')
    p.add_argument('--p-target', type=float, default=0.01)
    p.add_argument('--c-miss', type=float, default=1.0)
    p.add_argument('--c-fa', type=float, default=1.0)
    args = p.parse_args()

    enrol_path = Path(args.enrol).expanduser().resolve()
    trials_path = Path(args.trials).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    test_path = Path(args.test).expanduser().resolve() if args.test else None

    if not enrol_path.is_file():
        raise FileNotFoundError(
            f'找不到 enrol NPZ: {enrol_path}\n'
            f'请先运行 python -m Inference.extract 提取嵌入。'
        )
    if test_path is not None and not test_path.is_file():
        raise FileNotFoundError(f'找不到 test NPZ: {test_path}')
    if not trials_path.is_file():
        raise FileNotFoundError(
            f'找不到 trials 文件: {trials_path}\n'
            f'请先运行 python -m Dataset.demo 生成。'
        )

    enrol = load_embeddings(enrol_path)
    test = load_embeddings(test_path) if test_path else enrol
    records, scores, labels = [], [], []
    # 支持 0/1 和 nontarget/target 两种标签写法
    mapping = {'1': 1, 'target': 1, '0': 0, 'nontarget': 0}
    with trials_path.open(encoding='utf-8') as f:
        for number, line in enumerate(f, 1):
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            fields = line.split()
            if len(fields) != 3 or fields[2] not in mapping:
                raise ValueError(f'第 {number} 行: 期望 enrol_id test_id 0/1/target/nontarget')
            a, b, label = fields
            if a not in enrol or b not in test:
                raise KeyError(f'第 {number} 行: 缺少 {a} 或 {b} 的嵌入')
            score = float(np.clip(enrol[a] @ test[b], -1, 1))
            scores.append(score)
            labels.append(mapping[label])
            records.append([a, b, mapping[label], score])

    if not scores:
        raise ValueError(f'{trials_path}: 没有有效的 trial')

    metrics = compute_metrics(scores, labels, args.p_target, args.c_miss, args.c_fa)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'scores.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['enrol_id', 'test_id', 'label', 'score'])
        writer.writerows(records)
    (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f'评分结果已写入 {output_dir}')


if __name__ == '__main__':
    main()