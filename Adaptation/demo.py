"""AFSC-ECAPA 跨域自适应演示。

这是一个小规模演示：
  - 用第一部分训练好的 AFSC-ECAPA checkpoint
  - 把 demo 数据当作“目标域”
  - 走论文 3.5 节的 residual adapter + 标准化 + 0.5/0.5 fusion

真实 SITW 实验请改用 Adaptation.run（待补）。
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Dataset.data import read_manifest
from Evaluation.metrics import compute_metrics
from Adaptation.adapter import WideResidualAdapter
from Adaptation.dataset import EmbeddingDataset
from Adaptation.embed import load_system_from_checkpoint, embed_manifest
from Adaptation.losses import supcon_loss
from Adaptation.sampler import PKBatchSampler
from Adaptation.standardize import fit_standardizer, apply_standardizer
from Adaptation.trials import controlled_pairs, scores_from_embeddings


def default_paths():
    return (PROJECT_ROOT / 'runs' / 'demo' / 'last.pt', PROJECT_ROOT / 'demo_data' / 'train.csv', PROJECT_ROOT / 'demo_data' / 'eval.csv', PROJECT_ROOT / 'runs' / 'demo_adapt',)


def main():
    ckpt_d, train_d, eval_d, out_d = default_paths()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default=str(ckpt_d))
    p.add_argument('--train-csv', default=str(train_d))
    p.add_argument('--eval-csv', default=str(eval_d))
    p.add_argument('--output', default=str(out_d))
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--adapter-epochs', type=int, default=10)
    p.add_argument('--chunk-seconds', type=float, default=None, help='None 表示整段处理；真实跨域实验建议 30.0')
    p.add_argument('--seed', type=int, default=1234)
    args = p.parse_args()

    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    model, extractor, cfg = load_system_from_checkpoint(args.checkpoint, device)
    for param in model.parameters():
        param.requires_grad = False
    print(f'feature={cfg["feature"]}  embedding_dim={cfg["embedding_dim"]}')

    train_rows = read_manifest(args.train_csv, require_speakers=True)
    eval_rows = read_manifest(args.eval_csv, require_speakers=True)
    print(f'train: {len(train_rows)} utts, {len(set(r["spk"] for r in train_rows))} spks')
    print(f'eval : {len(eval_rows)} utts, {len(set(r["spk"] for r in eval_rows))} spks')

    chunk = args.chunk_seconds
    print('提取训练 embeddings ...')
    train_emb = embed_manifest(train_rows, model, extractor, device, chunk_seconds=chunk, label='train ')
    print('提取评估 embeddings ...')
    eval_emb = embed_manifest(eval_rows, model, extractor, device, chunk_seconds=chunk, label='eval  ')
    train_labels = np.array([r['spk'] for r in train_rows])
    eval_labels = np.array([r['spk'] for r in eval_rows])

    # 目标域标准化（论文 Eq.19）
    mu, sd = fit_standardizer(train_emb)
    eval_z = apply_standardizer(eval_emb, mu, sd)

    # 受控 trial
    target_pairs, non_target_pairs = controlled_pairs(eval_labels, nontargets_per_anchor=20, max_target_pairs_per_speaker=0, seed=20260705, )
    print(f'trials: {len(target_pairs)} target, {len(non_target_pairs)} non-target')

    # 基线
    base_scores, labels = scores_from_embeddings(eval_emb, target_pairs, non_target_pairs)
    base_metrics = compute_metrics(base_scores, labels)

    # 标准化后的分数
    z_scores, _ = scores_from_embeddings(eval_z, target_pairs, non_target_pairs)
    z_metrics = compute_metrics(z_scores, labels)

    # 适配器训练
    adapter = WideResidualAdapter(dim=cfg['embedding_dim'], hidden_dim=384, dropout=0.40, ).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = EmbeddingDataset(train_emb, train_labels)

    # 自适应 P / K / batches_per_epoch，兼容 demo 小数据
    n_speakers = len(set(map(str, train_labels)))
    p = min(8, max(2, n_speakers))
    k = 4
    batches_per_epoch = max(10, min(100, (len(train_rows) // (p * k)) * 10))
    print(f'P={p} K={k} batches_per_epoch={batches_per_epoch}')

    sampler = PKBatchSampler(ds.y.numpy(), p=p, k=k, batches_per_epoch=batches_per_epoch, seed=args.seed)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=0)

    history = []
    for epoch in range(1, args.adapter_epochs + 1):
        adapter.train()
        total_loss = total_sup = total_id = total_n = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            adapted = adapter(x)
            l_sup = supcon_loss(adapted, y, temperature=0.07)
            l_id = F.mse_loss(adapted, x)
            loss = l_sup + 0.05 * l_id
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
            optimizer.step()
            n = len(x)
            total_loss += float(loss.item()) * n
            total_sup += float(l_sup.item()) * n
            total_id += float(l_id.item()) * n
            total_n += n
        row = {'epoch': epoch, 'loss': total_loss / total_n, 'supcon': total_sup / total_n, 'identity': total_id / total_n, 'alpha': float(adapter.alpha.detach().cpu().item())}
        history.append(row)
        print(f'epoch={epoch} loss={row["loss"]:.4f} alpha={row["alpha"]:.4f}', flush=True)

    # 适配器分数
    adapter.eval()
    with torch.no_grad():
        x_eval = torch.as_tensor(eval_emb, dtype=torch.float32, device=device)
        adapted_eval = F.normalize(adapter(x_eval), p=2, dim=1).cpu().numpy()
    adapter_scores, _ = scores_from_embeddings(adapted_eval, target_pairs, non_target_pairs)
    adapter_metrics = compute_metrics(adapter_scores, labels)

    # 固定 0.5/0.5 fusion（论文 Eq.20）
    fusion_scores = 0.5 * adapter_scores + 0.5 * z_scores
    fusion_metrics = compute_metrics(fusion_scores, labels)

    summary = {'checkpoint': str(args.checkpoint), 'feature': cfg['feature'], 'embedding_dim': cfg['embedding_dim'], 'chunk_seconds': chunk, 'adapter_epochs': args.adapter_epochs,
               'baseline': base_metrics, 'standardized': z_metrics, 'adapter': adapter_metrics, 'fusion': fusion_metrics, }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    (out / 'history.json').write_text(json.dumps(history, indent=2))
    print()
    print(json.dumps(summary, indent=2))
    print(f'结果已写入 {out}')


if __name__ == '__main__':
    main()
