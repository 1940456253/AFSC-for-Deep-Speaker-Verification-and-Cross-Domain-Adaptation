"""适配器训练 + AFSC gap 更新。

提供两组函数：
  - train_adapter_epochs     : 用预计算 embedding 训练 adapter（快）
  - gap_update_stage         : AFSC gap 更新 + adapter 联合训练（慢）

以及一个端到端入口 main()，用法：
    python -m Adaptation.train --device cpu
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from Dataset.data import read_manifest, load_audio
from Evaluation.metrics import compute_metrics
from Adaptation.adapter import WideResidualAdapter
from Adaptation.dataset import EmbeddingDataset
from Adaptation.embed import load_system_from_checkpoint, embed_manifest
from Adaptation.losses import supcon_loss
from Adaptation.sampler import PKBatchSampler
from Adaptation.standardize import fit_standardizer, apply_standardizer
from Adaptation.trials import controlled_pairs, scores_from_embeddings


# ----------------------------------------------------------------------
# 通用：优化器
# ----------------------------------------------------------------------

def build_optimizer(adapter, raw_gap_params, adapter_lr=1e-3, adapter_wd=1e-4, afsc_filter_lr=3e-2):
    """构造 AdamW，可选 AFSC gap 参数组。

    raw_gap_params 为空时只有一个 adapter 参数组。
    """
    param_groups = [{'params': list(adapter.parameters()), 'lr': adapter_lr, 'weight_decay': adapter_wd, }]
    if raw_gap_params:
        param_groups.append({'params': list(raw_gap_params), 'lr': afsc_filter_lr, 'weight_decay': 0.0, })
    return torch.optim.AdamW(param_groups)


def _make_loader(embeddings, labels, p, k, batches_per_epoch, seed):
    dataset = EmbeddingDataset(embeddings, labels)
    sampler = PKBatchSampler(dataset.y.numpy(), p, k, batches_per_epoch, seed=seed)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=0)


# ----------------------------------------------------------------------
# 阶段 2：用预计算 embedding 训练 adapter
# ----------------------------------------------------------------------

def train_adapter_epochs(adapter, optimizer, train_emb, train_labels, epochs, device, p=8, k=4, batches_per_epoch=100, seed=0, log_prefix=''):
    """用预计算 embedding 训练适配器，返回 history 列表。"""
    loader = _make_loader(train_emb, train_labels, p, k, batches_per_epoch, seed)
    history = []
    for epoch in range(1, epochs + 1):
        adapter.train()
        total_loss = total_supcon = total_id = total_n = 0.0
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
            total_supcon += float(l_sup.item()) * n
            total_id += float(l_id.item()) * n
            total_n += n
        row = {'epoch': epoch, 'loss': total_loss / total_n, 'supcon': total_supcon / total_n, 'identity': total_id / total_n, 'alpha': float(adapter.alpha.detach().cpu().item()), }
        history.append(row)
        print(f'{log_prefix}epoch={epoch} loss={row["loss"]:.4f} '
              f'alpha={row["alpha"]:.4f}', flush=True)
    return history


# ----------------------------------------------------------------------
# 阶段 1：AFSC gap 更新（只对 afsc 特征）
# ----------------------------------------------------------------------

class AFSCGapDataset(Dataset):
    """AFSC gap 更新用：从波形裁 3 秒，走 frontend 再进 encoder。"""

    def __init__(self, rows, extractor, crop_seconds=3.0, seed=0):
        self.rows = rows
        self.extractor = extractor
        self.crop_len = int(round(crop_seconds * 16000))
        self.seed = int(seed)
        speakers = sorted({r['spk'] for r in rows})
        self.label_to_id = {s: i for i, s in enumerate(speakers)}
        self.epoch = 0

    def set_epoch(self, e):
        self.epoch = int(e)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        wav = load_audio(row)
        if wav.numel() < self.crop_len:
            wav = F.pad(wav, (0, self.crop_len - wav.numel()))
        elif wav.numel() > self.crop_len:
            rng = np.random.default_rng(self.seed + 1000003 * self.epoch + 9176 * i)
            start = int(rng.integers(0, wav.numel() - self.crop_len + 1))
            wav = wav[start:start + self.crop_len]
        feat = self.extractor(wav)
        return feat, self.label_to_id[row['spk']]


def train_gap_and_adapter_epoch(encoder, adapter, loader, optimizer, device, identity_lambda=0.05, grad_clip=5.0):
    """AFSC gap 更新 epoch。

    关键点：
      - encoder.eval()，固定 BN running stats
      - 只有 raw_gaps 和 adapter 可训练
      - identity loss 对 detached base 计算，避免把 filter 拉向 identity
    """
    encoder.eval()
    adapter.train()
    total_loss = total_sup = total_id = total_n = 0.0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        base = encoder(features)
        if isinstance(base, (tuple, list)):
            base = base[0]
        base = F.normalize(base, p=2, dim=1)
        adapted = adapter(base)
        l_sup = supcon_loss(adapted, labels, temperature=0.07)
        detached = base.detach()
        l_id = F.mse_loss(adapter(detached), detached)
        loss = l_sup + identity_lambda * l_id
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        trainable = [p for g in optimizer.param_groups for p in g['params'] if p.requires_grad and p.grad is not None]
        if trainable:
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step()
        n = len(features)
        total_loss += float(loss.item()) * n
        total_sup += float(l_sup.item()) * n
        total_id += float(l_id.item()) * n
        total_n += n
    return total_loss / total_n, total_sup / total_n, total_id / total_n


def gap_update_stage(model, adapter, rows, extractor, device, num_epochs, seed=0, p=8, k=4, batches_per_epoch=100):
    """执行若干个 AFSC gap 更新 epoch，返回 history。"""
    raw_gap_params = []
    for name, param in model.named_parameters():
        if 'raw_gaps' in name.lower():
            raw_gap_params.append(param)
    if not raw_gap_params:
        raise RuntimeError('No raw_gaps parameter found in model')

    for param in model.parameters():
        param.requires_grad = False
    for param in raw_gap_params:
        param.requires_grad = True

    optimizer = build_optimizer(adapter, raw_gap_params)
    dataset = AFSCGapDataset(rows, extractor, crop_seconds=3.0, seed=seed)

    # 自适应 P，兼容小数据
    n_speakers = len(dataset.label_to_id)
    p = min(p, max(2, n_speakers))
    labels = np.array([dataset.label_to_id[r['spk']] for r in rows])
    sampler = PKBatchSampler(labels, p, k, batches_per_epoch, seed=seed)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)

    history = []
    for epoch in range(1, num_epochs + 1):
        dataset.set_epoch(epoch)
        loss, sup, ident = train_gap_and_adapter_epoch(model, adapter, loader, optimizer, device)
        row = {'epoch': epoch, 'loss': loss, 'supcon': sup, 'identity': ident, 'alpha': float(adapter.alpha.detach().cpu().item())}
        history.append(row)
        print(f'gap-update epoch={epoch} loss={loss:.4f} '
              f'alpha={row["alpha"]:.4f}', flush=True)
    return history


# ----------------------------------------------------------------------
# 端到端入口
# ----------------------------------------------------------------------

def main():
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default=str(root / 'runs' / 'demo' / 'last.pt'))
    p.add_argument('--train-csv', default=str(root / 'demo_data' / 'train.csv'))
    p.add_argument('--eval-csv', default=str(root / 'demo_data' / 'eval.csv'))
    p.add_argument('--output', default=str(root / 'runs' / 'demo_train'))
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--gap-epochs', type=int, default=None, help='AFSC gap 更新 epoch；None 时按 feature 自动（afsc=2，其他=0）')
    p.add_argument('--adapter-epochs', type=int, default=10)
    p.add_argument('--chunk-seconds', type=float, default=None, help='None 表示整段处理；真实跨域实验建议 30.0')
    p.add_argument('--seed', type=int, default=1234)
    args = p.parse_args()

    device = torch.device(args.device)
    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    model, extractor, cfg = load_system_from_checkpoint(args.checkpoint, device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    print(f'feature={cfg["feature"]} embedding_dim={cfg["embedding_dim"]}')

    train_rows = read_manifest(args.train_csv, require_speakers=True)
    eval_rows = read_manifest(args.eval_csv, require_speakers=True)
    train_labels = np.array([r['spk'] for r in train_rows])
    eval_labels = np.array([r['spk'] for r in eval_rows])
    print(f'train: {len(train_rows)} utts, {len(set(train_labels))} spks')
    print(f'eval : {len(eval_rows)} utts, {len(set(eval_labels))} spks')

    # 自适应 P/K/batches，兼容 demo 小数据
    n_speakers = len(set(map(str, train_labels)))
    p_pk = min(8, max(2, n_speakers))
    k_pk = 4
    batches = max(10, min(100, (len(train_rows) // (p_pk * k_pk)) * 10))
    print(f'P={p_pk} K={k_pk} batches_per_epoch={batches}')

    gap_epochs = args.gap_epochs
    if gap_epochs is None:
        gap_epochs = 2 if cfg['feature'] == 'afsc' else 0

    adapter = WideResidualAdapter(dim=cfg['embedding_dim'], hidden_dim=384, dropout=0.40, ).to(device)

    # 阶段 1：AFSC gap 更新（可选）
    gap_history = []
    if gap_epochs > 0:
        if cfg['feature'] != 'afsc':
            raise ValueError(f'gap 更新需要 feature=afsc，得到 {cfg["feature"]}')
        print(f'AFSC gap 更新 {gap_epochs} epoch ...')
        gap_history = gap_update_stage(model, adapter, train_rows, extractor, device, num_epochs=gap_epochs, seed=args.seed, p=p_pk, k=k_pk, batches_per_epoch=batches, )

    # 阶段 2：提取 embedding
    print('提取训练 embeddings ...')
    train_emb = embed_manifest(train_rows, model, extractor, device, chunk_seconds=args.chunk_seconds, label='train ')
    print('提取评估 embeddings ...')
    eval_emb = embed_manifest(eval_rows, model, extractor, device, chunk_seconds=args.chunk_seconds, label='eval  ')

    mu, sd = fit_standardizer(train_emb)
    eval_z = apply_standardizer(eval_emb, mu, sd)

    target_pairs, non_target_pairs = controlled_pairs(eval_labels, nontargets_per_anchor=20, max_target_pairs_per_speaker=0, seed=20260705, )
    print(f'trials: {len(target_pairs)} target, {len(non_target_pairs)} non-target')

    base_scores, labels = scores_from_embeddings(eval_emb, target_pairs, non_target_pairs)
    base_metrics = compute_metrics(base_scores, labels)
    z_scores, _ = scores_from_embeddings(eval_z, target_pairs, non_target_pairs)
    z_metrics = compute_metrics(z_scores, labels)

    # 阶段 3：adapter 训练
    print(f'adapter 训练 {args.adapter_epochs} epoch ...')
    optimizer = build_optimizer(adapter, [], adapter_lr=1e-3, adapter_wd=1e-4)
    history = train_adapter_epochs(adapter, optimizer, train_emb, train_labels, epochs=args.adapter_epochs, device=device, p=p_pk, k=k_pk, batches_per_epoch=batches, seed=args.seed, )

    adapter.eval()
    with torch.no_grad():
        x_eval = torch.as_tensor(eval_emb, dtype=torch.float32, device=device)
        adapted_eval = F.normalize(adapter(x_eval), p=2, dim=1).cpu().numpy()
    adapter_scores, _ = scores_from_embeddings(adapted_eval, target_pairs, non_target_pairs)
    adapter_metrics = compute_metrics(adapter_scores, labels)
    fusion_scores = 0.5 * adapter_scores + 0.5 * z_scores
    fusion_metrics = compute_metrics(fusion_scores, labels)

    summary = {'checkpoint': str(args.checkpoint), 'feature': cfg['feature'], 'embedding_dim': cfg['embedding_dim'], 'gap_epochs': gap_epochs, 'adapter_epochs': args.adapter_epochs,
               'chunk_seconds': args.chunk_seconds, 'seed': args.seed, 'baseline': base_metrics, 'standardized': z_metrics, 'adapter': adapter_metrics, 'fusion': fusion_metrics, }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    (out / 'history.json').write_text(json.dumps(history, indent=2))
    if gap_history:
        (out / 'gap_history.json').write_text(json.dumps(gap_history, indent=2))
    print()
    print(json.dumps(summary, indent=2))
    print(f'结果已写入 {out}')


if __name__ == '__main__':
    main()
