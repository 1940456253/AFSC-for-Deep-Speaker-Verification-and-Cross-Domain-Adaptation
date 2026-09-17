"""监督对比损失，对应论文 3.5.2 节 / Eq.(16)。"""
import torch
import torch.nn.functional as F


def supcon_loss(z, labels, temperature=0.07):
    """式(16) 的 supervised contrastive loss。

    z       : [B, D] 适配器输出
    labels  : [B] 说话人 ID
    """
    z = F.normalize(z, dim=1)
    logits = z @ z.T / float(temperature)
    labels = labels.view(-1, 1)
    same = labels.eq(labels.T)
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    positive = same & ~eye
    valid = positive.any(dim=1)
    if not valid.any():
        return z.sum() * 0.0

    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * (~eye).float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    mean_positive = ((positive.float() * log_prob).sum(dim=1) / positive.float().sum(dim=1).clamp_min(1))
    return -mean_positive[valid].mean()


def identity_loss(adapted, base):
    """式(17)：限制 adapter 对 embedding 的形变。"""
    return F.mse_loss(adapted, base)
