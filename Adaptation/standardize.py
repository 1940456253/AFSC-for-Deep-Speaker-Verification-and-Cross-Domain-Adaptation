"""目标域 embedding 标准化，对应论文 3.5.3 节 / Eq.(19)。"""
import numpy as np


def fit_standardizer(embeddings, eps=1e-5):
    """从 target-domain embeddings 估计逐维 mu 和 sigma。"""
    x = np.asarray(embeddings, dtype=np.float32)
    mu = x.mean(axis=0, keepdims=True)
    sd = np.maximum(x.std(axis=0, keepdims=True), eps)
    return mu, sd


def apply_standardizer(embeddings, mu, sd):
    """式(19)：逐维标准化 + L2 归一化。"""
    x = np.asarray(embeddings, dtype=np.float32)
    z = (x - mu) / sd
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    return z / np.maximum(norms, 1e-12)


def l2_normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
