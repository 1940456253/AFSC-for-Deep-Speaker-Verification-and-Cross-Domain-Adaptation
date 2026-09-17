"""用第一部分 SpeakerSystem 提取 30 秒分块 embedding。"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from Dataset.data import load_audio
from Features.frontend import build_extractor
from Model.system import SpeakerSystem
from Adaptation.chunking import make_chunks


def load_system_from_checkpoint(checkpoint_path, device):
    """加载第一部分 checkpoint，返回 (model, extractor, config)。"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != 1:
        raise ValueError('需要第一部分 Model.train 生成的 checkpoint。\n'
                         '如果你有 3D-Speaker 旧版 checkpoint，请先运行 tools/convert_legacy.py。')
    cfg = checkpoint['config']
    model = SpeakerSystem(cfg).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    extractor = build_extractor(cfg['feature'])
    return model, extractor, cfg


@torch.inference_mode()
def embed_utterance(wav, model, extractor, device, chunk_seconds=30.0, hop_seconds=30.0):
    """对一条 utterance 提取 embedding。

    - chunk_seconds=None 时整段处理
    - 多 chunk 先算 raw embedding 平均，再 L2 归一化（论文 Eq.12-13）
    """
    if chunk_seconds is None:
        chunks = [wav.unsqueeze(0) if wav.dim() == 1 else wav]
    else:
        chunks = make_chunks(wav, sample_rate=16000, chunk_seconds=chunk_seconds, hop_seconds=hop_seconds)
    embs = []
    for chunk in chunks:
        feat = extractor(chunk).unsqueeze(0).to(device)
        emb = model(feat).squeeze(0)
        embs.append(emb)
    avg = torch.stack(embs).mean(0)
    if not torch.isfinite(avg).all() or avg.norm() < 1e-12:
        raise ValueError('Embedding is non-finite or zero')
    return F.normalize(avg, dim=0).cpu().numpy()


@torch.inference_mode()
def embed_manifest(rows, model, extractor, device, chunk_seconds=30.0, hop_seconds=30.0, log_every=100, label=''):
    """对整个 manifest 提取 embedding，返回 [N, D] numpy 数组。"""
    embeddings = []
    for i, row in enumerate(rows):
        wav = load_audio(row)
        embeddings.append(embed_utterance(wav, model, extractor, device, chunk_seconds, hop_seconds))
        if (i + 1) % log_every == 0:
            print(f'  {label}{i + 1}/{len(rows)}', flush=True)
    return np.stack(embeddings).astype(np.float32)
