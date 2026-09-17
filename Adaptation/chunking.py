"""长音频处理，对应论文 3.5.1 节和日文笔记 3.2 节。"""
import torch
import torch.nn.functional as F


def make_chunks(wav, sample_rate=16000, chunk_seconds=30.0, hop_seconds=30.0, min_tail_seconds=2.0):
    """按论文规则切 chunk。

    - 短于 chunk 的音频右补零
    - 30 秒 chunk，30 秒 hop
    - 未使用尾部 ≥ 2 秒时，追加末尾对齐 chunk
    """
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    total = wav.shape[-1]
    chunk_len = int(round(chunk_seconds * sample_rate))
    hop_len = int(round(hop_seconds * sample_rate))
    min_tail = int(round(min_tail_seconds * sample_rate))

    if total <= chunk_len:
        return [F.pad(wav, (0, chunk_len - total))]

    starts = list(range(0, total - chunk_len + 1, max(1, hop_len)))
    last_start = total - chunk_len
    if not starts:
        starts = [0]
    elif starts[-1] != last_start:
        uncovered = total - (starts[-1] + chunk_len)
        if uncovered >= min_tail:
            starts.append(last_start)

    return [wav[..., s:s + chunk_len].contiguous() for s in starts]
