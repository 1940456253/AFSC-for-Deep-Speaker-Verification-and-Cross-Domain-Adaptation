"""提取语音嵌入到 NPZ；默认对整段语音处理。"""
import argparse
import json
import sys
from pathlib import Path

# 让 `python Inference/extract.py` 在 PyCharm 里也能找到项目其他模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from Dataset.data import read_manifest, load_audio
from Features.frontend import build_extractor
from Model.system import SpeakerSystem


def default_checkpoint():
    """选择默认 checkpoint 和输出路径。

    这里有两套可选配置，随时切换：
      (A) 用户自己训练的模型  ——  当前默认启用
      (B) 论文提供的预训练模型 —— 已注释，需要时取消注释
    """
    # ========== (A) 用户自己训练的模型（当前默认） ==========
    return (
        PROJECT_ROOT / 'runs' / 'demo' / 'last.pt',           # checkpoint
        PROJECT_ROOT / 'runs' / 'demo' / 'embeddings.npz',    # 输出 NPZ
    )

    # ========== (B) 论文预训练模型（需要时启用） ==========
    # pretrained = PROJECT_ROOT / 'PreTrained' / 'cn_celeb2_afsc_ecapa.pt'
    # if pretrained.is_file():
    #     return pretrained, PROJECT_ROOT / 'runs' / 'pretrained_eval.npz'
    # # 若预训练模型不存在，则回退到用户自己训练的模型
    # return PROJECT_ROOT / 'runs' / 'demo' / 'last.pt', PROJECT_ROOT / 'runs' / 'demo' / 'embeddings.npz'


def load_system(checkpoint_path, device):
    """从 checkpoint 加载 SpeakerSystem 和前端特征提取器。"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    # 检查 checkpoint 格式，避免误用旧版
    if checkpoint.get('format_version') != 1:
        raise ValueError(
            '需要的是 Model.train 生成的 checkpoint，而不是旧版 embedding_model.ckpt。\n'
            '如果你有 3D-Speaker 旧版 checkpoint，请先用 tools/convert_legacy.py 转换。'
        )
    cfg = checkpoint['config']
    model = SpeakerSystem(cfg).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()  # 推理模式，关闭 dropout、固定 BN
    return model, build_extractor(cfg['feature']), cfg


def chunks(wav, seconds=None, sample_rate=16000):
    """把波形切成 chunk；seconds=None 表示整段处理。"""
    if seconds is None:
        # 极短音频补零到 0.1 秒，让前端产生足够多帧供 ECAPA 使用
        yield F.pad(wav, (0, max(0, 1600 - wav.numel())))
        return
    size = round(seconds * sample_rate)
    if size < 1600:
        raise ValueError('chunk_seconds 至少为 0.1 秒')
    if wav.numel() <= size:
        yield F.pad(wav, (0, size - wav.numel()))
        return
    full = wav.numel() // size
    for i in range(full):
        yield wav[i * size:(i + 1) * size]
    # 论文做法：剩余尾部 ≥ 2 秒时，再补一个末尾对齐的 chunk
    if wav.numel() - full * size >= 2 * sample_rate:
        yield wav[-size:]


@torch.inference_mode()
def embed(wav, model, extractor, device, chunk_seconds=None):
    """对一段波形提取归一化后的说话人嵌入。"""
    embeddings = [model(extractor(chunk).unsqueeze(0).to(device)).squeeze(0)
                  for chunk in chunks(wav, chunk_seconds)]
    averaged = torch.stack(embeddings).mean(0)  # 多 chunk 求平均
    if not torch.isfinite(averaged).all() or averaged.norm() < 1e-12:
        raise ValueError('嵌入无效或为零向量')
    return F.normalize(averaged, dim=0).cpu().numpy()  # L2 归一化一次


def main():
    # 调用上面 default_checkpoint() 决定用哪个 checkpoint 和输出路径
    default_ckpt, default_out = default_checkpoint()
    default_csv = PROJECT_ROOT / 'demo_data' / 'eval.csv'

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default=str(default_ckpt),
                   help=f'checkpoint 路径（默认: {default_ckpt}）')
    p.add_argument('--csv', default=str(default_csv),
                   help=f'要提取嵌入的清单 CSV（默认: {default_csv}）')
    p.add_argument('--output', default=str(default_out),
                   help=f'输出 NPZ 文件（默认: {default_out}）')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--chunk-seconds', type=float,
                   help='可选：非重叠 chunk 秒数；尾部 ≥2 秒会补一个末尾对齐 chunk')
    args = p.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    csv_path = Path(args.csv).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f'找不到 checkpoint: {checkpoint_path}\n'
            f'请先训练模型（python -m Model.train），或用 tools/convert_legacy.py 转换旧版 checkpoint。'
        )
    if not csv_path.is_file():
        raise FileNotFoundError(
            f'找不到清单 CSV: {csv_path}\n'
            f'请先运行 python -m Dataset.demo 生成 demo 数据。'
        )
    if not str(output_path).endswith('.npz'):
        raise ValueError('--output 必须以 .npz 结尾')

    device = torch.device(args.device)
    model, extractor, cfg = load_system(checkpoint_path, device)
    rows = read_manifest(csv_path)
    vectors = []
    for i, row in enumerate(rows):
        vectors.append(embed(load_audio(row), model, extractor, device, args.chunk_seconds))
        if i % 100 == 0:
            print(f'已提取 {i + 1}/{len(rows)}', flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path,
                        ids=np.asarray([r['ID'] for r in rows]),
                        embeddings=np.stack(vectors))
    # 顺便写一份 JSON 元信息，方便追溯
    output_path.with_suffix('.json').write_text(json.dumps({
        'checkpoint': str(checkpoint_path),
        'csv': str(csv_path),
        'feature': cfg['feature'],
        'utterances': len(rows),
        'chunk_seconds': args.chunk_seconds,
        'aggregation': 'raw embedding mean, then L2 normalization',
    }, indent=2, ensure_ascii=False))
    print(f'嵌入已写入 {output_path}')


if __name__ == '__main__':
    main()