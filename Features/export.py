"""导出 AFSC 滤波器权重和频率点；可选 PNG 可视化。"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from Inference.extract import load_system


def default_paths():
    """选择默认 checkpoint 和输出目录。

      (A) 用户自己训练的模型  ——  当前默认启用
      (B) 论文预训练模型      ——  已注释，需要时取消注释
    """
    # ========== (A) 用户自己训练的模型（当前默认） ==========
    return (
        PROJECT_ROOT / 'runs' / 'demo' / 'last.pt',
        PROJECT_ROOT / 'runs' / 'demo' / 'filters',
    )

    # ========== (B) 论文预训练模型（需要时启用） ==========
    # pretrained = PROJECT_ROOT / 'PreTrained' / 'cn_celeb2_afsc_ecapa.pt'
    # if pretrained.is_file():
    #     return pretrained, PROJECT_ROOT / 'runs' / 'pretrained_filters'
    # return PROJECT_ROOT / 'runs' / 'demo' / 'last.pt', PROJECT_ROOT / 'runs' / 'demo' / 'filters'


def main():
    default_ckpt, default_out = default_paths()

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default=str(default_ckpt),
                   help=f'AFSC checkpoint（默认: {default_ckpt}）')
    p.add_argument('--output', default=str(default_out),
                   help=f'输出目录（默认: {default_out}）')
    p.add_argument('--plot', action='store_true',
                   help='额外生成 PNG 可视化')
    args = p.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f'找不到 checkpoint: {checkpoint_path}\n'
            f'请先训练模型（python -m Model.train），或用 tools/convert_legacy.py 转换旧版 checkpoint。'
        )

    model, _, cfg = load_system(checkpoint_path, torch.device('cpu'))
    if cfg['feature'] != 'afsc':
        raise ValueError(f'需要 AFSC checkpoint，但得到 feature={cfg["feature"]!r}')
    with torch.no_grad():
        points = model.frontend.get_bin_points().numpy()
        filters = model.frontend.create_filterbank(torch.device('cpu'), torch.float32).numpy()

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / 'afsc_filters.npz',
             bin_points=points, hz=points * 31.25, filters=filters)
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(np.arange(257) * 31.25, filters.T, linewidth=0.7)
        ax.set(xlabel='Frequency (Hz)', ylabel='Filter weight',
               title='Learned AFSC filters')
        fig.tight_layout()
        fig.savefig(output_dir / 'afsc_filters.png', dpi=180)
        plt.close(fig)
    print(f'checkpoint: {checkpoint_path}')
    print(f'滤波器已写入 {output_dir}')


if __name__ == '__main__':
    main()