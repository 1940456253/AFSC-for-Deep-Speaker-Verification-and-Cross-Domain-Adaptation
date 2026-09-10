"""比较两段波形。同人/异人判定需要一个校准过的阈值。"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from Dataset.data import load_audio
from Inference.extract import load_system, embed

AUDIO_EXTENSIONS = {'.wav', '.flac', '.ogg', '.aiff', '.aif', '.au', '.sf'}


def default_checkpoint():
    """选择默认 checkpoint。

      (A) 用户自己训练的模型  ——  当前默认启用
      (B) 论文预训练模型      ——  已注释，需要时取消注释
    """
    # ========== (A) 用户自己训练的模型（当前默认） ==========
    return PROJECT_ROOT / 'runs' / 'demo' / 'last.pt'

    # ========== (B) 论文预训练模型（需要时启用） ==========
    # pretrained = PROJECT_ROOT / 'PreTrained' / 'cn_celeb2_afsc_ecapa.pt'
    # if pretrained.is_file():
    #     return pretrained
    # return PROJECT_ROOT / 'runs' / 'demo' / 'last.pt'


def pick_default_pair():
    """从 Speech_example/speech_demo1 里挑两条音频做默认对比。"""
    demo_dir = PROJECT_ROOT / 'Speech_example' / 'speech_demo1'
    if not demo_dir.is_dir():
        return None, None
    files = sorted(p for p in demo_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)
    if len(files) < 2:
        return None, None
    return files[0], files[1]


def main():
    default_ckpt = default_checkpoint()
    default_enrol, default_test = pick_default_pair()

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default=str(default_ckpt),
                   help=f'checkpoint 路径（默认: {default_ckpt}）')
    p.add_argument('--enrol', default=str(default_enrol) if default_enrol else None,
                   help='注册音频（默认: Speech_example/speech_demo1 下第一条）')
    p.add_argument('--test', default=str(default_test) if default_test else None,
                   help='测试音频（默认: Speech_example/speech_demo1 下第二条）')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--threshold', type=float,
                   help='在开发集上校准后再使用，没有通用默认阈值')
    args = p.parse_args()

    if args.enrol is None or args.test is None:
        raise ValueError(
            '当 Speech_example/speech_demo1 缺失或不足两条音频时，必须显式指定 --enrol 和 --test。\n'
            '例如：python -m Inference.verify --enrol path/to/a.wav --test path/to/b.wav'
        )

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    enrol_path = Path(args.enrol).expanduser().resolve()
    test_path = Path(args.test).expanduser().resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f'找不到 checkpoint: {checkpoint_path}\n'
            f'请先训练模型（python -m Model.train），或用 tools/convert_legacy.py 转换旧版 checkpoint。'
        )
    if not enrol_path.is_file():
        raise FileNotFoundError(f'找不到注册音频: {enrol_path}')
    if not test_path.is_file():
        raise FileNotFoundError(f'找不到测试音频: {test_path}')

    device = torch.device(args.device)
    model, extractor, _ = load_system(checkpoint_path, device)
    a = embed(load_audio({'ID': 'enrol', 'path': str(enrol_path)}),
              model, extractor, device)
    b = embed(load_audio({'ID': 'test', 'path': str(test_path)}),
              model, extractor, device)
    score = float(a @ b)
    result = {
        'checkpoint': str(checkpoint_path),
        'enrol': str(enrol_path),
        'test': str(test_path),
        'cosine_similarity': score,
    }
    if args.threshold is not None:
        result.update(threshold=args.threshold, same_speaker=score >= args.threshold)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()