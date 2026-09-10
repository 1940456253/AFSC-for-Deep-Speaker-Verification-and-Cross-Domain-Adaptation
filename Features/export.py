"""Export learned filter weights and points; optional PNG visualization."""
import argparse
from pathlib import Path
import numpy as np
import torch
from Inference.extract import load_system


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--plot', action='store_true')
    args = p.parse_args()
    model, _, cfg = load_system(args.checkpoint, torch.device('cpu'))
    if cfg['feature'] != 'afsc':
        raise ValueError('Expected an AFSC checkpoint')
    with torch.no_grad():
        points = model.frontend.get_bin_points().numpy()
        filters = model.frontend.create_filterbank(torch.device('cpu'), torch.float32).numpy()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'afsc_filters.npz', bin_points=points, hz=points * 31.25, filters=filters)
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(np.arange(257) * 31.25, filters.T, linewidth=0.7)
        ax.set(xlabel='Frequency (Hz)', ylabel='Filter weight', title='Learned AFSC filters')
        fig.tight_layout()
        fig.savefig(out / 'afsc_filters.png', dpi=180)
        plt.close(fig)


if __name__ == '__main__':
    main()
