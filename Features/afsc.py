"""AFSC filter geometry preserved from the supplied research implementation.
The default legacy initialization reproduces the supplied hard-coded sequence
and normalization, not an exact 20--7600 Hz Mel partition.
"""
import torch
from torch import nn
import torch.nn.functional as F

class TrainableAFSCFilter(nn.Module):
    def __init__(self, n_mels=80, spec_dim=257, mean_nor=True):
        super().__init__()
        self.n_mels = n_mels
        self.spec_dim = spec_dim
        self.mean_nor = mean_nor

        # Legacy reference sequence: 82 entries; 80 trainable gap parameters.
        full_init_points = torch.tensor([
            0.64125, 1.3486279, 2.0776815, 2.82907498, 3.60349291,
            4.4016408, 5.22424578, 6.07205729, 6.94584771, 7.8464131,
            8.77457389, 9.73117568, 10.71708997, 11.73321495, 12.78047636,
            13.85982829, 14.97225406, 16.11876713, 17.30041203, 18.51826525,
            19.77343633, 21.06706876, 22.40034108, 23.77446795, 25.19070126,
            26.65033124, 28.15468766, 29.70514105, 31.30310392, 32.95003208,
            34.64742594, 36.39683187, 38.19984366, 40.05810391, 41.97330556,
            43.94719343, 45.98156579, 48.07827604, 50.23923435, 52.46640943,
            54.76183032, 57.12758822, 59.56583842, 62.07880227, 64.66876915,
            67.33809861, 70.08922252, 72.92464724, 75.84695594, 78.85881096,
            81.96295619, 85.16221962, 88.45951591, 91.85784899, 95.36031487,
            98.97010442, 102.69050629, 106.5249099, 110.47680852, 114.54980247,
            118.74760239, 123.07403263, 127.53303472, 132.12867097, 136.86512817,
            141.74672138, 146.77789793, 151.96324139, 157.30747579, 162.81546991,
            168.49224174, 174.34296301, 180.37296394, 186.58773807, 192.99294729,
            199.59442696, 206.39819127, 213.41043868, 220.6375576, 228.0861322,
            235.76294838, 243.675,
        ], dtype=torch.float32)

        assert len(full_init_points) == n_mels + 2, \
            f"len(full_init_points)={len(full_init_points)}, expected={n_mels + 2}"

        # 固定首尾
        self.register_buffer("left_edge", full_init_points[:1].clone())  # [1]
        self.register_buffer("right_edge", full_init_points[-1:].clone())  # [1]

        # Reference interior points; normalized cumulative gaps repeat the right boundary.
        internal = full_init_points[1:-1].clone()
        assert len(internal) == n_mels, f"len(internal)={len(internal)}, n_mels={n_mels}"

        # 用 gap 参数化，保证中间点始终递增
        gaps = internal[1:] - internal[:-1]  # [79]
        first_gap = internal[:1] - self.left_edge  # [1]
        all_gaps = torch.cat([first_gap, gaps], dim=0)  # [80]

        self.raw_gaps = nn.Parameter(torch.log(torch.exp(all_gaps) - 1.0))

    def get_bin_points(self):
        gaps = F.softplus(self.raw_gaps)  # [80], 保证 > 0

        total_range = self.right_edge - self.left_edge
        gaps = gaps / gaps.sum() * total_range

        internal = self.left_edge + torch.cumsum(gaps, dim=0)  # [80]

        # 拼完整 82 个点
        bin_points = torch.cat([self.left_edge, internal, self.right_edge], dim=0)  # [82]
        return bin_points

    def create_filterbank(self, device, dtype):
        bin_points = self.get_bin_points().to(device=device, dtype=dtype)
        freq_bins = torch.arange(0, self.spec_dim, device=device, dtype=dtype)
        filters = []

        for j in range(self.n_mels):
            l = bin_points[j]
            c = bin_points[j + 1]
            r = bin_points[j + 2]

            left = (freq_bins - l) / (((c - l) ** 2) + 1e-12)
            right = (r - freq_bins) / (((r - c) ** 2) + 1e-12)

            tri = torch.where(freq_bins < c, left, right)
            tri = torch.where((freq_bins >= l) & (freq_bins <= r), tri, torch.zeros_like(tri))
            tri = torch.clamp(tri, min=0.0)
            filters.append(tri)

        return torch.stack(filters, dim=0)  # [80, 257]

    def forward(self, spec):
        device, dtype = spec.device, spec.dtype
        filt = self.create_filterbank(device, dtype)  # [80, 257]
        feat = torch.matmul(spec, filt.transpose(0, 1))  # [B, T, 80]
        feat = torch.clamp(feat, min=torch.finfo(dtype).eps)
        feat = torch.log(feat)

        if self.mean_nor:
            feat = feat - feat.mean(dim=1, keepdim=True)

        return feat
