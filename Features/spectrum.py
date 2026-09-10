# Derived from the user-provided 3D-Speaker processor.py; see THIRD_PARTY_NOTICES.md.
# Copyright 3D-Speaker. Licensed under the Apache License, Version 2.0.
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as Kaldi

class FBank(object):
    def __init__(self,
                 n_mels,
                 sample_rate,
                 mean_nor: bool = False,
                 ):
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mean_nor = mean_nor

    def __call__(self, wav, dither=0):
        sr = 16000
        assert sr == self.sample_rate
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)
        # select single channel
        if wav.shape[0] > 1:
            wav = wav[0, :]
            wav = wav.unsqueeze(0)
        assert len(wav.shape) == 2 and wav.shape[0] == 1
        feat = Kaldi.fbank(wav, num_mel_bins=self.n_mels,
                           sample_frequency=sr, dither=dither)
        # feat: [T, N]
        if self.mean_nor:
            feat = feat - feat.mean(0, keepdim=True)
        return feat

class MFCC(object):
    def __init__(self, n_mfcc, sample_rate, mean_nor: bool = False, ):
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate
        self.mean_nor = mean_nor

    def __call__(self, wav, dither=0):
        sr = 16000
        assert sr == self.sample_rate
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)
        if wav.shape[0] > 1:
            wav = wav[0, :]
            wav = wav.unsqueeze(0)
        assert len(wav.shape) == 2 and wav.shape[0] == 1
        feat = Kaldi.mfcc(wav, num_mel_bins=self.n_mfcc, num_ceps=self.n_mfcc, sample_frequency=sr, dither=dither)
        if self.mean_nor:
            feat = feat - feat.mean(0, keepdim=True)
        return feat

class SpecExtractor(nn.Module):
    def __init__(self, sample_rate=16000, mean_nor: bool = False):
        super().__init__()
        self.sample_rate = sample_rate
        self.mean_nor = mean_nor

    def _next_power_of_2(self, x: int) -> int:
        return 1 if x == 0 else 2 ** (x - 1).bit_length()

    def _get_strided(self, waveform: torch.Tensor, window_size: int, window_shift: int, snip_edges: bool = True):
        assert waveform.dim() == 1
        num_samples = waveform.size(0)
        strides = (window_shift * waveform.stride(0), waveform.stride(0))

        if snip_edges:
            if num_samples < window_size:
                return torch.empty((0, 0), dtype=waveform.dtype, device=waveform.device)
            m = 1 + (num_samples - window_size) // window_shift
        else:
            reversed_waveform = torch.flip(waveform, [0])
            m = (num_samples + (window_shift // 2)) // window_shift
            pad = window_size // 2 - window_shift // 2
            pad_right = reversed_waveform
            if pad > 0:
                pad_left = reversed_waveform[-pad:]
                waveform = torch.cat((pad_left, waveform, pad_right), dim=0)
            else:
                waveform = torch.cat((waveform[-pad:], pad_right), dim=0)

        sizes = (m, window_size)
        return waveform.as_strided(sizes, strides)

    def _povey_window(self, window_size: int, device, dtype):
        # 和 Kaldi fbank 保持一致
        return torch.hann_window(
            window_size,
            periodic=False,
            device=device,
            dtype=dtype
        ).pow(0.85)

    def forward(self, wav, dither=0.0):
        sr = 16000
        assert sr == self.sample_rate

        # [T] -> [1, T]
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)

        # 多通道只取单通道
        if wav.shape[0] > 1:
            wav = wav[0, :].unsqueeze(0)

        assert len(wav.shape) == 2 and wav.shape[0] == 1

        waveform = wav[0]  # [num_samples]
        device, dtype = waveform.device, waveform.dtype

        # 尽量对齐 Kaldi fbank 默认参数
        frame_length = 25.0
        frame_shift = 10.0
        preemphasis_coefficient = 0.97
        remove_dc_offset = True
        round_to_power_of_two = True
        snip_edges = True
        use_power = True

        window_shift = int(sr * frame_shift * 0.001)
        window_size = int(sr * frame_length * 0.001)
        padded_window_size = self._next_power_of_2(window_size) if round_to_power_of_two else window_size

        # 分帧
        frames = self._get_strided(
            waveform,
            window_size,
            window_shift,
            snip_edges=snip_edges
        )
        # dither
        if dither != 0.0:
            frames = frames + torch.randn_like(frames) * dither
        # 去直流
        if remove_dc_offset:
            frames = frames - frames.mean(dim=1, keepdim=True)
        # 预加重
        if preemphasis_coefficient != 0.0:
            offset_frames = F.pad(frames.unsqueeze(0), (1, 0), mode="replicate").squeeze(0)
            frames = frames - preemphasis_coefficient * offset_frames[:, :-1]
        # Povey window
        window = self._povey_window(window_size, device=device, dtype=dtype).unsqueeze(0)
        frames = frames * window
        # pad 到 2 的幂次
        if padded_window_size != window_size:
            frames = F.pad(
                frames.unsqueeze(0),
                (0, padded_window_size - window_size),
                mode="constant",
                value=0
            ).squeeze(0)
        # FFT
        spectrum = torch.fft.rfft(frames).abs()
        # power spectrum
        if use_power:
            spectrum = spectrum.pow(2.0)
        # 避免后续 log 出现 0
        spectrum = torch.clamp(spectrum, min=torch.finfo(dtype).eps)
        # 可选：频率维均值归一化
        if self.mean_nor:
            spectrum = spectrum - spectrum.mean(dim=0, keepdim=True)
        return spectrum

