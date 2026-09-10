"""AFSC/FBank/MFCC + the supplied ECAPA-TDNN encoder."""
import torch
from torch import nn
import torch.nn.functional as F
from Features.afsc import TrainableAFSCFilter
from .ecapa_tdnn import ECAPA_TDNN


class SpeakerSystem(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature = config['feature']
        if self.feature not in ('afsc', 'fbank', 'mfcc'):
            raise ValueError('feature must be afsc, fbank, or mfcc')
        self.frontend = TrainableAFSCFilter() if self.feature == 'afsc' else nn.Identity()
        self.encoder = ECAPA_TDNN(input_size=80, lin_neurons=config['embedding_dim'],
                                 channels=config['channels'])

    def forward(self, features):
        return self.encoder(self.frontend(features))

    def freeze_frontend(self, freeze=True):
        for p in self.frontend.parameters():
            p.requires_grad_(not freeze)


class CosineClassifier(nn.Module):
    def __init__(self, embedding_dim, num_speakers):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_speakers, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embedding):
        return F.linear(F.normalize(embedding, dim=-1), F.normalize(self.weight, dim=-1))


def angular_margin_loss(cosine, labels, scale=32.0, margin=0.3):
    """ArcFace with monotonic correction, easy_margin=False."""
    import math
    cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
    sine = (1.0 - cosine.square()).clamp_min(1e-7).sqrt()
    phi = cosine * math.cos(margin) - sine * math.sin(margin)
    phi = torch.where(cosine > math.cos(math.pi - margin), phi,
                      cosine - math.sin(math.pi - margin) * margin)
    one_hot = F.one_hot(labels, cosine.shape[-1]).to(cosine.dtype)
    return F.cross_entropy(scale * (one_hot * phi + (1 - one_hot) * cosine), labels)
