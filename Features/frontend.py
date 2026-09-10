"""CPU temporal preprocessing; trainable AFSC filtering stays inside the model."""
from .spectrum import SpecExtractor, FBank, MFCC


def build_extractor(feature):
    if feature == 'afsc':
        return SpecExtractor(sample_rate=16000, mean_nor=False)
    if feature == 'fbank':
        return FBank(n_mels=80, sample_rate=16000, mean_nor=True)
    if feature == 'mfcc':
        return MFCC(n_mfcc=80, sample_rate=16000, mean_nor=True)
    raise ValueError(f'Unknown feature: {feature}')
