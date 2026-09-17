"""P-K 采样器，对应论文 2.4 节。"""
import numpy as np
from torch.utils.data import Sampler


class PKBatchSampler(Sampler):
    """每 batch 采 P 个说话人，每人 K 条。"""

    def __init__(self, labels, p, k, batches_per_epoch, seed=0):
        self.labels = np.asarray(labels)
        self.p = int(p)
        self.k = int(k)
        self.batches = int(batches_per_epoch)
        self.seed = int(seed)
        self.iteration = 0
        self.by_label = {label: np.where(self.labels == label)[0] for label in sorted(set(self.labels.tolist()))}
        if len(self.by_label) < self.p:
            raise ValueError(f'Need P={self.p} speakers, only {len(self.by_label)} available')

    def __len__(self):
        return self.batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.iteration)
        self.iteration += 1
        labels = np.asarray(list(self.by_label.keys()), dtype=object)
        for _ in range(self.batches):
            chosen = rng.choice(labels, self.p, replace=False)
            batch = []
            for label in chosen:
                indices = self.by_label[label]
                picks = rng.choice(indices, self.k, replace=(len(indices) < self.k))
                batch.extend(picks.tolist())
            rng.shuffle(batch)
            yield batch
