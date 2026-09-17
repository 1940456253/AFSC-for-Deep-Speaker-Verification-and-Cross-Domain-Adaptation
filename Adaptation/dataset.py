"""适配器训练用的 embedding dataset。"""
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.x = torch.as_tensor(embeddings, dtype=torch.float32)
        unique = sorted(set(map(str, labels)))
        label_to_id = {s: i for i, s in enumerate(unique)}
        self.y = torch.tensor([label_to_id[str(s)] for s in labels], dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]
