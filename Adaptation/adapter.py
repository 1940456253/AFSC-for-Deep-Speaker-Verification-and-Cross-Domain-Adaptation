"""残差嵌入适配器，对应论文 3.5.2 节 / Eq.(14)。"""
import torch
import torch.nn as nn


class WideResidualAdapter(nn.Module):
    """192 -> 384 -> 192 的残差 MLP。

    结构：
        A(x) = x + alpha * [ W2 Dropout(ReLU(W1 x + b1)) + b2 ]

    - hidden_dim = 384
    - dropout = 0.40
    - fc2 零初始化，alpha 初始 0.1，所以初始时 A(x) = x
    """

    def __init__(self, dim=192, hidden_dim=384, dropout=0.40, alpha_init=0.1):
        super().__init__()
        self.fc1 = nn.Linear(int(dim), int(hidden_dim))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(float(dropout))
        self.fc2 = nn.Linear(int(hidden_dim), int(dim))
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))

    def residual(self, x):
        return self.fc2(self.dropout(self.activation(self.fc1(x))))

    def forward(self, x):
        return x + self.alpha * self.residual(x)
