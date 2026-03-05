import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    """
    输入:  [B, T, N]
    输出:  [B, N]  预测下一时刻
    """
    def __init__(self, num_features: int, hidden: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, num_features)

    def forward(self, x):
        out, _ = self.lstm(x)          # [B, T, H]
        last = out[:, -1, :]           # [B, H]
        yhat = self.head(last)         # [B, N]
        return yhat