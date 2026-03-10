# Src/models/gcn_bilstm.py
import torch
import torch.nn as nn


def normalize_adjacency(A: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    A: [V,V] non-negative weighted adjacency (with self-loop)
    return: D^{-1/2} A D^{-1/2}
    """
    A = torch.clamp(A, min=0.0)
    deg = A.sum(dim=1)
    inv_sqrt = torch.rsqrt(deg + eps)
    D = torch.diag(inv_sqrt)
    return D @ A @ D


class GCNBiLSTMPredictor(nn.Module):
    """
    向量化版（关键优化）：
    - 一次性对所有时间步做图传播，避免 Python for t in range(T)
    输入:  x [B,T,V]   (每节点 1 维特征)
    输出:  yhat [B,V]  (预测下一时刻)
    """
    def __init__(
        self,
        A_norm: torch.Tensor,
        gcn_hidden: int = 64,
        lstm_hidden: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.register_buffer("A_norm", A_norm)  # [V,V]

        self.gcn_in = nn.Linear(1, gcn_hidden)
        self.gcn_out = nn.Linear(gcn_hidden, gcn_hidden)

        self.lstm = nn.LSTM(
            input_size=gcn_hidden,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2 * lstm_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,V]
        B, T, V = x.shape
        x = x.unsqueeze(-1)  # [B,T,V,1]

        # 图传播：对所有时间步一次性做 A_norm @ x
        # A_norm: [V,V], x: [B,T,V,1] -> ax: [B,T,V,1]
        ax = torch.einsum("ij,btjf->btif", self.A_norm, x)

        h = torch.relu(self.gcn_in(ax))   # [B,T,V,H]
        h = torch.relu(self.gcn_out(h))   # [B,T,V,H]

        # 对每个节点做 BiLSTM： [B,T,V,H] -> [B*V,T,H]
        h = h.permute(0, 2, 1, 3).contiguous().view(B * V, T, -1)
        out, _ = self.lstm(h)             # [B*V,T,2*lstm_hidden]
        last = self.dropout(out[:, -1, :])
        yhat = self.head(last).view(B, V)
        return yhat