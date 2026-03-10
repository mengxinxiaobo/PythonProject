import torch
import torch.nn as nn


def rbf_mmd(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    x: [B, D], y: [B, D]
    MMD^2 with RBF kernel.
    """
    # pairwise squared distances
    xx = torch.cdist(x, x) ** 2
    yy = torch.cdist(y, y) ** 2
    xy = torch.cdist(x, y) ** 2

    k_xx = torch.exp(-xx / (2 * sigma * sigma))
    k_yy = torch.exp(-yy / (2 * sigma * sigma))
    k_xy = torch.exp(-xy / (2 * sigma * sigma))

    return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()


class CAE1D(nn.Module):
    """
    对单个变量的窗口序列做 1D Conv AutoEncoder
    输入:  [B, 1, T]
    输出:  重构 [B, 1, T]，隐向量 z [B, D]
    """
    def __init__(self, T: int, z_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),          # T/2
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),          # T/4
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # 计算编码后的长度
        enc_len = T // 4
        self.to_z = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * enc_len, 128),
            nn.ReLU(),
            nn.Linear(128, z_dim),
        )
        self.from_z = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * enc_len),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),  # T/2
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),  # T
            nn.Conv1d(32, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=3, padding=1),
        )

        self.enc_len = enc_len

    def forward(self, x):
        h = self.encoder(x)                      # [B, 64, T/4]
        z = self.to_z(h)                         # [B, D]
        h2 = self.from_z(z).view(-1, 64, self.enc_len)
        xhat = self.decoder(h2)                  # [B, 1, T]
        return xhat, z