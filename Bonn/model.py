"""
Bonn EEG 癫痫检测 - PyTorch 模型架构

实现混合深度学习架构: 1D-CNN + Bi-LSTM + Self-Attention
"""
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """Self-Attention 层，让模型关注关键时间步。"""

    def __init__(self, feature_dim: int, attention_units: int = 64):
        super().__init__()
        self.W = nn.Linear(feature_dim, attention_units, bias=True)
        self.u = nn.Linear(attention_units, 1, bias=False)

    def forward(self, x):
        # x: (batch, seq_len, feature_dim)
        score = torch.tanh(self.W(x))           # (batch, seq_len, attention_units)
        weights = torch.softmax(self.u(score), dim=1)  # (batch, seq_len, 1)
        context = torch.sum(x * weights, dim=1)  # (batch, feature_dim)
        return context, weights.squeeze(-1)


class CNNBiLSTMAttention(nn.Module):
    """
    1D-CNN + Bi-LSTM + Self-Attention 混合模型。

    Architecture:
        Input (batch, window_size, 1)
        -> Conv1D -> BN -> ReLU -> MaxPool -> Dropout
        -> Conv1D -> BN -> ReLU -> MaxPool -> Dropout
        -> Bi-LSTM (layer 1)
        -> Bi-LSTM (layer 2)
        -> Self-Attention
        -> Dense -> Dropout -> Output
    """

    def __init__(self,
                 input_channels: int = 1,
                 num_classes: int = 2,
                 cnn_filters: list[int] | None = None,
                 cnn_kernel_size: int = 7,
                 lstm_units: list[int] | None = None,
                 attention_units: int = 64,
                 dense_units: int = 64,
                 dropout_rate: float = 0.4):
        super().__init__()

        if cnn_filters is None:
            cnn_filters = [64, 128]
        if lstm_units is None:
            lstm_units = [128, 64]

        # ---- 1D-CNN 特征提取 ----
        cnn_layers = []
        in_ch = input_channels
        for filters in cnn_filters:
            cnn_layers.extend([
                nn.Conv1d(in_ch, filters, kernel_size=cnn_kernel_size, padding=cnn_kernel_size // 2),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(dropout_rate * 0.5),
            ])
            in_ch = filters
        self.cnn = nn.Sequential(*cnn_layers)

        # ---- Bi-LSTM 时序建模 ----
        self.lstm_layers = nn.ModuleList()
        self.lstm_dropouts = nn.ModuleList()
        lstm_input_size = cnn_filters[-1]
        for units in lstm_units:
            self.lstm_layers.append(
                nn.LSTM(lstm_input_size, units, batch_first=True, bidirectional=True)
            )
            self.lstm_dropouts.append(nn.Dropout(dropout_rate))
            lstm_input_size = units * 2  # bidirectional

        # ---- Self-Attention ----
        self.attention = SelfAttention(lstm_units[-1] * 2, attention_units)

        # ---- 分类头 ----
        self.classifier = nn.Sequential(
            nn.Linear(lstm_units[-1] * 2, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_units, 1 if num_classes == 2 else num_classes),
        )

        self.num_classes = num_classes

    def forward(self, x):
        # x: (batch, seq_len, 1)
        x = x.permute(0, 2, 1)   # -> (batch, channels, seq_len) for Conv1d
        x = self.cnn(x)
        x = x.permute(0, 2, 1)   # -> (batch, seq_len, features) for LSTM

        for lstm, dropout in zip(self.lstm_layers, self.lstm_dropouts):
            x, _ = lstm(x)
            x = dropout(x)

        context, attn_weights = self.attention(x)
        logits = self.classifier(context)

        if self.num_classes == 2:
            logits = logits.squeeze(-1)

        return logits, attn_weights


class BiLSTMAttention(nn.Module):
    """纯 Bi-LSTM + Attention 基线模型（无 CNN，用于对比实验）。"""

    def __init__(self,
                 input_channels: int = 1,
                 num_classes: int = 2,
                 lstm_units: list[int] | None = None,
                 attention_units: int = 64,
                 dense_units: int = 64,
                 dropout_rate: float = 0.4):
        super().__init__()

        if lstm_units is None:
            lstm_units = [128, 64]

        self.lstm_layers = nn.ModuleList()
        self.lstm_dropouts = nn.ModuleList()
        lstm_input_size = input_channels
        for units in lstm_units:
            self.lstm_layers.append(
                nn.LSTM(lstm_input_size, units, batch_first=True, bidirectional=True)
            )
            self.lstm_dropouts.append(nn.Dropout(dropout_rate))
            lstm_input_size = units * 2

        self.attention = SelfAttention(lstm_units[-1] * 2, attention_units)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_units[-1] * 2, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_units, 1 if num_classes == 2 else num_classes),
        )

        self.num_classes = num_classes

    def forward(self, x):
        # x: (batch, seq_len, 1)
        for lstm, dropout in zip(self.lstm_layers, self.lstm_dropouts):
            x, _ = lstm(x)
            x = dropout(x)

        context, attn_weights = self.attention(x)
        logits = self.classifier(context)

        if self.num_classes == 2:
            logits = logits.squeeze(-1)

        return logits, attn_weights


class Pure1DCNN(nn.Module):
    """
    纯 1D-CNN 基线模型（无 LSTM/Attention，用于验证时序建模的价值）。

    Architecture:
        Input -> Conv1D*4 (each with BN+ReLU+MaxPool+Dropout)
        -> Global Average Pooling -> Dense -> Output
    """

    def __init__(self,
                 input_channels: int = 1,
                 num_classes: int = 2,
                 cnn_filters: list[int] | None = None,
                 cnn_kernel_size: int = 7,
                 dense_units: int = 64,
                 dropout_rate: float = 0.4):
        super().__init__()

        if cnn_filters is None:
            cnn_filters = [32, 64, 128, 128]

        cnn_layers = []
        in_ch = input_channels
        for filters in cnn_filters:
            cnn_layers.extend([
                nn.Conv1d(in_ch, filters, kernel_size=cnn_kernel_size, padding=cnn_kernel_size // 2),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(dropout_rate * 0.5),
            ])
            in_ch = filters
        self.cnn = nn.Sequential(*cnn_layers)

        self.classifier = nn.Sequential(
            nn.Linear(cnn_filters[-1], dense_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_units, 1 if num_classes == 2 else num_classes),
        )

        self.num_classes = num_classes

    def forward(self, x):
        # x: (batch, seq_len, 1)
        x = x.permute(0, 2, 1)   # -> (batch, 1, seq_len)
        x = self.cnn(x)           # -> (batch, filters, reduced_seq)
        x = x.mean(dim=2)         # Global Average Pooling -> (batch, filters)
        logits = self.classifier(x)

        if self.num_classes == 2:
            logits = logits.squeeze(-1)

        return logits, None       # 无 attention weights，返回 None 保持接口一致


def build_model(model_type: str, num_classes: int, **kwargs) -> nn.Module:
    """工厂函数，根据类型创建模型。"""
    if model_type == "hybrid":
        return CNNBiLSTMAttention(num_classes=num_classes, **kwargs)
    elif model_type == "bilstm":
        return BiLSTMAttention(num_classes=num_classes, **kwargs)
    elif model_type == "cnn":
        return Pure1DCNN(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"未知模型类型: {model_type}，可选: hybrid, bilstm, cnn")


def count_parameters(model: nn.Module) -> int:
    """统计模型可训练参数总数。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from config import WINDOW_SIZE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dummy = torch.randn(4, WINDOW_SIZE, 1).to(device)

    print("=" * 60)
    print("CNN + Bi-LSTM + Attention (二分类)")
    print("=" * 60)
    m = build_model("hybrid", num_classes=2).to(device)
    out, attn = m(dummy)
    print(f"  参数量: {count_parameters(m):,}")
    print(f"  输出形状: {out.shape}")

    print("\n" + "=" * 60)
    print("CNN + Bi-LSTM + Attention (五分类)")
    print("=" * 60)
    m5 = build_model("hybrid", num_classes=5).to(device)
    out5, attn5 = m5(dummy)
    print(f"  参数量: {count_parameters(m5):,}")
    print(f"  输出形状: {out5.shape}")
