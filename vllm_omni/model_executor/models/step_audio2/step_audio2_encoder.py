"""
Step-Audio2 Audio Encoder and Adapter
"""

from typing import Optional, Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_non_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Create non-padding mask from lengths"""
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return ~mask


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert mask to attention bias"""
    assert mask.dtype == torch.bool
    assert dtype in [torch.float32, torch.bfloat16, torch.float16]
    mask = mask.to(dtype)
    mask = (1.0 - mask) * -1.0e+10
    return mask


class MultiHeadAttention(nn.Module):
    """Multi-head attention for audio encoder"""

    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ):
        _, T, D = q.shape
        scale = (D // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k  # (B, n_head, T, T)
        if mask is not None:
            qk = qk + mask
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):
    """Residual attention block for audio encoder"""

    def __init__(self, n_state: int, n_head: int):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_mlp),
            nn.GELU(),
            nn.Linear(n_mlp, n_state)
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    """
    Step-Audio2 Audio Encoder

    Lightweight audio encoder (6 layers, 512 hidden)
    Optimized for 25s audio chunks at 25Hz
    """

    def __init__(
        self,
        n_mels: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(
            n_state,
            n_state,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.positional_embedding = nn.Embedding(n_ctx, n_state)

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList([
            ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)
        ])
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.after_norm = nn.LayerNorm(n_state)

    def forward(
        self,
        x: torch.Tensor,
        x_len: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: mel spectrogram, shape (batch_size, n_mels, T)
            x_len: length of each audio, shape (batch_size,)

        Returns:
            x: encoded features, shape (batch_size, T//4, n_state)
            x_len: updated lengths, shape (batch_size,)
        """
        T = x.size(-1)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)  # (B, T // 2, n_state)

        # Create attention mask
        mask = make_non_pad_mask(x_len, T).unsqueeze(1)  # (B, 1, T)
        mask = mask_to_bias(mask[:, :, (T + 1) % 2::2], x.dtype)  # (B, 1, T // 2)

        # Add positional embedding
        x = (x + self.positional_embedding.weight[:x.shape[1], :]).to(x.dtype).contiguous()

        # Apply attention blocks
        for block in self.blocks:
            x = block(x, mask.unsqueeze(1))

        # Pool and normalize
        x = x.permute(0, 2, 1)
        x = self.avg_pooler(x)
        x = x.permute(0, 2, 1)
        x_len = (x_len + 1) // 2 // 2
        x = self.after_norm(x)

        return x, x_len


class Adaptor(nn.Module):
    """
    Adaptor to project audio features to LLM dimension

    Maps from n_state (512) to n_hidden (LLM dimension, e.g., 4096)
    with optional downsampling via convolution
    """

    def __init__(
        self,
        n_state: int = 512,
        n_hidden: int = 4096,
        kernel_size: int = 3,
        stride: int = 2,
        adapter_state: int = 2048
    ):
        super().__init__()
        self.stride = stride

        if self.stride != -1:
            self.conv = nn.Conv1d(
                n_state,
                n_state,
                kernel_size,
                stride,
                padding=1
            )

        self.linear1 = nn.Linear(n_state, adapter_state)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(adapter_state, n_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: audio features, shape (batch_size, T, n_state)

        Returns:
            x: projected features, shape (batch_size, T//stride, n_hidden)
        """
        if self.stride != -1:
            x = x.permute(0, 2, 1)  # (B, n_state, T)
            x = F.gelu(self.conv(x))
            x = x.permute(0, 2, 1)  # (B, T//stride, n_state)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x
