"""
推理场景复用 diffusers 的通用 embedding 实现。

我们只做推理且权重会从 checkpoint 加载时，手写初始化逻辑没有实际价值；
这里保留对外 API（`TimestepEmbedding` / `apply_rotary_emb`）不变即可。
"""

from diffusers.models.embeddings import TimestepEmbedding, apply_rotary_emb

__all__ = ["TimestepEmbedding", "apply_rotary_emb"]
