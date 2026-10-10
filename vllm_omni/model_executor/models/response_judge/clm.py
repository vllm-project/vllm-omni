# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CLM (Contrastive-LM, e.g. CLM-v0.1-8B) as a response judge (pooling runner).

Same maths as clm.engine / clm.heads (Apache-2.0): the Qwen3 encoder embeds the
state text with last-token pooling; the state head projects it; the score of
option ``i`` is ``scale * cos(state_head(s), action_head(option_i))``.

The judge's options are fixed per deployment, so their action-head
projections are computed once when the model directory is prepared and loaded
here as ``clm.option_proj`` (with the state head as ``clm.state_head.*`` and
the final ``clm.scale``). A judge request then costs one encoder pass; the
pooler returns one raw logit per option.
"""

from __future__ import annotations

from collections.abc import Iterable, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler import Pooler
from vllm.model_executor.models.adapters import as_embedding_model

from vllm_omni.model_executor.models.response_judge.qwen3 import ResponseJudgeQwen3ForCausalLM

_ACTIVATIONS = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}


class ClmHead(nn.Module):
    """``hidden -> width -> ... -> proj`` MLP, parameter names as in clm.heads.make_head."""

    def __init__(
        self,
        hidden: int,
        width: int,
        depth: int = 2,
        proj: int = 512,
        activation: str = "gelu",
        layernorm: bool = False,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.inp = nn.Linear(hidden, width)
        self.hidden = nn.ModuleList(nn.Linear(width, width) for _ in range(depth - 2))
        self.norms = nn.ModuleList((nn.LayerNorm(width) if layernorm else nn.Identity()) for _ in range(depth - 2))
        self.out = nn.Linear(width, proj)
        self.act = _ACTIVATIONS[activation]()
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.inp(x))
        for lin, norm in zip(self.hidden, self.norms):
            h = self.act(norm(lin(x)))
            x = x + h if self.residual else h
        return self.out(x)


class ClmDecisionPooler(Pooler):
    """Per request: raw logits ``scale * option_proj @ normalize(state_head(normalize(last)))``."""

    def __init__(self, head_cfg: dict, num_options: int) -> None:
        super().__init__()
        self.state_head = ClmHead(**head_cfg)
        proj = int(head_cfg.get("proj", 512))
        self.option_proj = nn.Parameter(torch.zeros(num_options, proj), requires_grad=False)
        self.scale = nn.Parameter(torch.ones(()), requires_grad=False)
        # The heads were trained in fp32 on fp32 embeddings.
        self.float()

    def get_supported_tasks(self) -> Set[str]:
        return {"classify"}

    def forward(self, hidden_states: torch.Tensor, pooling_metadata) -> list[torch.Tensor]:
        # Same contract as vLLM's LAST pooling: the last token scheduled this
        # step. With a prefix-cache hit that is the prompt's last token; the
        # runner drops outputs of requests whose prefill is not finished yet.
        cursor = pooling_metadata.get_pooling_cursor()
        last = hidden_states[cursor.last_token_indices_gpu].float()
        states = F.normalize(self.state_head(F.normalize(last, dim=-1)), dim=-1)
        options = F.normalize(self.option_proj.float(), dim=-1)
        logits = self.scale.float() * states @ options.T
        return list(logits.unbind(0))


class ClmDecisionModel(as_embedding_model(ResponseJudgeQwen3ForCausalLM)):  # type: ignore[misc]
    """Qwen3 encoder (last-token pooling) + CLM state head + fixed option projections."""

    def _init_pooler(self, vllm_config: VllmConfig, prefix: str = "") -> Pooler:
        config = vllm_config.model_config.hf_config
        head_cfg = dict(getattr(config, "clm_head"))
        head_cfg.setdefault("hidden", config.hidden_size)
        return ClmDecisionPooler(head_cfg, int(getattr(config, "clm_num_options")))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        head_params = dict(self.pooler.named_parameters())
        backbone: list[tuple[str, torch.Tensor]] = []
        loaded: set[str] = set()
        for name, tensor in weights:
            if name.startswith("clm."):
                key = name[len("clm.") :]
                param = head_params[key]
                if tensor.shape != param.shape:
                    raise ValueError(
                        f"CLM judge weight {name!r} has shape {tuple(tensor.shape)}; expected {tuple(param.shape)}"
                    )
                with torch.no_grad():
                    param.copy_(tensor.to(param.dtype))
                loaded.add(f"pooler.{key}")
            else:
                backbone.append((name, tensor))
        return loaded | set(super().load_weights(backbone))


__all__ = ["ClmDecisionModel", "ClmDecisionPooler", "ClmHead"]
