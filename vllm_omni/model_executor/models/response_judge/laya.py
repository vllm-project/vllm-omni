# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LAYA decision model (convaiinnovations/laya-*) as a response judge (pooling runner).

Same maths as laya.common.DecisionModel
(Apache-2.0): ModernBERT encoder -> + type embedding -> 2 TransformerEncoder
layers -> scorer at every [MASK] option marker. The act/escalate head is not
needed for a judge decision and is not loaded.

The prompt (``[CLS] <type> question: <ins> [SEP] [MASK] opt0 [MASK] opt1 [SEP]
state [SEP]``) is built by the ``response_judge`` bridge; the pooler finds the
option markers from the prompt token ids and returns one raw logit per option.
"""

from __future__ import annotations

from collections.abc import Iterable, Set

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler import Pooler
from vllm.model_executor.layers.pooler.common import PoolingParamsUpdate
from vllm.model_executor.models.interfaces_base import attn_type, default_pooling_type
from vllm.model_executor.models.modernbert import ModernBertModel
from vllm.model_executor.models.utils import maybe_prefix
from vllm.sequence import IntermediateTensors

# laya.common.QTYPES
LAYA_QTYPES = {"choice": 0, "score": 1, "noul": 2}


class LayaDecisionPooler(Pooler):
    """Decision head: per request, raw logits over that request's option markers."""

    def __init__(self, hidden_size: int, head_layers: int, mask_token_id: int, qtype: int) -> None:
        super().__init__()
        nhead = max(1, hidden_size // 64)
        layer = nn.TransformerEncoderLayer(hidden_size, nhead, 4 * hidden_size, 0.0, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False)
        self.type_emb = nn.Embedding(3, hidden_size)
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, 1)
        )
        self.mask_token_id = mask_token_id
        self.qtype = qtype

    def get_supported_tasks(self) -> Set[str]:
        return {"classify"}

    def get_pooling_updates(self, task: str) -> PoolingParamsUpdate:
        # Option markers are found from the prompt token ids.
        return PoolingParamsUpdate(requires_token_ids=True)

    def forward(self, hidden_states: torch.Tensor, pooling_metadata) -> list[torch.Tensor]:
        cursor = pooling_metadata.get_pooling_cursor()
        if cursor.is_partial_prefill():
            raise RuntimeError("LAYA judge is encoder-only: the whole prompt must run in one step")
        # vLLM fills the CPU copy when requires_token_ids is set.
        token_ids = pooling_metadata.prompt_token_ids_cpu
        if token_ids is None:
            token_ids = pooling_metadata.prompt_token_ids
        if token_ids is None:
            raise RuntimeError("LAYA judge needs the prompt token ids (requires_token_ids)")
        chunks = torch.split(hidden_states, cursor.num_scheduled_tokens_cpu.tolist())
        type_vec = self.type_emb.weight[self.qtype]
        outputs: list[torch.Tensor] = []
        for i, h in enumerate(chunks):
            ids = token_ids[i, : h.shape[0]]
            x = (h + type_vec).to(self.type_emb.weight.dtype).unsqueeze(0)
            x = self.head(x)[0]
            markers = (ids == self.mask_token_id).nonzero(as_tuple=True)[0].to(x.device)
            outputs.append(self.scorer(x[markers]).squeeze(-1).float())
        return outputs


def laya_question_type(config: object) -> str | int:
    """The question type the head scores, from ``laya_question_type`` or the
    ``response_judge.question_type`` the prompt is built with; they must agree."""
    model_qtype = getattr(config, "laya_question_type", None)
    judge = getattr(config, "response_judge", None)
    judge_qtype = judge.get("question_type") if isinstance(judge, dict) else None
    if model_qtype is not None and judge_qtype is not None and str(model_qtype) != str(judge_qtype):
        raise ValueError(
            f"laya_question_type={model_qtype!r} does not match response_judge.question_type={judge_qtype!r}"
        )
    return model_qtype if model_qtype is not None else (judge_qtype or "choice")


@attn_type("encoder_only")
@default_pooling_type(seq_pooling_type="CLS")
class LayaDecisionModel(nn.Module):
    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.encoder = ModernBertModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "encoder"))
        qtype = laya_question_type(config)
        self.pooler = LayaDecisionPooler(
            config.hidden_size,
            int(getattr(config, "laya_head_layers", 2)),
            int(config.mask_token_id),
            LAYA_QTYPES[qtype] if isinstance(qtype, str) else int(qtype),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        # The omni runner also passes bookkeeping kwargs (sampling_metadata, ...).
        return self.encoder(input_ids=input_ids, positions=positions, inputs_embeds=inputs_embeds)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        encoder_weights: list[tuple[str, torch.Tensor]] = []
        head_params = dict(self.pooler.named_parameters())
        loaded: set[str] = set()
        for name, tensor in weights:
            if name.startswith("encoder."):
                encoder_weights.append((name[len("encoder.") :], tensor))
            elif name in head_params:
                with torch.no_grad():
                    head_params[name].copy_(tensor.to(head_params[name].dtype))
                loaded.add(f"pooler.{name}")
            # act_head.* and the temperature buffer are not used by the judge.
        loaded |= {f"encoder.{n}" for n in self.encoder.load_weights(encoder_weights)}
        return loaded
