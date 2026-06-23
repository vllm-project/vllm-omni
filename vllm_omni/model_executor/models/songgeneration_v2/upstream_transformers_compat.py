# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compatibility shims for lazy-importing upstream SongGeneration code.

Upstream ``Flow1dVAE`` was written against Transformers 4.x. vLLM-Omni pins
Transformers 5.x, where several GPT-2 helper symbols moved or were removed.
Call :func:`ensure_upstream_transformers_compat` before
``from generate_septoken import Tango``.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from math import ceil
from typing import Any

import torch
import torch.nn as nn
from transformers.activations import get_activation
from transformers.configuration_utils import PretrainedConfig


def ensure_upstream_transformers_compat() -> None:
    """Backport missing Transformers 4.x symbols used by upstream Flow1dVAE."""
    _ensure_transformers_onnx_stub()
    _ensure_sequence_summary()
    _ensure_pytorch_utils_pruning_helpers()
    _ensure_model_parallel_utils_module()
    _ensure_pretrained_model_legacy_helpers()
    _ensure_upstream_gpt2_config_defaults()


# TF4-era GPT-2 config fields referenced by upstream ``GPT2Block`` but not set
# by ``models_gpt.models.gpt2_config.GPT2Config`` (subclass of PretrainedConfig).
_UPSTREAM_GPT2_CONFIG_DEFAULTS: dict[str, object] = {
    "add_cross_attention": False,
    "_attn_implementation": "eager",
    "output_attentions": False,
    "output_hidden_states": False,
    "use_return_dict": True,
}


def _ensure_transformers_onnx_stub() -> None:
    """Stub ``transformers.onnx`` for TF5.x where the submodule was removed.

    Upstream ``models_gpt`` config modules import ``OnnxConfigWithPast`` at
    import time even though inference never exports ONNX.
    """
    name = "transformers.onnx"
    if name in sys.modules:
        return
    try:
        import transformers.onnx as _onnx  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    mod = types.ModuleType(name)

    class OnnxConfigWithPast:  # noqa: D101
        pass

    class PatchingSpec:  # noqa: D101
        pass

    mod.OnnxConfigWithPast = OnnxConfigWithPast
    mod.PatchingSpec = PatchingSpec
    sys.modules[name] = mod


def _ensure_sequence_summary() -> None:
    import transformers.modeling_utils as modeling_utils

    if hasattr(modeling_utils, "SequenceSummary"):
        return

    class SequenceSummary(nn.Module):
        """Minimal backport of Transformers 4.x ``SequenceSummary``."""

        def __init__(self, config: PretrainedConfig) -> None:
            super().__init__()
            self.summary_type = getattr(config, "summary_type", "last")
            if self.summary_type == "attn":
                raise NotImplementedError

            self.summary: nn.Module = nn.Identity()
            if getattr(config, "summary_use_proj", False):
                if getattr(config, "summary_proj_to_labels", False) and getattr(config, "num_labels", 0) > 0:
                    num_classes = config.num_labels
                else:
                    num_classes = config.hidden_size
                self.summary = nn.Linear(config.hidden_size, num_classes)

            activation_string = getattr(config, "summary_activation", None)
            self.activation: Callable = get_activation(activation_string) if activation_string else nn.Identity()

            self.first_dropout: nn.Module = nn.Identity()
            if getattr(config, "summary_first_dropout", 0) > 0:
                self.first_dropout = nn.Dropout(config.summary_first_dropout)

            self.last_dropout: nn.Module = nn.Identity()
            if getattr(config, "summary_last_dropout", 0) > 0:
                self.last_dropout = nn.Dropout(config.summary_last_dropout)

        def forward(
            self,
            hidden_states: torch.Tensor,
            cls_index: torch.LongTensor | None = None,
        ) -> torch.Tensor:
            if self.summary_type == "last":
                output = hidden_states[:, -1]
            elif self.summary_type == "first":
                output = hidden_states[:, 0]
            elif self.summary_type == "mean":
                output = hidden_states.mean(dim=1)
            elif self.summary_type == "cls_index":
                if cls_index is None:
                    cls_index = torch.full_like(
                        hidden_states[..., :1, :],
                        hidden_states.shape[-2] - 1,
                        dtype=torch.long,
                    )
                else:
                    cls_index = cls_index.unsqueeze(-1).unsqueeze(-1)
                    cls_index = cls_index.expand((-1,) * (cls_index.dim() - 1) + (hidden_states.size(-1),))
                output = hidden_states.gather(-2, cls_index).squeeze(-2)
            else:
                raise NotImplementedError(self.summary_type)

            output = self.first_dropout(output)
            output = self.summary(output)
            output = self.activation(output)
            output = self.last_dropout(output)
            return output

    modeling_utils.SequenceSummary = SequenceSummary  # type: ignore[attr-defined]


def _ensure_pytorch_utils_pruning_helpers() -> None:
    import transformers.pytorch_utils as pytorch_utils

    if hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
        return

    Conv1D = pytorch_utils.Conv1D

    def prune_conv1d_layer(layer: Conv1D, index: torch.LongTensor, dim: int = 1) -> Conv1D:
        index = index.to(layer.weight.device)
        w = layer.weight.index_select(dim, index).clone().detach()
        if dim == 0:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
        new_size = list(layer.weight.size())
        new_size[dim] = len(index)
        new_layer = Conv1D(new_size[1], new_size[0]).to(layer.weight.device)
        new_layer.weight.requires_grad = False
        new_layer.weight.copy_(w.contiguous())
        new_layer.weight.requires_grad = True
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
        return new_layer

    def find_pruneable_heads_and_indices(
        heads: list[int],
        n_heads: int,
        head_size: int,
        already_pruned_heads: set[int],
    ) -> tuple[set[int], torch.LongTensor]:
        mask = torch.ones(n_heads, head_size)
        heads_set = set(heads) - already_pruned_heads
        for head in heads_set:
            head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index: torch.LongTensor = torch.arange(len(mask))[mask].long()
        return heads_set, index

    pytorch_utils.prune_conv1d_layer = prune_conv1d_layer  # type: ignore[attr-defined]
    pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices  # type: ignore[attr-defined]


def _ensure_model_parallel_utils_module() -> None:
    name = "transformers.utils.model_parallel_utils"
    if name in sys.modules:
        return

    mod = types.ModuleType(name)

    def assert_device_map(device_map, num_blocks):  # noqa: ANN001
        blocks = list(range(0, num_blocks))
        device_map_blocks = [item for sublist in list(device_map.values()) for item in sublist]
        duplicate_blocks = []
        for i in device_map_blocks:
            if device_map_blocks.count(i) > 1 and i not in duplicate_blocks:
                duplicate_blocks.append(i)
        missing_blocks = [i for i in blocks if i not in device_map_blocks]
        extra_blocks = [i for i in device_map_blocks if i not in blocks]
        if duplicate_blocks:
            raise ValueError("Duplicate attention blocks specified in device_map: " + str(duplicate_blocks))
        if missing_blocks:
            raise ValueError("Attention blocks not in device_map: " + str(missing_blocks))
        if extra_blocks:
            raise ValueError("device_map has extra attention blocks: " + str(extra_blocks))

    def get_device_map(n_layers, devices):  # noqa: ANN001
        layers = list(range(n_layers))
        n_blocks = int(ceil(n_layers / len(devices)))
        layers_list = [layers[i : i + n_blocks] for i in range(0, n_layers, n_blocks)]
        return dict(zip(devices, layers_list))

    mod.assert_device_map = assert_device_map
    mod.get_device_map = get_device_map
    sys.modules[name] = mod


def _ensure_pretrained_model_legacy_helpers() -> None:
    """Backport TF4 ``PreTrainedModel`` helpers removed in Transformers 5.x."""
    import transformers.modeling_utils as modeling_utils

    if hasattr(modeling_utils.PreTrainedModel, "get_head_mask"):
        return

    def _convert_head_mask_to_5d(
        self,
        head_mask: torch.Tensor,
        num_hidden_layers: int,
    ) -> torch.Tensor:
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        return head_mask.to(dtype=self.dtype)

    def get_head_mask(
        self,
        head_mask: torch.Tensor | None,
        num_hidden_layers: int,
        is_attention_chunked: bool = False,
    ) -> Any:
        if head_mask is not None:
            head_mask = _convert_head_mask_to_5d(self, head_mask, num_hidden_layers)
            if is_attention_chunked:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask

    def invert_attention_mask(self, encoder_attention_mask: torch.Tensor) -> torch.Tensor:
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        else:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)
        return (1.0 - encoder_extended_attention_mask) * torch.finfo(self.dtype).min

    def warn_if_padding_and_no_attention_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> None:
        del self, input_ids, attention_mask

    modeling_utils.PreTrainedModel._convert_head_mask_to_5d = _convert_head_mask_to_5d  # type: ignore[attr-defined]
    modeling_utils.PreTrainedModel.get_head_mask = get_head_mask  # type: ignore[attr-defined]
    modeling_utils.PreTrainedModel.invert_attention_mask = invert_attention_mask  # type: ignore[attr-defined]
    modeling_utils.PreTrainedModel.warn_if_padding_and_no_attention_mask = (  # type: ignore[attr-defined]
        warn_if_padding_and_no_attention_mask
    )


def _ensure_upstream_gpt2_config_defaults() -> None:
    """Patch upstream ``GPT2Config`` so TF5 ``PretrainedConfig.__getattribute__`` works.

    Upstream ``model_septoken`` builds ``GPT2Config(...)`` from
    ``models_gpt.models.gpt2_config``, which omits several fields that the
    vendored ``GPT2Block`` reads at init time (e.g. ``add_cross_attention``).
    """
    name = "models_gpt.models.gpt2_config"
    mod = sys.modules.get(name)
    if mod is None:
        try:
            import models_gpt.models.gpt2_config as mod  # type: ignore
        except ModuleNotFoundError:
            return
    gpt2_config_cls = getattr(mod, "GPT2Config", None)
    if gpt2_config_cls is None or getattr(gpt2_config_cls, "_vllm_omni_compat_patched", False):
        return

    _orig_init = gpt2_config_cls.__init__

    def _patched_init(self, *args, **kwargs):  # noqa: ANN001
        _orig_init(self, *args, **kwargs)
        for key, value in _UPSTREAM_GPT2_CONFIG_DEFAULTS.items():
            if not hasattr(self, key):
                setattr(self, key, value)
        if getattr(self, "_attn_implementation", None) is None:
            self._attn_implementation = "eager"

    gpt2_config_cls.__init__ = _patched_init  # type: ignore[method-assign]
    gpt2_config_cls._vllm_omni_compat_patched = True
