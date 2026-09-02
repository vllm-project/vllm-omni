# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Compatibility helpers for vLLM model weight loading."""

from collections.abc import Iterable

import torch
from torch import nn
from vllm.model_executor.models.utils import (
    AutoWeightsLoader as _VllmAutoWeightsLoader,
)
from vllm.model_executor.models.utils import (
    WeightsMapper,
)


class AutoWeightsLoader(_VllmAutoWeightsLoader):
    """Preserve vLLM's pre-0.29 weight-filtering arguments.

    vLLM removed ``skip_prefixes`` and ``skip_substrs`` from
    :class:`AutoWeightsLoader` in favor of mapper-based filtering. Omni models
    still use those arguments to exclude checkpoint components owned by a
    different stage. Convert the filters to a ``WeightsMapper`` so the old
    behavior is retained without bypassing the current loader implementation.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        skip_prefixes: list[str] | None = None,
        skip_substrs: list[str] | None = None,
        ignore_unexpected_prefixes: list[str] | None = None,
        ignore_unexpected_suffixes: list[str] | None = None,
    ) -> None:
        super().__init__(
            module,
            ignore_unexpected_prefixes=ignore_unexpected_prefixes,
            ignore_unexpected_suffixes=ignore_unexpected_suffixes,
        )
        self._skip_mapper = WeightsMapper(
            orig_to_new_prefix={prefix: None for prefix in skip_prefixes or ()},
            orig_to_new_substr={substring: None for substring in skip_substrs or ()},
        )

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        mapper: WeightsMapper | None = None,
    ) -> set[str]:
        mapper = (mapper or WeightsMapper()) | self._skip_mapper
        return super().load_weights(weights, mapper=mapper)
