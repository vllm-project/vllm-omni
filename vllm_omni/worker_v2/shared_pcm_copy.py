# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Copy request PCM views sharing one owned batch allocation together."""

from collections import defaultdict
from collections.abc import Callable

import torch


def copy_shared_pcm_views(
    values: list[torch.Tensor], copy_tensor: Callable[[torch.Tensor], torch.Tensor]
) -> list[torch.Tensor]:
    if len(values) < 2:
        return [copy_tensor(value) for value in values]
    groups = defaultdict(list)
    result = [None] * len(values)
    for index, value in enumerate(values):
        if not isinstance(value, torch.Tensor):
            raise TypeError("Batched PCM copies require a tensor list")
        if value.layout != torch.strided or value.is_quantized or value.numel() == 0:
            result[index] = copy_tensor(value)
            continue
        key = (value.device, value.dtype, value.untyped_storage().data_ptr())
        groups[key].append(index)

    for indices in groups.values():
        spans = sorted(
            (
                values[i].storage_offset(),
                values[i].storage_offset() + 1 + sum((n - 1) * s for n, s in zip(values[i].shape, values[i].stride())),
            )
            for i in indices
        )
        begin, end = spans[0][0], max(stop for _, stop in spans)
        overlap = any(stop > following_start for (_, stop), (following_start, _) in zip(spans, spans[1:]))
        # Bound copying gaps in sparse views; unrelated/isolated outputs keep
        # their ordinary per-tensor copy behavior.
        if len(indices) == 1 or overlap or end - begin > 2 * sum(values[i].numel() for i in indices):
            for index in indices:
                result[index] = copy_tensor(values[index])
            continue
        span = values[indices[0]].as_strided((end - begin,), (1,), begin)
        host = copy_tensor(span)
        for index in indices:
            value = values[index]
            result[index] = host.as_strided(value.shape, value.stride(), value.storage_offset() - begin)
    return result
