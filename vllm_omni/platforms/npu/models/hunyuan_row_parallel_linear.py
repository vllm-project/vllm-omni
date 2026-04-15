# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import vllm.forward_context as _vllm_fc
from vllm_ascend.ops.linear import AscendRowParallelLinear
from vllm.distributed import get_tensor_model_parallel_world_size

def _ensure_forward_context_attr(name: str, annotation: Any, default: Any) -> None:
    if name not in _vllm_fc.ForwardContext.__annotations__:
        _vllm_fc.ForwardContext.__annotations__[name] = annotation
    if not hasattr(_vllm_fc.ForwardContext, name):
        setattr(_vllm_fc.ForwardContext, name, default)


def prepare_hunyuan_row_parallel_linear_runtime() -> None:
    _ensure_forward_context_attr("row_parallel_linear_fc1_enabled", bool, False)
    _ensure_forward_context_attr("mmrs_fusion", bool, False)
    _ensure_forward_context_attr("fc1_pad_size", bool, 0)


def _set_hunyuan_row_parallel_linear_forward_context(num_tokens: int) -> None:
    if not _vllm_fc.is_forward_context_available():
        return

    forward_context = _vllm_fc.get_forward_context()
    # forward_context.num_tokens = num_tokens
    import os
    forward_context.row_parallel_linear_fc1_enabled = bool(int(os.getenv("HY_IMAGE_3_ENABLE_FLASHCOMM1", 0)))
    forward_context.mmrs_fusion = True
    if forward_context.row_parallel_linear_fc1_enabled:
        tp_world_size = get_tensor_model_parallel_world_size()
        fc1_pad_size = (tp_world_size - (num_tokens % tp_world_size)) % tp_world_size
        forward_context.fc1_pad_size = fc1_pad_size

class AscendHunyuanRowParallelLinear(AscendRowParallelLinear):
    def __init__(self, *args: Any, prefix: str = "", **kwargs: Any) -> None:
        super().__init__(*args, prefix=prefix, **kwargs)
        self._prefix = prefix
        self._init_hook_handle = self.register_forward_pre_hook(
            self._initialize_kernel_hook,
            with_kwargs=True,
        )

    def _initialize_kernel_hook(self, module: Any, args: Any, kwargs: Any) -> None:
        if self.quant_method:
            self.quant_method.process_weights_after_loading(self)
        self._init_hook_handle.remove()

    def forward(
        self,
        input_: Any,
        **kwargs: Any,
    ) -> Any:
        _set_hunyuan_row_parallel_linear_forward_context(input_.shape[0])
        return super().forward(input_, **kwargs)

