# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path

import regex as re
from safetensors import safe_open
from torch import nn
from vllm.model_executor.models.utils import WeightsMapper

_DIT_NAMES = WeightsMapper(
    orig_to_new_regex={
        re.compile(r"^double_blocks\.(\d+)\.action_module\.(.+)$"): r"action_modules.\1.\2",
        re.compile(r"\._checkpoint_wrapped_module\."): ".",
        re.compile(r"_checkpoint_wrapped_module\."): "",
    },
)


def validate_dit_checkpoint(transformer: nn.Module, checkpoint: Path) -> None:
    # The training loader allows new camera parameters; distilled inference requires every parameter.
    with safe_open(str(checkpoint), framework="pt", device="cpu") as weights:
        loaded_names = set(_DIT_NAMES.apply_list(list(weights.keys())))
    missing = set(dict(transformer.named_parameters())) - loaded_names
    if missing:
        raise ValueError(f"Hy-Nomas checkpoint is missing {len(missing)} DiT parameters: {sorted(missing)[:10]}")
