# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any


def is_transformer_block_module(name: str, module: Any) -> bool:
    """Return True for numbered modules under `transformer_blocks`."""
    return "transformer_blocks" in name and name.split(".")[-1].isdigit()
