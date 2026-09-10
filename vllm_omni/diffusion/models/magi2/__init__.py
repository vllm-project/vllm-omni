# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""SandAI MAGI-2 Preview diffusion support."""

from __future__ import annotations

from typing import Any

__all__ = ["Magi2Pipeline", "get_magi2_post_process_func"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .pipeline_magi2 import Magi2Pipeline, get_magi2_post_process_func

        return {
            "Magi2Pipeline": Magi2Pipeline,
            "get_magi2_post_process_func": get_magi2_post_process_func,
        }[name]
    raise AttributeError(name)
