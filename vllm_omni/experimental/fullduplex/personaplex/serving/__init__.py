# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PersonaPlex full-duplex WebSocket serving (PCM in / PCM + text out)."""

from typing import Any

__all__ = ["create_app", "main"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from vllm_omni.experimental.fullduplex.personaplex.serving import server

    return getattr(server, name)
