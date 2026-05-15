# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

__all__ = [
    "omni_run_server",
    "build_async_omni",
    "omni_init_app_state",
    "OmniOpenAIServingChat",
]


def __getattr__(name: str):
    if name in {"omni_run_server", "build_async_omni", "omni_init_app_state"}:
        from vllm_omni.entrypoints.openai import api_server

        return getattr(api_server, name)
    if name == "OmniOpenAIServingChat":
        from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

        return OmniOpenAIServingChat
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
