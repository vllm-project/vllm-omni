# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from vllm_omni.diffusion.models.interface import ProxyStreamSpec


def invoker_render_proxy_stream_spec(*, model: str | None, revision: str | None = None) -> ProxyStreamSpec:
    del revision
    from vllm_omni.diffusion.models.invoker_render.assets import InvokerRenderAssets
    from vllm_omni.diffusion.models.invoker_render.proxy import invoker_render_proxy_spec

    if not model:
        raise RuntimeError("Missing prepared Invoker-Render bundle for streaming")
    return invoker_render_proxy_spec(InvokerRenderAssets.read(model).proxy_size)
