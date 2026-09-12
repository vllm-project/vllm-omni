# SPDX-License-Identifier: Apache-2.0  # noqa: N999
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Top-level package for comfyui_vllm_omni."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """vLLM-Omni Team"""
__email__ = "vllm-omni@vllm.ai"
__version__ = "0.0.1"

import logging
from urllib.parse import urlparse

import aiohttp
from aiohttp import web

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server route: proxy model list from a vLLM-Omni server
# ---------------------------------------------------------------------------
# The PromptServer import is guarded because this module may be imported
# outside of a running ComfyUI process (e.g. during linting or testing).
try:
    from server import PromptServer  # type: ignore[import-untyped]

    @PromptServer.instance.routes.get("/vllm_omni/models")
    async def get_vllm_models(request: web.Request) -> web.Response:
        """Proxy endpoint to fetch available models from a vLLM-Omni server.

        Query Parameters:
            url: Base URL of the vLLM-Omni server (e.g. ``http://localhost:8000/v1``).

        Returns:
            JSON object with a ``models`` key containing a list of model ID strings.
        """
        url = request.query.get("url", "").strip()
        if not url:
            return web.json_response({"error": "Missing 'url' query parameter"}, status=400)

        # Basic SSRF protection: only allow http(s) schemes.
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return web.json_response({"error": f"Unsupported URL scheme: {parsed.scheme}"}, status=400)

        models_url = url.rstrip("/") + "/models"
        timeout = aiohttp.ClientTimeout(total=5)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(models_url) as response:
                    if not response.ok:
                        return web.json_response(
                            {"error": f"Server returned status {response.status}"},
                            status=502,
                        )
                    data = await response.json()
                    model_ids = [m["id"] for m in data.get("data", [])]
                    return web.json_response({"models": model_ids})
        except Exception:
            logger.debug("Failed to fetch models from %s", models_url, exc_info=True)
            return web.json_response(
                {"error": f"Could not connect to vLLM-Omni server at {url}"},
                status=502,
            )

except ImportError:
    logger.debug("PromptServer not available; skipping route registration.")

# ---------------------------------------------------------------------------
# Node registration
# ---------------------------------------------------------------------------
from .comfyui_vllm_omni.nodes import (  # noqa: E402
    VLLMOmniARSampling,
    VLLMOmniDiffusionSampling,
    VLLMOmniGenerateImage,
    VLLMOmniGenerateVideo,
    VLLMOmniMiniMaxH3Params,
    VLLMOmniQwenTTSParams,
    VLLMOmniRemoteLoRA,
    VLLMOmniSamplingParamsList,
    VLLMOmniTTS,
    VLLMOmniUnderstanding,
    VLLMOmniVideoReferences,
    VLLMOmniVoiceClone,
    VLLMOmniWanParams,
)

NODE_CLASS_MAPPINGS = {
    # === Generation ===
    "VLLMOmniGenerateImage": VLLMOmniGenerateImage,
    "VLLMOmniGenerateVideo": VLLMOmniGenerateVideo,
    "VLLMOmniUnderstanding": VLLMOmniUnderstanding,
    "VLLMOmniTTS": VLLMOmniTTS,
    "VLLMOmniVoiceClone": VLLMOmniVoiceClone,
    "VLLMOmniVideoReferences": VLLMOmniVideoReferences,
    # === Params ===
    "VLLMOmniARSampling": VLLMOmniARSampling,
    "VLLMOmniDiffusionSampling": VLLMOmniDiffusionSampling,
    "VLLMOmniSamplingParamsList": VLLMOmniSamplingParamsList,
    "VLLMOmniRemoteLoRA": VLLMOmniRemoteLoRA,
    "VLLMOmniQwenTTSParams": VLLMOmniQwenTTSParams,
    "VLLMOmniWanParams": VLLMOmniWanParams,
    "VLLMOmniMiniMaxH3Params": VLLMOmniMiniMaxH3Params,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # === Generation ===
    "VLLMOmniGenerateImage": "Generate Image",
    "VLLMOmniGenerateVideo": "Generate Video",
    "VLLMOmniUnderstanding": "Multimodality Understanding",
    "VLLMOmniTTS": "TTS (Text to Speech)",
    "VLLMOmniVoiceClone": "TTS Voice Cloning",
    "VLLMOmniVideoReferences": "Video References",
    # === Params ===
    "VLLMOmniARSampling": "AR Sampling Params",
    "VLLMOmniDiffusionSampling": "Diffusion Sampling Params",
    "VLLMOmniSamplingParamsList": "Multi-Stage Sampling Params List",
    "VLLMOmniRemoteLoRA": "LoRA",
    "VLLMOmniQwenTTSParams": "Qwen TTS Params",
    "VLLMOmniWanParams": "Wan Video Params",
    "VLLMOmniMiniMaxH3Params": "MiniMax-H3 Video Params",
}

WEB_DIRECTORY = "./web"
