# SPDX-License-Identifier: Apache-2.0
"""Optional MagiCompiler integration for vLLM-Omni (opt-in via environment).

When enabled, wraps hot submodules used by **Qwen2.5-Omni** pipeline stages:

- **Thinker / Talker**: each ``Qwen2DecoderLayer`` under ``language_model.model.layers``.
- **Token2Wav DiT**: each ``DiTDecoderLayer`` in ``transformer_blocks``.
- **Token2Wav BigVGAN**: each ``AMPBlock`` in ``resblocks``.

Enable::

    export VLLM_OMNI_MAGI_COMPILE=1

Requires a separate install of ``magi_compiler`` (see project optional extra ``[magi]``).
"""

from __future__ import annotations

import os

import torch.nn as nn
from vllm.logger import init_logger

logger = init_logger(__name__)

_ENV_MAGI = "VLLM_OMNI_MAGI_COMPILE"

# DiTDecoderLayer.forward(hidden_states, timestep, position_embeddings=None, block_diff=None)
# hidden_states: (B, T, C); timestep: (B, D); block_diff: (B, H, T, T); RoPE tuple traced as-is.
_DIT_DECODER_DYNAMIC_ARG_DIMS = {
    "hidden_states": 1,
    "timestep": 0,
    "block_diff": [2, 3],
}

# AMPBlock.forward(hidden_states) on Conv1d layout (B, C, L).
_BIGVGAN_AMP_DYNAMIC_ARG_DIMS = {
    "hidden_states": 2,
}


def is_magi_compile_requested() -> bool:
    """Return True if Magi compilation is enabled via :envvar:`VLLM_OMNI_MAGI_COMPILE`."""
    val = os.environ.get(_ENV_MAGI, "").strip().lower()
    return val in ("1", "true", "yes", "on")


def apply_magi_to_qwen_decoder_layers(
    language_model: nn.Module,
    *,
    model_tag_prefix: str = "qwen2_5_omni_thinker",
) -> int:
    """Wrap each vLLM ``Qwen2DecoderLayer`` with :func:`magi_compiler.magi_compile`.

    Expects ``language_model`` to follow the usual ``Qwen2ForCausalLM`` layout:
    ``language_model.model.layers`` is ``ModuleList`` of decoder blocks.

    Dynamic shapes follow vLLM conventions: token dimension 0 on activations; ``positions``
    uses the last dimension (1-D positions or MRoPE ``(3, seq)``).

    Returns:
        Number of layers wrapped, or ``0`` if disabled, unavailable, or nothing to wrap.
    """
    if not is_magi_compile_requested():
        return 0

    try:
        from magi_compiler import magi_compile
    except ImportError:
        logger.warning_once(
            "%s is set but `magi_compiler` is not installed. "
            "Install the project extra `[magi]` or `pip install` MagiCompiler from source.",
            _ENV_MAGI,
        )
        return 0

    inner = getattr(language_model, "model", None)
    if inner is None:
        logger.warning("MagiCompiler: `language_model` has no `model`; skipping.")
        return 0

    layers = getattr(inner, "layers", None)
    if not layers or len(layers) == 0:
        logger.warning("MagiCompiler: no `model.layers` found; skipping.")
        return 0

    # Aligns with vLLM `Qwen2DecoderLayer.forward(positions, hidden_states, residual)`.
    dynamic_arg_dims = {
        "positions": -1,
        "hidden_states": 0,
        "residual": 0,
    }

    for i, layer in enumerate(layers):
        magi_compile(
            layer,
            dynamic_arg_dims=dynamic_arg_dims,
            model_tag=f"{model_tag_prefix}_layer{i}",
        )

    logger.info(
        "MagiCompiler: applied to %d decoder layers (prefix=%s).",
        len(layers),
        model_tag_prefix,
    )
    return len(layers)


def apply_magi_to_dit_decoder_layers(
    dit_module: nn.Module,
    *,
    model_tag_prefix: str = "qwen2_5_omni_token2wav_dit",
) -> int:
    """Wrap each ``DiTDecoderLayer`` in ``dit_module.transformer_blocks`` with ``magi_compile``."""
    if not is_magi_compile_requested():
        return 0

    try:
        from magi_compiler import magi_compile
    except ImportError:
        logger.warning_once(
            "%s is set but `magi_compiler` is not installed. "
            "Install the project extra `[magi]` or `pip install` MagiCompiler from source.",
            _ENV_MAGI,
        )
        return 0

    blocks = getattr(dit_module, "transformer_blocks", None)
    if not blocks or len(blocks) == 0:
        logger.warning("MagiCompiler: DiT has no `transformer_blocks`; skipping.")
        return 0

    for i, block in enumerate(blocks):
        magi_compile(
            block,
            dynamic_arg_dims=_DIT_DECODER_DYNAMIC_ARG_DIMS,
            model_tag=f"{model_tag_prefix}_layer{i}",
        )

    logger.info(
        "MagiCompiler: applied to %d DiT blocks (prefix=%s).",
        len(blocks),
        model_tag_prefix,
    )
    return len(blocks)


def apply_magi_to_bigvgan_amp_blocks(
    bigvgan_module: nn.Module,
    *,
    model_tag_prefix: str = "qwen2_5_omni_token2wav_bigvgan",
) -> int:
    """Wrap each ``AMPBlock`` in ``bigvgan_module.resblocks`` with ``magi_compile``."""
    if not is_magi_compile_requested():
        return 0

    try:
        from magi_compiler import magi_compile
    except ImportError:
        logger.warning_once(
            "%s is set but `magi_compiler` is not installed. "
            "Install the project extra `[magi]` or `pip install` MagiCompiler from source.",
            _ENV_MAGI,
        )
        return 0

    resblocks = getattr(bigvgan_module, "resblocks", None)
    if not resblocks or len(resblocks) == 0:
        logger.warning("MagiCompiler: BigVGAN has no `resblocks`; skipping.")
        return 0

    for i, block in enumerate(resblocks):
        magi_compile(
            block,
            dynamic_arg_dims=_BIGVGAN_AMP_DYNAMIC_ARG_DIMS,
            model_tag=f"{model_tag_prefix}_amp{i}",
        )

    logger.info(
        "MagiCompiler: applied to %d BigVGAN AMP blocks (prefix=%s).",
        len(resblocks),
        model_tag_prefix,
    )
    return len(resblocks)
