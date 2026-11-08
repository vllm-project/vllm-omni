# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Configuration helpers for the LingBot World pipeline."""

from __future__ import annotations


def _vae_decode_fast_path(model_config: dict) -> str | None:
    """Resolve ``lingbot_vae_decode_fast_path`` from the model config: ``None``, ``"exact"`` or ``"fused"``.

    Off by default. ``true``/``exact`` installs the decoder's exact fast path after the decode dtype and the
    spatial shard are configured (see ``wan_decoder_fast_path``): no output bit changes, only launches
    disappear. ``fused`` adds the channels_last decoder with the fused RMSNorm+SiLU kernel, which is not
    bit-exact and requires subjective video quality review before enabling.
    """
    value = model_config.get("lingbot_vae_decode_fast_path", False)
    if isinstance(value, bool) or value is None:
        return "exact" if value else None
    name = str(value).strip().lower()
    if name in ("", "0", "false", "no", "off", "none"):
        return None
    if name in ("1", "true", "yes", "on", "exact"):
        return "exact"
    if name == "fused":
        return "fused"
    raise ValueError(f"lingbot_vae_decode_fast_path must be off, exact or fused; got {value!r}")
