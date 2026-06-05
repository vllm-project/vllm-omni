# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vendored VAE / audio-VAE / vocoder modules from JoyAI-Echo upstream
(``ltx-core/src/ltx_core/model``). These ship raw Lightricks-internal
architectures (no ``mid_block``, separate ``conv`` upsamplers) and therefore
cannot be replaced by diffusers' ``AutoencoderKLLTX2Video`` /
``AutoencoderKLLTX2Audio`` / ``LTX2VocoderWithBWE``.
"""
