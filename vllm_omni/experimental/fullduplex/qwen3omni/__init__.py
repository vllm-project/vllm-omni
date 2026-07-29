# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Experimental Qwen3-Omni full-duplex integration.

Status: engine/serving layers implemented and contract-validated; the
worker-side stage-0 audio embedding path is NOT implemented. See
``stage0.py`` and ``docs/design/qwen3_omni_duplex_assessment.md``.

This package mirrors the structure of the MiniCPM-o 4.5 integration in
``vllm_omni/experimental/fullduplex/minicpmo45/``, which is the only
working reference for a native multi-stage duplex model.
"""
