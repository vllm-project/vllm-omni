# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 NVIDIA CORPORATION.

"""Lazy builder for the IndexTTS 2.5 BigVGAN fused activation."""

from functools import lru_cache
from pathlib import Path

import torch
from torch.utils import cpp_extension
from vllm.logger import init_logger

logger = init_logger(__name__)


@lru_cache(maxsize=1)
def load():
    """Compile/load the fused op once per architecture and process."""
    if not torch.cuda.is_available() or cpp_extension.CUDA_HOME is None:
        raise ImportError("CUDA toolkit is required for fused BigVGAN activation")
    major, minor = torch.cuda.get_device_capability()
    architecture = f"{major}{minor}"
    source_dir = Path(__file__).resolve().parent
    logger.info("Loading IndexTTS fused BigVGAN activation for sm%s", architecture)
    extension = cpp_extension.load(
        name=f"vllm_omni_indextts_alias_free_sm{architecture}",
        sources=[
            str(source_dir / "anti_alias_activation.cpp"),
            str(source_dir / "anti_alias_activation_cuda.cu"),
        ],
        extra_include_paths=[str(source_dir)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-gencode",
            f"arch=compute_{architecture},code=sm_{architecture}",
        ],
        verbose=False,
    )
    logger.info("IndexTTS fused BigVGAN activation ready for sm%s", architecture)
    return extension
