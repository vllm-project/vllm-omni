# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in switch for bitwise-reproducible runs."""

import os

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

ENV_VAR = "VLLM_OMNI_DETERMINISTIC"


def is_deterministic_requested() -> bool:
    return os.environ.get(ENV_VAR, "0").strip().lower() in ("1", "true", "yes", "on")


def maybe_enable_determinism() -> None:
    """Restrict convolution to deterministic algorithms when ``VLLM_OMNI_DETERMINISTIC`` is set.

    cuDNN/MIOpen may pick convolution algorithms whose cross-workgroup reduction uses
    atomics, in which case the reduction order follows thread completion order and the
    output differs run to run even at a fixed shape. Models with a VAE or a convolutional
    image encoder therefore replay the same request differently.

    The guarantee is intra-request: the same request, under the same config, in the same
    process, produces bitwise identical output. Results are *not* made invariant to batch
    composition.

    Off by default -- the deterministic algorithms are slower. Intended for evaluation,
    accuracy CI and RL rollouts.
    """
    if not is_deterministic_requested():
        return
    if not torch.backends.cudnn.is_available():
        logger.warning("%s is set but cuDNN/MIOpen is unavailable; convolution is left as is.", ENV_VAR)
        return
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("%s=1: convolution restricted to deterministic algorithms.", ENV_VAR)
