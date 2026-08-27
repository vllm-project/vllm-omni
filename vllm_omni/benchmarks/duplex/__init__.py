# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU-friendly Omni-DuplexEval generation helpers."""

from .omni_duplex_eval_dataset import DuplexSample, load_samples

__all__ = ["DuplexSample", "load_samples"]
