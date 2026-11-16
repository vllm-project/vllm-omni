# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Read the live Qwen3-Omni thinker projection through its real worker."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner as LegacyRunner
    from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner


class Qwen3OmniMergeProbe:
    model_runner: "LegacyRunner | OmniARModelRunner"

    def merged_memory_probe(self, reset: bool = False) -> dict[str, int]:
        if reset:
            torch.accelerator.reset_peak_memory_stats()
        free, total = torch.accelerator.get_memory_info()
        return {
            "allocated": torch.accelerator.memory_allocated(),
            "reserved": torch.accelerator.memory_reserved(),
            "peak_allocated": torch.accelerator.max_memory_allocated(),
            "peak_reserved": torch.accelerator.max_memory_reserved(),
            "free": free,
            "total": total,
        }

    def merged_q_projection(self) -> dict:
        for name, module in self.model_runner.model.named_modules():
            if name == "thinker.language_model.model.layers.0.self_attn.qkv_proj":
                weight = module.weight
                return {
                    "name": name,
                    "input_size": module.input_size,
                    "q_rows": module.total_num_heads * module.head_size,
                    "data_ptr": weight.data_ptr(),
                    "sample": weight[:16, :16].float().sum().item(),
                }
        raise RuntimeError("Qwen3-Omni thinker qkv projection was not loaded")
