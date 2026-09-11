# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Observation-only worker extension for frozen-checkpoint qualification.

Hooks are active only for the unmeasured warmup. Measured runs use the native
pipeline without hooks; RPCs read/reset counters outside the timed region.
"""

import json
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import save_file

from vllm_omni.platforms import current_omni_platform


class QualificationWorkerExtension:
    def qualification_memory(self, reset=False):
        current_omni_platform.synchronize()
        if reset:
            current_omni_platform.reset_peak_memory_stats()
        return {
            "rank": self.rank,
            "peak_allocated_bytes": current_omni_platform.max_memory_allocated(),
            "peak_reserved_bytes": current_omni_platform.max_memory_reserved(),
        }

    def qualification_observe(self, directory):
        root = Path(directory) / f"rank-{self.rank}"
        root.mkdir(parents=True, exist_ok=False)
        pipeline = self.model_runner.pipeline
        transformer = pipeline.gen_transformer
        dtypes = {}
        for name in ("gen_transformer", "gen_image_condition_refiner", "gen_vae"):
            counts = Counter()
            for parameter in getattr(pipeline, name).parameters():
                counts[str(parameter.dtype)] += parameter.numel()
            dtypes[name] = dict(counts)
        (root / "model.json").write_text(json.dumps({"parameter_dtypes": dtypes}, indent=2))
        counter = {"calls": 0}

        def initial_inputs(module, args, kwargs):
            if counter["calls"] == 0:
                tensors = {
                    name: value.detach().cpu().contiguous()
                    for name, value in kwargs.items()
                    if isinstance(value, torch.Tensor)
                }
                save_file(tensors, str(root / "initial_inputs.safetensors"))

        def predictions(module, args, output):
            index = counter["calls"]
            if index in (0, 1, 48, 49, 98, 99):
                save_file(
                    {"prediction": output.detach().cpu().contiguous()},
                    str(root / f"prediction-{index:03d}.safetensors"),
                )
            counter["calls"] += 1

        def before_decode(module, args):
            # AutoencoderKL.decode is a method, so the module's decoder is the
            # actual forward boundary. Its input follows post_quant_conv.
            save_file(
                {"decoder_input": args[0].detach().cpu().contiguous()},
                str(root / "decoder_input.safetensors"),
            )

        self.qualification_handles = [
            transformer.register_forward_pre_hook(initial_inputs, with_kwargs=True),
            transformer.register_forward_hook(predictions),
            pipeline.gen_vae.decoder.register_forward_pre_hook(before_decode),
        ]
        return {"rank": self.rank, "hooks": len(self.qualification_handles)}

    def qualification_remove_observers(self):
        for handle in self.qualification_handles:
            handle.remove()
        self.qualification_handles = []
        return {"rank": self.rank, "hooks": 0}
