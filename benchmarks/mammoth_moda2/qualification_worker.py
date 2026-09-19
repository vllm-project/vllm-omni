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
    def qualification_runtime_all_ranks(self, dlo):
        """Return every worker's record through the control RPC's rank-zero reply."""
        from vllm_omni.diffusion.distributed.parallel_state import get_world_group

        group = get_world_group()
        # Exchange errors as data so a failed local assertion cannot strand a
        # healthy peer inside the reporting collective.
        try:
            local = {"runtime": self.qualification_runtime(dlo)}
        except Exception as exc:
            local = {"error": f"rank {self.rank}: {type(exc).__name__}: {exc}"}
        records = [None] * group.world_size
        torch.distributed.all_gather_object(records, local, group=group.cpu_group)
        errors = [record["error"] for record in records if "error" in record]
        if errors:
            raise RuntimeError("; ".join(errors))
        return [record["runtime"] for record in records]

    def qualification_runtime(self, dlo):
        pipeline = self.model_runner.pipeline
        backend = self.model_runner.offload_backend
        expected = len(pipeline.gen_transformer.layers)
        groups = getattr(backend, "_all_hook_groups", [])
        if dlo != "none":
            from vllm_omni.diffusion.offloader.distributed_layerwise_backend import DistributedLayerwiseOffloadBackend

            assert isinstance(backend, DistributedLayerwiseOffloadBackend) and backend.enabled
            assert len(groups) == 1 and len(groups[0]) == expected
            transport_size = self.od_config.parallel_config.ulysses_degree if dlo == "allgather" else 1
            assert all(hook.dp_size == transport_size for hook in groups[0])
        else:
            assert not groups
        return {
            "rank": self.rank,
            "dlo": dlo,
            "main_layers": expected,
            "offload_backend": type(backend).__name__,
            "offload_ring_layers": [len(group) for group in groups],
            "transport_sizes": [[hook.dp_size for hook in group] for group in groups],
        }

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
