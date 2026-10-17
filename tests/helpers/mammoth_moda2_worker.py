# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Read-only DLO runtime checks for the full-checkpoint MammothModa2 tests."""

from typing import Any


class QualificationWorkerExtension:
    model_runner: Any
    od_config: Any
    rank: int

    def qualification_runtime_all_ranks(self, dlo_mode: str) -> list[dict[str, Any]]:
        from vllm_omni.diffusion.offloader.distributed_layerwise_backend import DistributedLayerwiseOffloadBackend
        from vllm_omni.diffusion.worker.diffusion_worker import _run_and_gather_rank_values

        def inspect_runtime() -> dict[str, Any]:
            assert dlo_mode in ("ring", "allgather")
            pipeline = self.model_runner.pipeline
            backend = self.model_runner.offload_backend
            assert isinstance(backend, DistributedLayerwiseOffloadBackend) and backend.enabled
            layer_count = len(pipeline.gen_transformer.layers)
            groups = backend._all_hook_groups
            assert len(groups) == 1 and len(groups[0]) == layer_count
            transport_size = self.od_config.parallel_config.ulysses_degree if dlo_mode == "allgather" else 1
            assert all(hook.dp_size == transport_size for hook in groups[0])
            return {
                "rank": self.rank,
                "dlo": dlo_mode,
                "main_layers": layer_count,
                "offload_backend": type(backend).__name__,
                "offload_ring_layers": [len(group) for group in groups],
                "transport_sizes": [[hook.dp_size for hook in group] for group in groups],
            }

        # Gather failures as well as successful probes so a bad rank cannot
        # strand its peers in the control-plane collective.
        return _run_and_gather_rank_values("MammothModa2 DLO runtime qualification", inspect_runtime)
