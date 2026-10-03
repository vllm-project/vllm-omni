# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Worker-side execution evidence for encoder graph E2E tests."""

from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner


class EncoderGraphProbe:
    model_runner: OmniGPUModelRunner

    def encoder_graph_state(self) -> tuple[bool, int, int]:
        manager = self.model_runner.encoder_cudagraph_manager
        if manager is None:
            return False, 0, 0
        return manager.is_captured(), sum(len(graphs) for graphs in manager.budget_graphs.values()), manager.graph_hits
