# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_gaudi.v1.worker.hpu_worker import HPUWorker

from vllm_omni.worker.hpu.hpu_generation_model_runner import HPUGenerationModelRunner


class HPUGenerationWorker(HPUWorker):
    """HPU generation worker for code2wav stage in Omni model."""

    def init_device(self):
        device = self._init_device()

        self.model_runner: HPUGenerationModelRunner = HPUGenerationModelRunner(self.vllm_config, device)
