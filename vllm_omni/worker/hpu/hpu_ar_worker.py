# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_gaudi.v1.worker.hpu_worker import HPUWorker

from vllm_omni.worker.hpu.hpu_ar_model_runner import HPUARModelRunner


class HPUARWorker(HPUWorker):
    """HPU AR worker for thinker/talker stages in Omni model."""

    def init_device(self) -> None:
        device = super().init_device()
        self.model_runner: HPUARModelRunner = HPUARModelRunner(self.vllm_config, device)
