# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from vllm.v1.core.sched.output import SchedulerOutput

from vllm_omni.core.sched.omni_generation_scheduler import (
    OmniGenerationScheduler,
)


class MiniCPMO45Code2WavScheduler(OmniGenerationScheduler):
    """Briefly coalesce same-wave codec chunks before Code2Wav scheduling."""

    _POLL_INTERVAL_S = 0.0001

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._code2wav_batch_wait_s = self._batch_wait_seconds(self.vllm_config.model_config)

    @staticmethod
    def _batch_wait_seconds(model_config: Any) -> float:
        connector = getattr(model_config, "stage_connector_config", None)
        if isinstance(connector, Mapping):
            extra = connector.get("extra", connector)
        else:
            extra = getattr(connector, "extra", None)
        if not isinstance(extra, Mapping):
            extra = {}
        wait_ms = float(extra.get("code2wav_batch_wait_ms", 0.0) or 0.0)
        if wait_ms < 0:
            raise ValueError("MiniCPM-o Code2Wav code2wav_batch_wait_ms must be non-negative")
        return wait_ms / 1000.0

    def _ready_and_target_chunk_counts(self) -> tuple[int, int]:
        adapter = self.chunk_transfer_adapter
        if adapter is None:
            return 0, 0
        ready = len(adapter._finished_load_reqs)
        target = min(self.max_num_running_reqs, len(self.requests))
        return ready, target

    def _wait_for_ready_chunk_batch(self) -> None:
        if self._code2wav_batch_wait_s <= 0:
            return

        ready, target = self._ready_and_target_chunk_counts()
        if ready <= 0 or target <= 1 or ready >= target:
            return

        deadline = time.monotonic() + self._code2wav_batch_wait_s
        while ready < target:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(self._POLL_INTERVAL_S, remaining))
            ready, target = self._ready_and_target_chunk_counts()

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self._wait_for_ready_chunk_batch()
        return super().schedule(throttle_prefills)
