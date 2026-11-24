# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Own and execute MindIE-SD distributed sparse attention resources."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger

from vllm_omni import platforms
from vllm_omni.diffusion.data import resolve_mindie_sparse_backend

if TYPE_CHECKING:
    import torch.distributed as dist
    from mindiesd.parallel import ParallelState

    from vllm_omni.diffusion.data import MindIEParallelConfig

logger = init_logger(__name__)


class MindIEParallelResource:
    """Own SDK state; drain work before collecting cached buffers at teardown."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        device: torch.device,
        config: MindIEParallelConfig,
        *,
        cp_group: dist.ProcessGroup | None = None,
        precision: str = "bf16",
        sparse_type: str = "rf_v3",
        bsa_backend: str | None = None,
    ):
        self.device = device
        self.config = config
        self.precision = precision
        self.sparse_type = sparse_type
        self.bsa_backend = resolve_mindie_sparse_backend(sparse_type, precision, bsa_backend)
        from mindiesd.parallel import ParallelState

        self.state: ParallelState | None = ParallelState(
            group=group,
            cp_group=cp_group,
            overlap_comm=config.overlap_comm,
            optimize_attention_pipeline=config.optimize_attention_pipeline,
            bsa_backend=self.bsa_backend,
            quantized_comm=config.quantized_comm,
        )
        self.failed = False
        self.active = False
        self.completed_calls = 0

    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        spans: list,
        *,
        used_len: int,
        scale: float,
        sparsity: float,
        pre_rearranged: bool,
    ) -> torch.Tensor:
        if self.state is None or self.failed:
            raise RuntimeError("MindIE parallel resource is closed or failed; recreate the worker")
        if self.active:
            raise RuntimeError("concurrent calls cannot share a MindIE parallel resource")
        from mindiesd.parallel import distributed_sparse_attention

        self.active = True
        try:
            output = distributed_sparse_attention(
                q,
                k,
                v,
                spans,
                state=self.state,
                sparse_type=self.sparse_type,
                used_len=used_len,
                precision=self.precision,
                scale=scale,
                sparsity=sparsity,
                pre_rearranged=pre_rearranged,
            )
            self.completed_calls += 1
            return output
        except Exception:
            # No alternate transport or replay after a partial collective.
            self.failed = True
            raise
        finally:
            self.active = False

    def close(self) -> None:
        if self.state is None:
            return
        if self.active:
            raise RuntimeError("cannot close active MindIE parallel resource")
        # Synchronization failure leaves the state and its storage intact.
        with torch.npu.device(self.device):
            platforms.current_omni_platform.synchronize()
        logger.debug("MindIE-SD pipeline closed.", extra={"completed_calls": self.completed_calls})
        self.state = None
        gc.collect()
