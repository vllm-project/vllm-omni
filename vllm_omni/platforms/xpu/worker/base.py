# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Base XPU worker class for vLLM-Omni sleep/wake support."""

from __future__ import annotations

import gc
from contextlib import AbstractContextManager, nullcontext

import torch
from vllm.logger import init_logger
from vllm.v1.worker.xpu_worker import XPUWorker

from vllm_omni.diffusion.data import OmniACK, OmniSleepTask, OmniWakeTask
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


class OmniXPUWorkerBase(XPUWorker):
    """Base XPU worker for vLLM-Omni with sleep/wake RPC support."""

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        v1_config_enabled = False
        if hasattr(self, "vllm_config"):
            model_cfg = getattr(self.vllm_config, "model_config", None)
            v1_config_enabled = getattr(model_cfg, "enable_sleep_mode", False)

        is_sleep_enabled = v1_config_enabled or getattr(self.cache_config, "enable_sleep_mode", False)
        if not is_sleep_enabled:
            logger.warning(f"[LLM Worker {self.rank}] Sleep Mode DISABLED.")
            return nullcontext()

        current_omni_platform.synchronize()
        gc.collect()
        from vllm.device_allocator import get_mem_allocator_instance

        allocator = get_mem_allocator_instance()
        logger.info(f"[LLM Worker {self.rank}] Sleep Mode ENABLED. Activating XpuMem pool for tag: {tag}")
        return allocator.use_memory_pool(tag=tag)

    def _get_used_memory(self) -> int:
        free_mem, total_mem = current_omni_platform.get_device_memory(self.device)
        return total_mem - free_mem

    def sleep(self, level: int = 1) -> bool:
        from vllm.device_allocator import get_mem_allocator_instance

        mem_before = self._get_used_memory()
        offload_tags = ("weights",) if level == 1 else tuple()
        allocator = get_mem_allocator_instance()
        allocator.sleep(offload_tags=offload_tags)
        current_omni_platform.empty_cache()
        current_omni_platform.synchronize()
        mem_after = self._get_used_memory()
        freed = max(0, mem_before - mem_after)
        remaining_gb = mem_after / 1024**3
        logger.info(
            f"[LLM Worker {self.rank}] Level {level} Sleep: Freed "
            f"{freed / 1024**3:.2f} GiB. {remaining_gb:.2f}GiB memory "
            "is still in use."
        )
        return True

    def wake_up(self, tags: list[str] | None = None) -> bool:
        from vllm.device_allocator import get_mem_allocator_instance

        allocator = get_mem_allocator_instance()
        allocator.wake_up(tags)
        current_omni_platform.synchronize()
        logger.info(f"[LLM Worker {self.rank}] Wake-up complete.")
        return True

    def handle_sleep_task(self, task: OmniSleepTask | dict) -> OmniACK | None:
        try:
            if isinstance(task, dict):
                task = OmniSleepTask(**task)
            logger.info(f"[LLM Worker {self.rank}] Handshake Received: Task {task.task_id}, Level {task.level}")
            if task.level == 2 and hasattr(self.model_runner, "graph_runners"):
                self.model_runner.graph_runners.clear()
                logger.info(f"[LLM Worker {self.rank}] LLM CUDA Graphs cleared.")

            free_before = current_omni_platform.get_free_memory(self.device)
            self.sleep(level=task.level)
            free_after = current_omni_platform.get_free_memory(self.device)
            rank_freed = max(0, free_after - free_before)
            if torch.distributed.is_initialized():
                t_freed = torch.tensor([float(rank_freed)], device=self.device)
                torch.distributed.all_reduce(t_freed)
                total_freed = int(t_freed.item())
                torch.distributed.barrier()
            else:
                total_freed = rank_freed

            if self.rank != 0:
                return None

            current_stage_id = getattr(self.vllm_config.model_config, "stage_id", 0)
            _, total_mem = current_omni_platform.get_device_memory(self.device)
            residual_gib = (total_mem - free_after) / 1024**3
            ack = OmniACK(
                task_id=task.task_id,
                status="SUCCESS",
                stage_id=current_stage_id,
                rank=self.rank,
                freed_bytes=total_freed,
                metadata={
                    "source": "omni_platform_audit",
                    "total_freed_gib": f"{total_freed / 1024**3:.2f}",
                    "rank_residual_gib": f"{residual_gib:.2f}",
                },
            )
            if hasattr(self, "result_mq") and self.result_mq:
                self.result_mq.put(ack)
            logger.info(f"[LLM Worker {self.rank}] ACK emitted for Task {task.task_id}")
            return ack
        except Exception as e:
            logger.error(f"[LLM Worker {self.rank}] Sleep Task Failed: {e}", exc_info=True)
            if torch.distributed.is_initialized():
                try:
                    torch.distributed.barrier()
                except Exception:
                    pass
            tid = task.task_id if hasattr(task, "task_id") else "unknown"
            return OmniACK(task_id=tid, status="ERROR", error_msg=str(e))

    def handle_wake_task(self, task: OmniWakeTask | dict) -> OmniACK | None:
        try:
            if isinstance(task, dict):
                task = OmniWakeTask(**task)
            self.wake_up(tags=task.tags)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            gc.collect()
            current_omni_platform.synchronize()
            free_now = current_omni_platform.get_free_memory(self.device)
            _, total_mem = current_omni_platform.get_device_memory(self.device)
            current_used_gib = (total_mem - free_now) / 1024**3

            if self.rank != 0:
                return None

            current_stage_id = getattr(self.vllm_config.model_config, "stage_id", 0)
            ack = OmniACK(
                task_id=task.task_id,
                status="SUCCESS",
                stage_id=current_stage_id,
                rank=self.rank,
                metadata={"state": "WARM", "current_vram_gib": f"{current_used_gib:.2f}"},
            )
            if hasattr(self, "result_mq") and self.result_mq:
                self.result_mq.put(ack)
            logger.info(f"[LLM Worker {self.rank}] Wake-up ACK emitted.")
            return ack
        except Exception as e:
            logger.error(f"[LLM Worker {self.rank}] Wake-up Failed: {e}", exc_info=True)
            if torch.distributed.is_initialized():
                try:
                    torch.distributed.barrier()
                except Exception:
                    pass
            tid = task.task_id if hasattr(task, "task_id") else "unknown"
            return OmniACK(task_id=tid, status="ERROR", error_msg=str(e))
