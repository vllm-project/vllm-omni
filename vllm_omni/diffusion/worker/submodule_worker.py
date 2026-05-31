# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker for lightweight diffusion submodule stages."""

from __future__ import annotations

import os
from contextlib import AbstractContextManager, nullcontext
from types import SimpleNamespace

import torch
from vllm.config import CompilationConfig, DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.logger import init_logger
from vllm.profiler.wrapper import CudaProfilerWrapper, WorkerProfiler
from vllm.transformers_utils.config import get_config, get_hf_text_config
from vllm.utils.mem_utils import GiB_bytes
from vllm.v1.worker.workspace import init_workspace_manager

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.diffusion_submodule_runner import DiffusionSubmoduleRunner
from vllm_omni.platforms import current_omni_platform
from vllm_omni.profiler import OmniTorchProfilerWrapper, create_omni_profiler
from vllm_omni.worker.gpu_memory_utils import get_process_gpu_memory

logger = init_logger(__name__)


class SubModuleWorker:
    """Device/process owner for one encode/decode submodule runner."""

    def __init__(
        self,
        local_rank: int,
        rank: int,
        od_config: OmniDiffusionConfig,
    ) -> None:
        self.local_rank = local_rank
        self.rank = rank
        self.od_config = od_config
        self.device: torch.device | None = None
        self.vllm_config: VllmConfig | None = None
        self.model_runner: DiffusionSubmoduleRunner | None = None
        self.profiler: WorkerProfiler | None = None

        self.init_device()
        self.model_runner = DiffusionSubmoduleRunner(
            vllm_config=self.vllm_config,
            od_config=self.od_config,
            device=self.device,
        )
        self.profiler = self._create_profiler()
        logger.info("SubModuleWorker %d initialized.", self.rank)

    def init_device(self) -> None:
        """Initialize device/distributed state for submodule execution."""
        parallel_config = self.od_config.parallel_config
        world_size = self.od_config.num_gpus
        vae_patch_parallel_size = int(getattr(parallel_config, "vae_patch_parallel_size", 1) or 1)
        if world_size != 1 or vae_patch_parallel_size != 1:
            raise ValueError("Submodule stages support StagePool replicas, not intra-replica parallelism.")
        rank = self.rank

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(self.od_config.master_port)
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        self.device = current_omni_platform.get_torch_device(rank)
        current_omni_platform.set_device(self.device)

        vllm_config = VllmConfig(
            compilation_config=CompilationConfig(),
            device_config=DeviceConfig(device=self.device),
        )
        vllm_config.parallel_config.tensor_parallel_size = self.od_config.parallel_config.tensor_parallel_size
        vllm_config.parallel_config.data_parallel_size = self.od_config.parallel_config.data_parallel_size
        vllm_config.parallel_config.enable_expert_parallel = self.od_config.parallel_config.enable_expert_parallel
        vllm_config.profiler_config = getattr(self.od_config, "profiler_config", None)
        try:
            hf_config = get_config(
                self.od_config.model,
                trust_remote_code=self.od_config.trust_remote_code,
            )
        except ValueError:
            hf_config = None
            logger.info("Skipping hf_config loading for diffusion model %r", self.od_config.model_class_name)
        hf_text_config = get_hf_text_config(hf_config) if hf_config is not None else None
        # Minimal config for current submodules; refine if encoder/VAE stages need broader model_config support.
        vllm_config.model_config = SimpleNamespace(
            hf_config=hf_config,
            hf_text_config=hf_text_config,
            enforce_eager=self.od_config.enforce_eager,
            dtype=self.od_config.dtype,
            enable_return_routed_experts=False,
        )
        vllm_config.quant_config = self.od_config.quantization_config
        vllm_config.kernel_config.ir_op_priority = current_omni_platform.get_default_ir_op_priority(vllm_config)
        self.vllm_config = vllm_config

        with (
            set_forward_context(vllm_config=self.vllm_config, omni_diffusion_config=self.od_config),
            set_current_vllm_config(self.vllm_config),
        ):
            init_distributed_environment(world_size=world_size, rank=rank)
            initialize_model_parallel(
                data_parallel_size=parallel_config.data_parallel_size,
                cfg_parallel_size=parallel_config.cfg_parallel_size,
                sequence_parallel_size=parallel_config.sequence_parallel_size,
                ulysses_degree=parallel_config.ulysses_degree,
                ring_degree=parallel_config.ring_degree,
                tensor_parallel_size=parallel_config.tensor_parallel_size,
                pipeline_parallel_size=parallel_config.pipeline_parallel_size,
                fully_shard_degree=parallel_config.hsdp_shard_size if parallel_config.use_hsdp else 1,
                hsdp_replicate_size=parallel_config.hsdp_replicate_size if parallel_config.use_hsdp else 1,
                enable_expert_parallel=parallel_config.enable_expert_parallel,
            )
            init_workspace_manager(self.device)

    def load_model(self, load_format: str = "default", custom_pipeline_name: str | None = None) -> None:
        """Load the submodule pipeline through ``DiffusionSubmoduleRunner``."""
        assert self.model_runner is not None, "Model runner not initialized"
        with (
            set_forward_context(vllm_config=self.vllm_config, omni_diffusion_config=self.od_config),
            set_current_vllm_config(self.vllm_config),
        ):
            self.model_runner.load_model(
                memory_pool_context_fn=self._maybe_get_memory_pool_context,
                load_format=load_format,
                custom_pipeline_name=custom_pipeline_name,
            )

        process_memory = get_process_gpu_memory(self.local_rank)
        if process_memory is not None:
            logger.info(
                "SubModuleWorker %d: process-scoped GPU memory after model loading: %.2f GiB.",
                self.rank,
                process_memory / GiB_bytes,
            )

    def _create_profiler(self) -> WorkerProfiler | None:
        profiler_config = getattr(self.od_config, "profiler_config", None)
        profiler_type = getattr(profiler_config, "profiler", None)
        if profiler_type == "torch":
            return create_omni_profiler(
                profiler_config=profiler_config,
                worker_name=f"diffusion_submodule_rank{self.rank}",
                local_rank=self.local_rank,
            )
        if profiler_type == "cuda":
            return CudaProfilerWrapper(profiler_config)
        if profiler_type is not None:
            logger.warning("Unknown profiler backend %r on submodule worker %s", profiler_type, self.rank)
        return None

    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
        profiler = self.profiler
        if profiler is None:
            return

        if is_start:
            if isinstance(profiler, OmniTorchProfilerWrapper):
                filename = profile_prefix or f"diffusion_submodule_rank{self.rank}"
                profiler.set_trace_filename(filename)
            profiler.start()
        else:
            profiler.stop()

    def execute_submodule(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        """Execute one submodule request and return a submodule-shaped output."""
        assert self.model_runner is not None, "Model runner not initialized"
        profiler = self.profiler
        ctx = profiler.annotate_context_manager("diffusion_submodule_forward") if profiler else nullcontext()
        with ctx:
            output = self.model_runner.execute_model(req)
        if profiler:
            profiler.step()
        return output

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        if self.od_config.enable_sleep_mode:
            from vllm.device_allocator.cumem import CuMemAllocator

            allocator = CuMemAllocator.get_instance()
            if tag == "weights":
                assert allocator.get_current_usage() == 0, "Sleep mode can only be used for one instance per process."
            return allocator.use_memory_pool(tag=tag)
        return nullcontext()

    def shutdown(self) -> None:
        destroy_distributed_env()
