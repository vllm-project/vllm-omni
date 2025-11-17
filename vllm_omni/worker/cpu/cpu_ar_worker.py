import os
import platform

from vllm import envs
import torch
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.platforms import current_platform, CpuArchEnum


from vllm.v1.worker.cpu_worker import CPUWorker
from vllm.v1.worker.gpu_worker import init_worker_distributed_environment

from vllm_omni.worker.cpu.cpu_ar_model_runner import CPUARModelRunner

logger = init_logger(__name__)


class CPUARWorker(CPUWorker):
    def init_device(self):
        # Setup OpenMP threads affinity.
        omp_cpuids = envs.VLLM_CPU_OMP_THREADS_BIND
        if omp_cpuids == "auto" and platform.system() == "Linux":
            cpu_arch = current_platform.get_cpu_architecture()
            if cpu_arch in (CpuArchEnum.POWERPC, CpuArchEnum.S390X):
                # For S390X/POWERPC SMT-8/4/2
                self.local_omp_cpuid = self._get_autobind_cpu_ids(
                    lambda cpus: [cpu for cpu in cpus if cpu.id % 8 < 4])
            elif current_platform.get_cpu_architecture() == CpuArchEnum.X86:
                # For x86 SMT-2, use 1 CPU per core
                self.local_omp_cpuid = self._get_autobind_cpu_ids(
                    lambda cpus: cpus[-1:])
            else:
                self.local_omp_cpuid = "all"
        else:
            local_dp_rank = self.parallel_config.data_parallel_rank_local
            omp_cpuids = omp_cpuids.split("|")
            if local_dp_rank is not None:
                world_size = self.parallel_config.world_size
                omp_cpuids = omp_cpuids[local_dp_rank *
                                        world_size:(local_dp_rank + 1) *
                                                   world_size]
            self.local_omp_cpuid = omp_cpuids[self.rank]

        if self.local_omp_cpuid != "all":
            ret = torch.ops._C_utils.init_cpu_threads_env(self.local_omp_cpuid)
            if ret:
                logger.info(ret)

        # Note: unique identifier for creating allreduce shared memory
        os.environ["VLLM_DIST_IDENT"] = self.distributed_init_method.split(
            ":")[-1]
        # Initialize the distributed environment.
        init_worker_distributed_environment(self.vllm_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank,
                                            current_platform.dist_backend)
        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Construct the model runner
        self.model_runner: CPUARModelRunner = CPUARModelRunner(
            self.vllm_config, torch.device("cpu"))
