from vllm.distributed.parallel_state import init_distributed_environment,GroupCoordinator

from vllm.config import VllmConfig,set_current_vllm_config



vllm_config = VllmConfig()
vllm_config.parallel_config.tensor_parallel_size = 2

set_current_vllm_config(vllm_config)



def init_worker_distributed_environment(world_size: int, rank: int,local_rank: int):
    """
    Initialize the distributed environment for the worker process.
    This function sets up the necessary environment variables and
    initializes the distributed backend.
    """
    init_distributed_environment(world_size=world_size, rank=rank,local_rank=local_rank)
