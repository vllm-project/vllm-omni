import os

import torch.distributed as dist


class COMM_INFO:

    def __init__(self):
        self.group = None
        self.sp_size = 1
        self.global_rank = 0
        self.rank_within_group = 0
        self.group_id = 0


nccl_info = COMM_INFO()
_SEQUENCE_PARALLEL_STATE = False


def initialize_sequence_parallel_state(sequence_parallel_size):
    global _SEQUENCE_PARALLEL_STATE
    if sequence_parallel_size > 1:
        _SEQUENCE_PARALLEL_STATE = True
        initialize_sequence_parallel_group(sequence_parallel_size)
    else:
        nccl_info.sp_size = 1
        nccl_info.global_rank = int(os.getenv("RANK", "0"))
        nccl_info.rank_within_group = 0
        nccl_info.group_id = int(os.getenv("RANK", "0"))


def set_sequence_parallel_state(state):
    global _SEQUENCE_PARALLEL_STATE
    _SEQUENCE_PARALLEL_STATE = state


def get_sequence_parallel_state():
    return _SEQUENCE_PARALLEL_STATE


def initialize_sequence_parallel_group(sequence_parallel_size):
    """Initialize the sequence parallel group."""
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    assert (
        world_size % sequence_parallel_size == 0
    ), "world_size must be divisible by sequence_parallel_size, but got world_size: {}, sequence_parallel_size: {}".format(
        world_size, sequence_parallel_size)
    nccl_info.sp_size = sequence_parallel_size
    nccl_info.global_rank = rank
    num_sequence_parallel_groups: int = world_size // sequence_parallel_size
    for i in range(num_sequence_parallel_groups):
        ranks = range(i * sequence_parallel_size,
                      (i + 1) * sequence_parallel_size)
        group = dist.new_group(ranks)
        if rank in ranks:
            nccl_info.group = group
            nccl_info.rank_within_group = rank - i * sequence_parallel_size
            nccl_info.group_id = i



def initialize_sequence_parallel_group_custom(process_group):
    set_sequence_parallel_state(True)
    """Initialize an unsafe sequence parallel group with a pre-formed group."""
    rank = dist.get_rank(group=process_group)
    sequence_parallel_size = dist.get_world_size(group=process_group)
    nccl_info.sp_size = sequence_parallel_size
    nccl_info.global_rank = dist.get_rank()  # global rank
    nccl_info.group = process_group
    nccl_info.rank_within_group = rank
    nccl_info.group_id = 0


def destroy_sequence_parallel_group():
    """Destroy the sequence parallel group."""
    dist.destroy_process_group()

