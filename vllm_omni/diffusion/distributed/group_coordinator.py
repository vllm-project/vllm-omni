# Copyright 2024 xDiT team.
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/distributed/parallel_state.py
# Copyright 2023 The vLLM team.
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
from collections import namedtuple
from typing import Any, Optional, Union

import torch
import torch.distributed
from torch.cuda import synchronize
from torch.distributed import Backend

from vllm_omni.diffusion import envs

if envs._is_npu():
    print("torch.npu synchronize")
    from torch.npu import synchronize

from vllm.distributed.parallel_state import GroupCoordinator
from vllm.logger import init_logger

logger = init_logger(__name__)

TensorMetadata = namedtuple("TensorMetadata", ["device", "dtype", "size"])

env_info = envs.PACKAGES_CHECKER.get_packages_info()


def _split_tensor_dict(
    tensor_dict: dict[str, Union[torch.Tensor, Any]], prefix: str = ""
) -> tuple[list[tuple[str, Any]], list[torch.Tensor]]:
    """Split the tensor dictionary into two parts:
    1. A list of (key, value) pairs. If the value is a tensor, it is replaced
         by its metadata.
    2. A list of tensors.

    If the Tensor is nested under `tensor_dict["key1"]["key2"]`, the key of its
    metadata will be "key1%key2".
    """
    metadata_list: list[tuple[str, Any]] = []
    tensor_list = []
    for key, value in tensor_dict.items():
        assert "%" not in key, "Avoid having '%' in key as it is used as a separator for nested entries."
        if isinstance(value, torch.Tensor):
            # Note: we cannot use `value.device` here,
            # because it contains not only the device type but also the device
            # index (e.g. "cuda:0"). We only need the device type.
            # receiving side will set the device index.
            device = value.device.type
            metadata_list.append((prefix + key, TensorMetadata(device, value.dtype, value.size())))
            tensor_list.append(value)
        elif isinstance(value, dict):
            if len(value) == 0:
                metadata_list.append((prefix + key, value))
            inner_metadata_list, inner_tensor_list = _split_tensor_dict(value, prefix + key + "%")
            metadata_list.extend(inner_metadata_list)
            tensor_list.extend(inner_tensor_list)
        else:
            metadata_list.append((prefix + key, value))
    return metadata_list, tensor_list


def _update_nested_dict(nested_dict, flattened_key, value):
    key_splits = flattened_key.split("%")
    cur_dict = nested_dict
    for k in key_splits[:-1]:
        if k not in cur_dict:
            cur_dict[k] = {}
        cur_dict = cur_dict[k]
    cur_dict[key_splits[-1]] = value


class PipelineGroupCoordinator(GroupCoordinator):
    """
    available attributes:
    rank: int  # global rank
    ranks: list[int]  # global ranks in the group
    world_size: int  # size of the group
    difference between `local_rank` and `rank_in_group`:
    if we have a group of size 4 across two nodes:
    Process | Node | Rank | Local Rank | Rank in Group
      0     |   0  |  0   |     0      |       0
      1     |   0  |  1   |     1      |       1
      2     |   1  |  2   |     0      |       2
      3     |   1  |  3   |     1      |       3
    local_rank: int  # local rank used to assign devices
    rank_in_group: int  # rank inside the group
    cpu_group: ProcessGroup  # group for CPU communication
    device_group: ProcessGroup  # group for device communication
    """

    def __init__(
        self,
        group_ranks: list[list[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, Backend],
    ):
        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank
        self.device_group = None
        self.cpu_group = None
        self.cpu_groups = []
        self.device_groups = []
        if len(group_ranks[0]) > 2 or len(group_ranks[0]) == 1:
            for ranks in group_ranks:
                device_group = torch.distributed.new_group(ranks, backend=torch_distributed_backend)
                # a group with `gloo` backend, to allow direct coordination between
                # processes through the CPU.
                cpu_group = torch.distributed.new_group(ranks, backend="gloo")
                if self.rank in ranks:
                    self.ranks = ranks
                    self.world_size = len(ranks)
                    self.rank_in_group = ranks.index(self.rank)
                    self.device_group = device_group
                    self.cpu_group = cpu_group
        # when pipeline parallelism is 2, we need to create two groups to avoid
        #   communication stall.
        # *_group_0_1 represents the group for communication from device 0 to
        #   device 1.
        # *_group_1_0 represents the group for communication from device 1 to
        #   device 0.
        elif len(group_ranks[0]) == 2:
            for ranks in group_ranks:
                device_group_0_1 = torch.distributed.new_group(ranks, backend=torch_distributed_backend)
                device_group_1_0 = torch.distributed.new_group(ranks, backend=torch_distributed_backend)
                # a group with `gloo` backend, to allow direct coordination between
                # processes through the CPU.
                cpu_group_0_1 = torch.distributed.new_group(ranks, backend="gloo")
                cpu_group_1_0 = torch.distributed.new_group(ranks, backend="gloo")
                if self.rank in ranks:
                    self.ranks = ranks
                    self.world_size = len(ranks)
                    self.rank_in_group = ranks.index(self.rank)
                    self.device_groups = [device_group_0_1, device_group_1_0]
                    self.cpu_groups = [cpu_group_0_1, cpu_group_1_0]
                    self.device_group = device_group_0_1
                    self.cpu_group = cpu_group_0_1

        assert self.cpu_group is not None
        assert self.device_group is not None

        self.device = envs.get_device(local_rank)

        self.recv_buffer_set: bool = False
        self.recv_tasks_queue: list[tuple[str, int]] = []
        self.receiving_tasks: list[tuple[torch.distributed.Work, str, int]] = []
        self.dtype: Optional[torch.dtype] = None
        self.num_pipefusion_patches: Optional[int] = None

        self.recv_shape: dict[str, dict[int, torch.Size]] = {}
        self.send_shape: dict[str, dict[int, torch.Size]] = {}
        self.recv_buffer: dict[str, dict[int, torch.Size]] = {}

        self.skip_tensor_recv_buffer_set: bool = False
        self.recv_skip_tasks_queue: list[Union[int, tuple[str, int]]] = []
        self.receiving_skip_tasks: list[tuple[torch.distributed.Work, str, int]] = []
        self.skip_tensor_recv_buffer: Optional[Union[list[torch.Tensor], torch.Tensor]] = None
        self.skip_device_group = None
        for ranks in group_ranks:
            skip_device_group = torch.distributed.new_group(ranks, backend=torch_distributed_backend)
            if self.rank in ranks:
                self.skip_device_group = skip_device_group
        assert self.skip_device_group is not None

    def reset_buffer(self):
        self.recv_tasks_queue = []
        self.receiving_tasks = []
        self.recv_shape = {}
        self.send_shape = {}
        self.recv_buffer = {}

        self.recv_skip_tasks_queue = []
        self.receiving_skip_tasks = []
        self.skip_tensor_recv_buffer = {}

    def set_config(self, dtype: torch.dtype):
        self.dtype = dtype

    def set_recv_buffer(
        self,
        num_pipefusion_patches: int,
        patches_shape_list: list[list[int]],
        feature_map_shape: list[int],
        dtype: torch.dtype,
    ):
        assert isinstance(dtype, torch.dtype), "dtype must be a torch.dtype object"
        assert isinstance(num_pipefusion_patches, int) and num_pipefusion_patches >= 1, (
            "num_pipefusion_patches must be greater than or equal to 1"
        )
        self.dtype = dtype
        self.num_pipefusion_patches = num_pipefusion_patches
        self.recv_buffer = [torch.zeros(*shape, dtype=self.dtype, device=self.device) for shape in patches_shape_list]
        self.recv_buffer.append(torch.zeros(*feature_map_shape, dtype=self.dtype, device=self.device))
        self.recv_buffer_set = True

    def set_extra_tensors_recv_buffer(
        self,
        name: str,
        shape: list[int],
        num_buffers: int = 1,
        dtype: torch.dtype = torch.float16,
    ):
        self.extra_tensors_recv_buffer[name] = [
            torch.zeros(*shape, dtype=dtype, device=self.device) for _ in range(num_buffers)
        ]

    def _check_shape_and_buffer(
        self,
        tensor_send_to_next=None,
        recv_prev=False,
        name: Optional[str] = None,
        segment_idx: int = 0,
    ):
        send_flag = False
        name = name or "latent"
        if tensor_send_to_next is not None:
            shape_list = self.send_shape.get(name, None)
            if shape_list is None:
                self.send_shape[name] = {segment_idx: tensor_send_to_next.shape}
                send_flag = True
            elif shape_list.get(segment_idx, None) is None:
                self.send_shape[name][segment_idx] = tensor_send_to_next.shape
                send_flag = True

        recv_flag = False
        if recv_prev:
            shape_list = self.recv_shape.get(name, None)
            if shape_list is None:
                recv_flag = True
            elif shape_list.get(segment_idx, None) is None:
                recv_flag = True

        recv_prev_shape = self._communicate_shapes(
            tensor_send_to_next=tensor_send_to_next if send_flag else None,
            recv_prev=recv_flag,
        )

        if recv_flag:
            if self.recv_shape.get(name, None) is None:
                self.recv_shape[name] = {segment_idx: recv_prev_shape}
            else:
                self.recv_shape[name][segment_idx] = recv_prev_shape

            if self.recv_buffer.get(name, None) is None:
                self.recv_buffer[name] = {
                    segment_idx: torch.zeros(recv_prev_shape, device=self.device, dtype=self.dtype)
                }
            else:
                if self.recv_buffer[name].get(segment_idx, None) is not None:
                    logger.warning(f"Recv buffer [name: {name}, segment_idx: {segment_idx}] already exist. updating...")
                self.recv_buffer[name][segment_idx] = torch.zeros(recv_prev_shape, device=self.device, dtype=self.dtype)

    def _communicate_shapes(self, tensor_send_to_next=None, recv_prev=False):
        """Communicate tensor shapes between stages. Used to communicate
        tensor shapes before the actual tensor communication happens.

        Args:
            tensor_send_next: tensor to send to next rank (no tensor sent if
                              set to None).
            recv_prev: boolean for whether tensor should be received from
                       previous rank.
        """

        ops = []
        if recv_prev:
            recv_prev_dim_tensor = torch.empty((1), device=self.device, dtype=torch.int64)
            recv_prev_dim_op = torch.distributed.P2POp(
                torch.distributed.irecv,
                recv_prev_dim_tensor,
                self.prev_rank,
                self.device_group,
            )
            ops.append(recv_prev_dim_op)

        if tensor_send_to_next is not None:
            send_next_dim_tensor = torch.tensor(tensor_send_to_next.dim(), device=self.device, dtype=torch.int64)
            send_next_dim_op = torch.distributed.P2POp(
                torch.distributed.isend,
                send_next_dim_tensor,
                self.next_rank,
                self.device_group,
            )
            ops.append(send_next_dim_op)

        if len(ops) > 0:
            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        # To protect against race condition when using batch_isend_irecv().
        # should take this out once the bug with batch_isend_irecv is resolved.
        synchronize()

        ops = []
        recv_prev_shape_tensor = None
        if recv_prev:
            recv_prev_shape_tensor = torch.empty(
                torch.Size(recv_prev_dim_tensor), device=self.device, dtype=torch.int64
            )
            recv_prev_shape_op = torch.distributed.P2POp(
                torch.distributed.irecv,
                recv_prev_shape_tensor,
                self.prev_rank,
                self.device_group,
            )
            ops.append(recv_prev_shape_op)

        if tensor_send_to_next is not None:
            send_next_shape_tensor = torch.tensor(tensor_send_to_next.size(), device=self.device, dtype=torch.int64)
            send_next_shape_op = torch.distributed.P2POp(
                torch.distributed.isend,
                send_next_shape_tensor,
                self.next_rank,
                self.device_group,
            )
            ops.append(send_next_shape_op)

        if len(ops) > 0:
            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        synchronize()

        recv_prev_shape = [0, 0, 0]
        if recv_prev_shape_tensor is not None:
            recv_prev_shape = recv_prev_shape_tensor
        return torch.Size(recv_prev_shape)

    def pipeline_send(self, tensor: torch.Tensor, name: str = "latent", segment_idx: int = -1) -> None:
        tensor = tensor.contiguous()
        self._check_shape_and_buffer(tensor_send_to_next=tensor, name=name, segment_idx=segment_idx)
        self._pipeline_isend(tensor).wait()

    def pipeline_isend(self, tensor: torch.Tensor, name: str = "latent", segment_idx: int = -1) -> None:
        tensor = tensor.contiguous()
        self._check_shape_and_buffer(tensor_send_to_next=tensor, name=name, segment_idx=segment_idx)
        self._pipeline_isend(tensor)

    def pipeline_recv(self, idx: int = -1, name: str = "latent") -> torch.Tensor:
        name = name or "latent"
        self._check_shape_and_buffer(recv_prev=True, name=name, segment_idx=idx)
        self._pipeline_irecv(self.recv_buffer[name][idx]).wait()
        return self.recv_buffer[name][idx]

    def add_pipeline_recv_task(self, idx: int = -1, name: str = "latent"):
        name = name or "latent"
        self.recv_tasks_queue.append((name, idx))

    def recv_next(self):
        if len(self.recv_tasks_queue) == 0:
            raise ValueError("No more tasks to receive")
        elif len(self.recv_tasks_queue) > 0:
            name, idx = self.recv_tasks_queue.pop(0)
            self._check_shape_and_buffer(recv_prev=True, name=name, segment_idx=idx)
            self.receiving_tasks.append((self._pipeline_irecv(self.recv_buffer[name][idx]), name, idx))

    def get_pipeline_recv_data(self, idx: int = -1, name: str = "latent") -> torch.Tensor:
        assert len(self.receiving_tasks) > 0, "No tasks to receive, call add_pipeline_recv_task first"
        receiving_task = self.receiving_tasks.pop(0)
        receiving_task[0].wait()
        assert receiving_task[1] == name and receiving_task[2] == idx, "Received tensor does not match the requested"
        return self.recv_buffer[name][idx]

    def _pipeline_irecv(self, tensor: torch.tensor):
        return torch.distributed.irecv(
            tensor,
            src=self.prev_rank,
            group=(self.device_groups[(self.rank_in_group + 1) % 2] if self.world_size == 2 else self.device_group),
        )

    def _pipeline_isend(self, tensor: torch.tensor):
        return torch.distributed.isend(
            tensor,
            dst=self.next_rank,
            group=(self.device_groups[self.rank_in_group % 2] if self.world_size == 2 else self.device_group),
        )

    def set_skip_tensor_recv_buffer(
        self,
        patches_shape_list: list[list[int]],
        feature_map_shape: list[int],
    ):
        self.skip_tensor_recv_buffer = [
            torch.zeros(*shape, dtype=self.dtype, device=self.device) for shape in patches_shape_list
        ]
        self.skip_tensor_recv_buffer.append(torch.zeros(*feature_map_shape, dtype=self.dtype, device=self.device))
        self.skip_tensor_recv_buffer_set = True

    def pipeline_send_skip(self, tensor: torch.Tensor) -> None:
        tensor = tensor.contiguous()
        self._pipeline_isend_skip(tensor).wait()

    def pipeline_isend_skip(self, tensor: torch.Tensor) -> None:
        tensor = tensor.contiguous()
        self._pipeline_isend_skip(tensor)

    def pipeline_recv_skip(self, idx: int = -1) -> torch.Tensor:
        self._pipeline_irecv_skip(self.skip_tensor_recv_buffer[idx]).wait()
        return self.skip_tensor_recv_buffer[idx]

    def add_pipeline_recv_skip_task(self, idx: int = -1):
        self.recv_skip_tasks_queue.append(idx)

    def get_pipeline_recv_skip_data(self, idx: int = -1) -> torch.Tensor:
        assert len(self.receiving_skip_tasks) > 0, "No tasks to receive, call add_pipeline_recv_skip_task first"
        receiving_skip_task = self.receiving_skip_tasks.pop(0)
        receiving_skip_task[0].wait()
        assert receiving_skip_task[2] == idx, "Received tensor does not match the requested"
        return self.skip_tensor_recv_buffer[idx]

    def recv_skip_next(self):
        if len(self.recv_skip_tasks_queue) == 0:
            raise ValueError("No more tasks to receive")
        elif len(self.recv_skip_tasks_queue) > 0:
            task = self.recv_skip_tasks_queue.pop(0)
            idx = task
            self.receiving_skip_tasks.append(
                (
                    self._pipeline_irecv_skip(self.skip_tensor_recv_buffer[idx]),
                    None,
                    idx,
                )
            )

    def _pipeline_irecv_skip(self, tensor: torch.tensor):
        return torch.distributed.irecv(tensor, src=self.skip_rank, group=self.skip_device_group)

    def _pipeline_isend_skip(self, tensor: torch.tensor):
        return torch.distributed.isend(tensor, dst=self.skip_rank, group=self.skip_device_group)


class SequenceParallelGroupCoordinator(GroupCoordinator):
    def __init__(
        self,
        group_ranks: list[list[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, Backend],
        **kwargs,
    ):
        super().__init__(
            group_ranks=group_ranks,
            local_rank=local_rank,
            torch_distributed_backend=torch_distributed_backend,
        )

        ulysses_group = kwargs.get("ulysses_group", None)
        ring_group = kwargs.get("ring_group", None)
        if ulysses_group is None:
            raise RuntimeError(
                "Please pass argument 'ulysses_group' when calling init func of SequenceParallelGroupCoordinator"
            )
        if ring_group is None:
            raise RuntimeError(
                "Please pass argument 'ring_group' when calling init func of SequenceParallelGroupCoordinator"
            )
        self.ulysses_group = ulysses_group
        self.ring_group = ring_group

        self.ulysses_world_size = torch.distributed.get_world_size(self.ulysses_group)
        self.ulysses_rank = torch.distributed.get_rank(self.ulysses_group)
        self.ring_world_size = torch.distributed.get_world_size(self.ring_group)
        self.ring_rank = torch.distributed.get_rank(self.ring_group)
