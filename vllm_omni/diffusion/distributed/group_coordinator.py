# Copyright 2024 xDiT team.
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/distributed/parallel_state.py
# Copyright 2023 The vLLM team.
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import pickle
from collections import namedtuple
from contextlib import nullcontext
from typing import Any

import torch
import torch.distributed
from torch.distributed import Backend, ProcessGroup
from vllm.logger import init_logger

from vllm_omni.diffusion import envs
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


TensorMetadata = namedtuple("TensorMetadata", ["device", "dtype", "size"])

env_info = envs.PACKAGES_CHECKER.get_packages_info()


def _split_tensor_dict(
    tensor_dict: dict[str, torch.Tensor | Any], prefix: str = ""
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


class GroupCoordinator:
    """
    PyTorch ProcessGroup wrapper for a group of processes.
    PyTorch ProcessGroup is bound to one specific communication backend,
        e.g. NCCL, Gloo, MPI, etc.
    GroupCoordinator takes charge of all the communication operations among
        the processes in the group. It can route the communication to
        a specific implementation (e.g. switch allreduce implementation
        based on the tensor size and cuda graph mode).
    """

    # available attributes:
    rank: int  # global rank
    ranks: list[int]  # global ranks in the group
    world_size: int  # size of the group
    # difference between `local_rank` and `rank_in_group`:
    # if we have a group of size 4 across two nodes:
    # Process | Node | Rank | Local Rank | Rank in Group
    #   0     |   0  |  0   |     0      |       0
    #   1     |   0  |  1   |     1      |       1
    #   2     |   1  |  2   |     0      |       2
    #   3     |   1  |  3   |     1      |       3
    local_rank: int  # local rank used to assign devices
    rank_in_group: int  # rank inside the group
    cpu_group: ProcessGroup  # group for CPU communication
    device_group: ProcessGroup  # group for device communication

    def __init__(
        self,
        group_ranks: list[list[int]],
        local_rank: int,
        torch_distributed_backend: str | Backend,
    ):
        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank
        self.device_group = None
        self.cpu_group = None
        self.shm_broadcaster = None

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

        assert self.cpu_group is not None
        assert self.device_group is not None

        self.device = current_omni_platform.get_torch_device(local_rank)

    @property
    def first_rank(self):
        """Return the global rank of the first process in the group"""
        return self.ranks[0]

    @property
    def last_rank(self):
        """Return the global rank of the last process in the group"""
        return self.ranks[-1]

    @property
    def is_first_rank(self):
        """Return whether the caller is the first process in the group"""
        return self.rank == self.first_rank

    @property
    def is_last_rank(self):
        """Return whether the caller is the last process in the group"""
        return self.rank == self.last_rank

    @property
    def next_rank(self):
        """Return the global rank of the process that follows the caller"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return self.ranks[(rank_in_group + 1) % world_size]

    @property
    def prev_rank(self):
        """Return the global rank of the process that precedes the caller"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return self.ranks[(rank_in_group - 1) % world_size]

    @property
    def group_next_rank(self):
        """Return the group rank of the process that follows the caller"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return (rank_in_group + 1) % world_size

    @property
    def group_prev_rank(self):
        """Return the group rank of the process that precedes the caller"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return (rank_in_group - 1) % world_size

    @property
    def skip_rank(self):
        """Return the global rank of the process that skip connects with the caller"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return self.ranks[(world_size - rank_in_group - 1) % world_size]

    @property
    def group_skip_rank(self):
        """Return the group rank of the process that skip connects with the caller"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return (world_size - rank_in_group - 1) % world_size

    def all_reduce(self, input_: torch.Tensor, op=torch._C._distributed_c10d.ReduceOp.SUM) -> torch.Tensor:
        """
        NOTE: This operation will be applied in-place or out-of-place.
        Always assume this function modifies its input, but use the return
        value as the output.
        """
        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return input_
        else:
            torch.distributed.all_reduce(input_, op=op, group=self.device_group)
        return input_

    def all_gather(
        self, input_: torch.Tensor, dim: int = 0, separate_tensors: bool = False
    ) -> torch.Tensor | list[torch.Tensor]:
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        # Allocate output tensor.
        input_size = list(input_.size())
        input_size[0] *= world_size
        output_tensor = torch.empty(input_size, dtype=input_.dtype, device=input_.device)
        # All-gather.
        torch.distributed.all_gather_into_tensor(output_tensor, input_.contiguous(), group=self.device_group)
        if dim != 0:
            input_size[0] //= world_size
            output_tensor = output_tensor.reshape(
                [
                    world_size,
                ]
                + input_size
            )
            output_tensor = output_tensor.movedim(0, dim)

        if separate_tensors:
            tensor_list = [
                output_tensor.view(-1).narrow(0, input_.numel() * i, input_.numel()).view_as(input_)
                for i in range(world_size)
            ]
            return tensor_list
        else:
            input_size = list(input_.size())
            input_size[dim] = input_size[dim] * world_size
            # Reshape
            output_tensor = output_tensor.reshape(input_size)
            return output_tensor

    def gather(self, input_: torch.Tensor, dst: int = 0, dim: int = -1) -> torch.Tensor:
        """
        NOTE: We assume that the input tensor is on the same device across
        all the ranks.
        NOTE: `dst` is the local rank of the destination rank.
        """
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        # Allocate output tensor.
        if self.rank_in_group == dst:
            gather_list = [torch.empty_like(input_) for _ in range(world_size)]
        else:
            gather_list = None
        # Gather.
        torch.distributed.gather(input_, gather_list, dst=self.ranks[dst], group=self.device_group)
        if self.rank_in_group == dst:
            output_tensor = torch.cat(gather_list, dim=dim)
        else:
            output_tensor = None
        return output_tensor

    def broadcast(self, input_: torch.Tensor, src: int = 0):
        """Broadcast the input tensor.
        NOTE: `src` is the local rank of the source rank.
        """
        assert src < self.world_size, f"Invalid src rank ({src})"

        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return input_
        # Broadcast.
        torch.distributed.broadcast(input_, src=self.ranks[src], group=self.device_group)
        return input_

    def broadcast_object(self, obj: Any | None = None, src: int = 0):
        """Broadcast the input object.
        NOTE: `src` is the local rank of the source rank.
        """
        assert src < self.world_size, f"Invalid src rank ({src})"

        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return obj
        if self.shm_broadcaster is not None:
            assert src == 0, "Shared memory broadcaster only supports src=0"
            return self.shm_broadcaster.broadcast_object(obj)
        if self.rank_in_group == src:
            torch.distributed.broadcast_object_list([obj], src=self.ranks[src], group=self.cpu_group)
            return obj
        else:
            recv = [None]
            torch.distributed.broadcast_object_list(recv, src=self.ranks[src], group=self.cpu_group)
            return recv[0]

    def broadcast_object_list(self, obj_list: list[Any], src: int = 0, group: ProcessGroup | None = None):
        """Broadcast the input object list.
        NOTE: `src` is the local rank of the source rank.
        """
        assert src < self.world_size, f"Invalid src rank ({src})"

        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return obj_list
        # Broadcast.
        torch.distributed.broadcast_object_list(obj_list, src=self.ranks[src], group=self.device_group)
        return obj_list

    def send_object(self, obj: Any, dst: int) -> None:
        """Send the input object list to the destination rank."""
        """NOTE: `dst` is the local rank of the destination rank."""

        assert dst < self.world_size, f"Invalid dst rank ({dst})"

        assert dst != self.rank_in_group, "Invalid destination rank. Destination rank is the same as the current rank."

        # Serialize object to tensor and get the size as well
        object_tensor = torch.frombuffer(pickle.dumps(obj), dtype=torch.uint8)

        size_tensor = torch.tensor([object_tensor.numel()], dtype=torch.long, device="cpu")

        # Send object size

        torch.distributed.send(size_tensor, dst=self.ranks[dst], group=self.cpu_group)

        # Send object
        torch.distributed.send(object_tensor, dst=self.ranks[dst], group=self.cpu_group)

        return None

    def recv_object(self, src: int) -> Any:
        """Receive the input object list from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""

        assert src < self.world_size, f"Invalid src rank ({src})"

        assert src != self.rank_in_group, "Invalid source rank. Source rank is the same as the current rank."

        size_tensor = torch.empty(1, dtype=torch.long, device="cpu")

        # Receive object size
        rank_size = torch.distributed.recv(size_tensor, src=self.ranks[src], group=self.cpu_group)

        # Tensor to receive serialized objects into.
        object_tensor = torch.empty(  # type: ignore[call-overload]
            size_tensor.item(),  # type: ignore[arg-type]
            dtype=torch.uint8,
            device="cpu",
        )

        rank_object = torch.distributed.recv(object_tensor, src=self.ranks[src], group=self.cpu_group)

        assert rank_object == rank_size, "Received object sender rank does not match the size sender rank."

        obj = pickle.loads(object_tensor.numpy().tobytes())

        return obj

    def broadcast_tensor_dict(
        self,
        tensor_dict: dict[str, torch.Tensor | Any] | None = None,
        src: int = 0,
        group: ProcessGroup | None = None,
        metadata_group: ProcessGroup | None = None,
    ) -> dict[str, torch.Tensor | Any] | None:
        """Broadcast the input tensor dictionary.
        NOTE: `src` is the local rank of the source rank.
        """
        # Bypass the function if we are using only 1 GPU.
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return tensor_dict

        group = self.device_group
        metadata_group = self.cpu_group
        assert src < self.world_size, f"Invalid src rank ({src})"
        src = self.ranks[src]

        rank = self.rank
        if rank == src:
            metadata_list: list[tuple[Any, Any]] = []
            assert isinstance(tensor_dict, dict), f"Expecting a dictionary, got {type(tensor_dict)}"
            metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
            # `metadata_list` lives in CPU memory.
            # `broadcast_object_list` has serialization & deserialization,
            # all happening on CPU. Therefore, we can use the CPU group.
            self.broadcast_object(metadata_list, src=src)
            async_handles = []
            for tensor in tensor_list:
                if tensor.numel() == 0:
                    # Skip broadcasting empty tensors.
                    continue
                if tensor.is_cpu:
                    # use metadata_group for CPU tensors
                    handle = torch.distributed.broadcast(tensor, src=src, group=metadata_group, async_op=True)
                else:
                    # use group for GPU tensors
                    handle = torch.distributed.broadcast(tensor, src=src, group=group, async_op=True)
                async_handles.append(handle)
            for async_handle in async_handles:
                async_handle.wait()

        else:
            metadata_list = self.broadcast_object(None, src=src)
            tensor_dict = {}
            async_handles = []
            for key, value in metadata_list:
                if isinstance(value, TensorMetadata):
                    tensor = torch.empty(value.size, dtype=value.dtype, device=value.device)
                    if tensor.numel() == 0:
                        # Skip broadcasting empty tensors.
                        _update_nested_dict(tensor_dict, key, tensor)
                        continue
                    if tensor.is_cpu:
                        # use metadata_group for CPU tensors
                        handle = torch.distributed.broadcast(tensor, src=src, group=metadata_group, async_op=True)
                    else:
                        # use group for GPU tensors
                        handle = torch.distributed.broadcast(tensor, src=src, group=group, async_op=True)
                    async_handles.append(handle)
                    _update_nested_dict(tensor_dict, key, tensor)
                else:
                    _update_nested_dict(tensor_dict, key, value)
            for async_handle in async_handles:
                async_handle.wait()
        return tensor_dict

    def isend_tensor_dict(
        self,
        tensor_dict: dict[str, torch.Tensor | Any],
        dst: int | None = None,
    ) -> list[torch.distributed.Work]:
        """Non-blocking send of a tensor dictionary.

        Sends metadata via the Gloo CPU group (blocking) then starts a
        non-blocking NCCL isend for each GPU tensor.  Returns the list of
        Work handles; the caller must call handle.wait() before the tensors
        can be safely reused or freed.

        NOTE: `dst` is the group rank of the destination.
        """
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return []

        if dst is None:
            dst = self.group_next_rank
        assert dst < self.world_size, f"Invalid dst rank ({dst})"

        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        self.send_object(metadata_list, dst=dst)

        handles: list[torch.distributed.Work] = []
        for tensor in tensor_list:
            if tensor.numel() == 0:
                continue
            group = self.cpu_group if tensor.is_cpu else self.device_group
            handle = torch.distributed.isend(tensor, dst=self.ranks[dst], group=group)
            if tensor.is_cuda:
                # Keep allocator from reusing this CUDA buffer before the async send finishes.
                tensor.record_stream(torch.cuda.current_stream(tensor.device))
            handles.append(handle)
        return handles

    def irecv_tensor_dict(
        self,
        src: int | None = None,
    ) -> tuple[dict[str, torch.Tensor | Any], list[torch.distributed.Work], list]:
        """Non-blocking receive of a tensor dictionary.

        Receives metadata via the Gloo CPU group (blocking) then starts a
        non-blocking NCCL irecv for each GPU tensor.  Returns
        ``(tensor_dict, comm_handles, comm_postprocess)`` matching the
        interface expected by ``AsyncIntermediateTensors``.

        NOTE: `src` is the group rank of the source.
        """
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return {}, [], []

        if src is None:
            src = self.group_prev_rank
        assert src < self.world_size, f"Invalid src rank ({src})"

        recv_metadata_list = self.recv_object(src=src)
        tensor_dict: dict[str, Any] = {}
        handles: list[torch.distributed.Work] = []

        for key, value in recv_metadata_list:
            if isinstance(value, TensorMetadata):
                tensor = torch.empty(value.size, dtype=value.dtype, device=value.device)
                if tensor.numel() > 0:
                    group = self.cpu_group if tensor.is_cpu else self.device_group
                    handles.append(torch.distributed.irecv(tensor, src=self.ranks[src], group=group))
                _update_nested_dict(tensor_dict, key, tensor)
            else:
                _update_nested_dict(tensor_dict, key, value)

        return tensor_dict, handles, []

    def send_tensor_dict(
        self,
        tensor_dict: dict[str, torch.Tensor | Any],
        dst: int | None = None,
    ) -> dict[str, torch.Tensor | Any] | None:
        """Send the input tensor dictionary.
        NOTE: `dst` is the local rank of the source rank.
        """
        # Bypass the function if we are using only 1 GPU.
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return tensor_dict

        group = self.device_group
        metadata_group = self.cpu_group

        if dst is None:
            dst = self.group_next_rank
        assert dst < self.world_size, f"Invalid dst rank ({dst})"

        metadata_list: list[tuple[Any, Any]] = []
        assert isinstance(tensor_dict, dict), f"Expecting a dictionary, got {type(tensor_dict)}"
        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        # `metadata_list` lives in CPU memory.
        # `send_object_list` has serialization & deserialization,
        # all happening on CPU. Therefore, we can use the CPU group.
        self.send_object(metadata_list, dst=dst)
        for tensor in tensor_list:
            if tensor.numel() == 0:
                # Skip sending empty tensors.
                continue
            if tensor.is_cpu:
                # use metadata_group for CPU tensors
                torch.distributed.send(tensor, dst=self.ranks[dst], group=metadata_group)
            else:
                # use group for GPU tensors
                torch.distributed.send(tensor, dst=self.ranks[dst], group=group)
        return None

    def recv_tensor_dict(self, src: int | None = None) -> dict[str, torch.Tensor | Any] | None:
        """Recv the input tensor dictionary.
        NOTE: `src` is the local rank of the source rank.
        """
        # Bypass the function if we are using only 1 GPU.
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return None

        group = self.device_group
        metadata_group = self.cpu_group

        if src is None:
            src = self.group_prev_rank
        assert src < self.world_size, f"Invalid src rank ({src})"

        recv_metadata_list = self.recv_object(src=src)
        tensor_dict: dict[str, Any] = {}
        for key, value in recv_metadata_list:
            if isinstance(value, TensorMetadata):
                tensor = torch.empty(value.size, dtype=value.dtype, device=value.device)
                if tensor.numel() == 0:
                    # Skip broadcasting empty tensors.
                    _update_nested_dict(tensor_dict, key, tensor)
                    continue
                if tensor.is_cpu:
                    # use metadata_group for CPU tensors
                    torch.distributed.recv(tensor, src=self.ranks[src], group=metadata_group)
                else:
                    # use group for GPU tensors
                    torch.distributed.recv(tensor, src=self.ranks[src], group=group)
                _update_nested_dict(tensor_dict, key, tensor)
            else:
                _update_nested_dict(tensor_dict, key, value)
        return tensor_dict

    def barrier(self):
        """Barrier synchronization among the group.
        NOTE: don't use `device_group` here! `barrier` in NCCL is
        terrible because it is internally a broadcast operation with
        secretly created GPU tensors. It is easy to mess up the current
        device. Use the CPU group instead.
        """
        torch.distributed.barrier(group=self.cpu_group)

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        """Sends a tensor to the destination rank in a non-blocking way"""
        """NOTE: `dst` is the rank_in_group of the destination rank."""
        if dst is None:
            dst = self.group_next_rank

        torch.distributed.send(
            tensor,
            self.ranks[dst],
            group=(self.device_groups[self.rank_in_group % 2] if self.world_size == 2 else self.device_group),
        )

    def recv(self, size: torch.Size, dtype: torch.dtype, src: int | None = None) -> torch.Tensor:
        """Receives a tensor from the src rank."""
        """NOTE: `src` is the rank_in_group of the source rank."""
        if src is None:
            src = self.group_prev_rank

        tensor = torch.empty(size, dtype=dtype, device=self.device)
        torch.distributed.recv(
            tensor,
            self.ranks[src],
            (self.device_groups[(self.rank_in_group + 1) % 2] if self.world_size == 2 else self.device_group),
        )
        return tensor

    def destroy(self):
        if self.device_group is not None:
            torch.distributed.destroy_process_group(self.device_group)
            self.device_group = None
        if self.cpu_group is not None:
            torch.distributed.destroy_process_group(self.cpu_group)
            self.cpu_group = None


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
        torch_distributed_backend: str | Backend,
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

        self.device = current_omni_platform.get_torch_device(local_rank)

        self.recv_buffer_set: bool = False
        self.recv_tasks_queue: list[tuple[str, int]] = []
        self.receiving_tasks: list[tuple[torch.distributed.Work, str, int]] = []
        self.dtype: torch.dtype | None = None
        self.num_pipefusion_patches: int | None = None

        self.recv_shape: dict[str, dict[int, torch.Size]] = {}
        self.send_shape: dict[str, dict[int, torch.Size]] = {}
        self.recv_buffer: dict[str, dict[int, torch.Size]] = {}

        # Cached dict schema and pre-allocated recv buffers for
        # `pipeline_isend_tensor_dict` / `pipeline_irecv_tensor_dict`.
        # Keyed by (name, segment_idx). Recv buffer leaf is a length-2 list
        # for double buffering. Caller picks the slot via buf_idx.
        self.dict_schema_cache: dict[tuple[str, int], list[tuple[str, Any]]] = {}
        self.dict_recv_buffer: dict[tuple[str, int], list[dict[str, torch.Tensor]]] = {}
        self._comms_stream: Any = None # Dedicated comms stream for PP P2P. None on CPU.

        self.dict_schema_keepalive: list[torch.Tensor] = []

        self.skip_tensor_recv_buffer_set: bool = False
        self.recv_skip_tasks_queue: list[int | tuple[str, int]] = []
        self.receiving_skip_tasks: list[tuple[torch.distributed.Work, str, int]] = []
        self.skip_tensor_recv_buffer: list[torch.Tensor] | torch.Tensor | None = None
        self.skip_device_group = None
        for ranks in group_ranks:
            skip_device_group = torch.distributed.new_group(ranks, backend=torch_distributed_backend)
            if self.rank in ranks:
                self.skip_device_group = skip_device_group
        assert self.skip_device_group is not None

        self._warmup_nccl_comms()

    def _warmup_nccl_comms(self) -> None:
        """Force eager ncclCommInit on every P2P group while all ranks are
        synchronized at __init__. Otherwise the first real P2P op would
        trigger a collective comm-init that blocks the early-arriving
        rank — breaks temporal-PP where one rank deliberately runs ahead.
        """
        if self.world_size == 1:
            return

        dummy = torch.zeros(1, device=self.device, dtype=torch.uint8)

        if self.world_size == 2:
            for group_idx in (0, 1):
                group = self.device_groups[group_idx]
                if self.rank_in_group == group_idx:
                    op = torch.distributed.P2POp(torch.distributed.isend, dummy, self.next_rank, group)
                else:
                    op = torch.distributed.P2POp(torch.distributed.irecv, dummy, self.prev_rank, group)
                for req in torch.distributed.batch_isend_irecv([op]):
                    req.wait()
        else:
            for req in torch.distributed.batch_isend_irecv(
                [
                    torch.distributed.P2POp(torch.distributed.isend, dummy, self.next_rank, self.device_group),
                    torch.distributed.P2POp(torch.distributed.irecv, dummy, self.prev_rank, self.device_group),
                ]
            ):
                req.wait()

        for req in torch.distributed.batch_isend_irecv(
            [
                torch.distributed.P2POp(torch.distributed.isend, dummy, self.skip_rank, self.skip_device_group),
                torch.distributed.P2POp(torch.distributed.irecv, dummy, self.skip_rank, self.skip_device_group),
            ]
        ):
            req.wait()

    def reset_buffer(self):
        self.recv_tasks_queue = []
        self.receiving_tasks = []
        self.recv_shape = {}
        self.send_shape = {}
        self.recv_buffer = {}

        self.dict_schema_cache = {}
        self.dict_recv_buffer = {}
        self.dict_schema_keepalive = []

        self.recv_skip_tasks_queue = []
        self.receiving_skip_tasks = []
        self.skip_tensor_recv_buffer = {}

    @property
    def comms_stream(self):
        """Dedicated stream for PP P2P comms."""
        if self._comms_stream is None and self.device.type != "cpu":
            mod = getattr(torch, self.device.type, None)
            if mod is not None and hasattr(mod, "Stream"):
                self._comms_stream = mod.Stream(device=self.device)
        return self._comms_stream

    def _comms_stream_ctx(self):
        """Context manager that makes ``comms_stream`` the current stream."""
        stream = self.comms_stream
        if stream is None:
            return nullcontext()
        return getattr(torch, self.device.type).stream(stream)

    def _record_compute_event(self):
        """Record an event on the default (compute) stream for later
        ``comms_stream.wait_event``."""
        if self.comms_stream is None:
            return None
        mod = getattr(torch, self.device.type)
        ev = mod.Event()
        ev.record(mod.current_stream(self.device))
        return ev

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
        name: str | None = None,
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
        send_group = (
            self.device_groups[self.rank_in_group % 2] if self.world_size == 2 else self.device_group
        )
        recv_group = (
            self.device_groups[(self.rank_in_group + 1) % 2] if self.world_size == 2 else self.device_group
        )

        ops = []
        if recv_prev:
            recv_prev_dim_tensor = torch.empty((1), device=self.device, dtype=torch.int64)
            recv_prev_dim_op = torch.distributed.P2POp(
                torch.distributed.irecv,
                recv_prev_dim_tensor,
                self.prev_rank,
                recv_group,
            )
            ops.append(recv_prev_dim_op)

        if tensor_send_to_next is not None:
            send_next_dim_tensor = torch.tensor(tensor_send_to_next.dim(), device=self.device, dtype=torch.int64)
            send_next_dim_op = torch.distributed.P2POp(
                torch.distributed.isend,
                send_next_dim_tensor,
                self.next_rank,
                send_group,
            )
            ops.append(send_next_dim_op)

        if len(ops) > 0:
            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        # To protect against race condition when using batch_isend_irecv().
        # should take this out once the bug with batch_isend_irecv is resolved.
        current_omni_platform.synchronize()

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
                recv_group,
            )
            ops.append(recv_prev_shape_op)

        if tensor_send_to_next is not None:
            send_next_shape_tensor = torch.tensor(tensor_send_to_next.size(), device=self.device, dtype=torch.int64)
            send_next_shape_op = torch.distributed.P2POp(
                torch.distributed.isend,
                send_next_shape_tensor,
                self.next_rank,
                send_group,
            )
            ops.append(send_next_shape_op)

        if len(ops) > 0:
            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        current_omni_platform.synchronize()

        recv_prev_shape = [0, 0, 0]
        if recv_prev_shape_tensor is not None:
            recv_prev_shape = recv_prev_shape_tensor
        return torch.Size(recv_prev_shape)

    def _isend_dict_schema(
        self, send_metadata: list[tuple[str, Any]]
    ) -> tuple[list[torch.distributed.Work], list[torch.Tensor]]:
        """Non-blocking schema send. Returns (handles, keepalive_tensors).
        Caller must keep the tensors alive until the handles complete.
        """
        send_group = (
            self.device_groups[self.rank_in_group % 2] if self.world_size == 2 else self.device_group
        )
        payload_bytes = pickle.dumps(send_metadata)
        payload_array = bytearray(payload_bytes)
        payload_tensor = torch.frombuffer(payload_array, dtype=torch.uint8).to(self.device)
        send_size_tensor = torch.tensor(
            [payload_tensor.numel()], device=self.device, dtype=torch.int64
        )
        # batch_isend_irecv (not plain isend) — plain P2P on size-2 PG
        # triggers lazy sub-comm creation that requires the peer present.
        ops = [
            torch.distributed.P2POp(torch.distributed.isend, send_size_tensor, self.next_rank, send_group),
            torch.distributed.P2POp(torch.distributed.isend, payload_tensor, self.next_rank, send_group),
        ]
        handles = list(torch.distributed.batch_isend_irecv(ops))
        return handles, [send_size_tensor, payload_tensor]

    def _recv_dict_schema(self) -> list[tuple[str, Any]]:
        """Blocking schema recv - must wait because the size value is
        needed before allocating the payload buffer.
        """
        recv_group = (
            self.device_groups[(self.rank_in_group + 1) % 2] if self.world_size == 2 else self.device_group
        )
        recv_size_tensor = torch.empty(1, device=self.device, dtype=torch.int64)
        for req in torch.distributed.batch_isend_irecv(
            [torch.distributed.P2POp(torch.distributed.irecv, recv_size_tensor, self.prev_rank, recv_group)]
        ):
            req.wait()
        recv_payload = torch.empty(int(recv_size_tensor.item()), device=self.device, dtype=torch.uint8)
        for req in torch.distributed.batch_isend_irecv(
            [torch.distributed.P2POp(torch.distributed.irecv, recv_payload, self.prev_rank, recv_group)]
        ):
            req.wait()
        return pickle.loads(recv_payload.cpu().numpy().tobytes())

    def pipeline_send(self, tensor: torch.Tensor, name: str = "latent", segment_idx: int = -1) -> None:
        tensor = tensor.contiguous()
        self._check_shape_and_buffer(tensor_send_to_next=tensor, name=name, segment_idx=segment_idx)
        self._pipeline_isend(tensor).wait()

    def pipeline_isend(
        self, tensor: torch.Tensor, name: str = "latent", segment_idx: int = -1
    ) -> torch.distributed.Work:
        tensor = tensor.contiguous()
        self._check_shape_and_buffer(tensor_send_to_next=tensor, name=name, segment_idx=segment_idx)
        handle = self._pipeline_isend(tensor)
        if tensor.is_cuda:
            # Keep allocator from reusing this CUDA buffer before the async send finishes.
            tensor.record_stream(torch.cuda.current_stream(tensor.device))
        return handle

    def pipeline_recv(self, idx: int = -1, name: str = "latent") -> torch.Tensor:
        name = name or "latent"
        self._check_shape_and_buffer(recv_prev=True, name=name, segment_idx=idx)
        self._pipeline_irecv(self.recv_buffer[name][idx]).wait()
        return self.recv_buffer[name][idx]

    def set_recv_dict_buffer(
        self,
        name: str,
        segment_idx: int,
        template_dict: dict[str, torch.Tensor | Any],
    ) -> None:
        """Pre-populate schema cache + a double-buffer pair (indices 0/1) for
        (name, segment_idx).
        """
        metadata_list, _ = _split_tensor_dict(template_dict)
        key = (name, segment_idx)
        self.dict_schema_cache[key] = metadata_list
        buffer_pair: list[dict[str, torch.Tensor]] = []
        for _ in range(2):
            buffers: dict[str, torch.Tensor] = {}
            for key_, value in metadata_list:
                if isinstance(value, TensorMetadata):
                    if torch.Size(value.size).numel() == 0:
                        continue
                    device = self.device if value.device == "cuda" else torch.device(value.device)
                    buffers[key_] = torch.empty(value.size, dtype=value.dtype, device=device)
            buffer_pair.append(buffers)
        self.dict_recv_buffer[key] = buffer_pair

    def pipeline_isend_tensor_dict(
        self,
        tensor_dict: dict[str, torch.Tensor | Any],
        name: str = "dict",
        segment_idx: int = -1,
    ) -> list[torch.distributed.Work]:
        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)

        key = (name, segment_idx)
        handles: list[torch.distributed.Work] = []
        if key not in self.dict_schema_cache:
            schema_handles, keepalive = self._isend_dict_schema(metadata_list)
            handles.extend(schema_handles)
            self.dict_schema_keepalive.extend(keepalive)
            self.dict_schema_cache[key] = metadata_list

        compute_done = self._record_compute_event()
        comms = self.comms_stream
        with self._comms_stream_ctx():
            if comms is not None and compute_done is not None:
                comms.wait_event(compute_done)
            for tensor in tensor_list:
                if tensor.numel() == 0:
                    continue
                tensor = tensor.contiguous()
                if tensor.is_cuda and comms is not None:
                    tensor.record_stream(comms)
                handles.append(self._pipeline_isend(tensor))
        return handles

    def pipeline_irecv_tensor_dict(
        self,
        name: str = "dict",
        segment_idx: int = -1,
        buf_idx: int = 0,
    ) -> tuple[dict[str, torch.Tensor | Any], list[torch.distributed.Work], list]:
        """Async tensor-dict recv into the ``buf_idx`` slot (0 or 1) of the
        double-buffer pair for (name, segment_idx). Caller picks the slot
        — typically ``micro_step % 2`` — so consecutive recvs alternate and
        the previous result stays readable until its consumer is done.
        Posts irecvs on ``comms_stream``.
        """
        key = (name, segment_idx)
        if key not in self.dict_schema_cache:
            metadata_list = self._recv_dict_schema()
            self.dict_schema_cache[key] = metadata_list
            buffer_pair: list[dict[str, torch.Tensor]] = []
            for _ in range(2):
                buffers: dict[str, torch.Tensor] = {}
                for k, value in metadata_list:
                    if isinstance(value, TensorMetadata):
                        if torch.Size(value.size).numel() == 0:
                            continue
                        device = self.device if value.device == "cuda" else torch.device(value.device)
                        buffers[k] = torch.empty(value.size, dtype=value.dtype, device=device)
                buffer_pair.append(buffers)
            self.dict_recv_buffer[key] = buffer_pair

        metadata_list = self.dict_schema_cache[key]
        buffers = self.dict_recv_buffer[key][buf_idx]
        comms = self.comms_stream

        tensor_dict: dict[str, Any] = {}
        handles: list[torch.distributed.Work] = []
        with self._comms_stream_ctx():
            for k, value in metadata_list:
                if isinstance(value, TensorMetadata):
                    if torch.Size(value.size).numel() == 0:
                        _update_nested_dict(
                            tensor_dict,
                            k,
                            torch.empty(value.size, dtype=value.dtype, device=self.device),
                        )
                        continue
                    tensor = buffers[k]
                    if tensor.is_cuda and comms is not None:
                        tensor.record_stream(comms)
                    handles.append(self._pipeline_irecv(tensor))
                    _update_nested_dict(tensor_dict, k, tensor)
                else:
                    _update_nested_dict(tensor_dict, k, value)

        return tensor_dict, handles, []

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
        # batch_isend_irecv (not plain irecv) — plain P2P on size-2 PG
        # triggers lazy sub-comm creation that requires the peer present.
        group = self.device_groups[(self.rank_in_group + 1) % 2] if self.world_size == 2 else self.device_group
        op = torch.distributed.P2POp(torch.distributed.irecv, tensor, self.prev_rank, group)
        return torch.distributed.batch_isend_irecv([op])[0]

    def _pipeline_isend(self, tensor: torch.tensor):
        group = self.device_groups[self.rank_in_group % 2] if self.world_size == 2 else self.device_group
        op = torch.distributed.P2POp(torch.distributed.isend, tensor, self.next_rank, group)
        return torch.distributed.batch_isend_irecv([op])[0]

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
        torch_distributed_backend: str | Backend,
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
