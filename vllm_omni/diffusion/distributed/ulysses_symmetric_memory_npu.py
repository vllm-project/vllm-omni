# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Eager NPU Ulysses transport using symmetric memory and a fused AscendC kernel.

The existing ``--ulysses-a2a-permute`` path uses one kernel per exchange to
synchronize ranks, pull peer data and relayout it for attention. All WORLD
ranks must participate in the same order and stream in one HCCS domain.
Requires TorchNPU's NPUSHMEM backend, CANN SHMEM/AscendC headers, a C++ toolchain
and libshmem.so on LD_LIBRARY_PATH.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
from functools import cache
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from filelock import FileLock
from torch.distributed.distributed_c10d import _resolve_process_group
from vllm.logger import init_logger

logger = init_logger(__name__)
_WORKSPACE = None
_WORKSPACE_GROUP = None


def _resolve_soc_version() -> str:
    """Return the exact Ascend target used by the AscendC compiler.

    ``ASCENDC_SOC_VERSION`` is an explicit cross-compile override. Runtime
    builds otherwise use the active device name so a cached binary cannot be
    silently compiled for a different SoC family or variant.
    """
    override = os.environ.get("ASCENDC_SOC_VERSION")
    if override and override.strip():
        return override.strip()
    try:
        soc_version = torch.npu.get_device_name(torch.npu.current_device())
    except Exception as exc:
        raise RuntimeError(
            "Unable to determine the active Ascend SoC; set ASCENDC_SOC_VERSION only when intentionally cross-compiling"
        ) from exc
    if not isinstance(soc_version, str) or not soc_version.strip():
        raise RuntimeError(f"Ascend returned an invalid SoC name: {soc_version!r}")
    return soc_version.strip()


def _update_version_metadata(fingerprint: hashlib._Hash, root: Path) -> None:
    """Include installed toolchain metadata so upgrades cannot reuse a build."""
    for relative in (
        "version.info",
        "opp/version.info",
        "aarch64-linux/ascend_toolkit_install.info",
    ):
        metadata = root / relative
        if metadata.is_file():
            fingerprint.update(relative.encode())
            fingerprint.update(metadata.read_bytes())


@cache
def ensure_a2a_permute_available() -> None:
    """Compile/load once per process; serialize builds across worker ranks."""
    import torch_npu
    from torch.utils.cpp_extension import get_default_build_root, load

    source = Path(__file__).parent / "csrc/ulysses_symmetric_memory_fused"
    soc_version = _resolve_soc_version()
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")).resolve()
    backend_spec = importlib.util.find_spec("shmem")
    backend_path = os.environ.get("SHMEM_ROOT")
    if backend_path:
        backend_root = Path(backend_path).resolve()
    elif backend_spec is not None and backend_spec.submodule_search_locations:
        backend_root = Path(next(iter(backend_spec.submodule_search_locations))).resolve()
    else:
        raise RuntimeError("Install the CANN SHMEM package or set SHMEM_ROOT to its source/include root")
    if not (backend_root / "include/device/gm2gm/shmem_device_cc.h").is_file():
        raise RuntimeError(f"CANN SHMEM device headers not found under {backend_root}")
    fingerprint = hashlib.sha256()
    for path in sorted(source.iterdir()):
        if path.is_file() and path.suffix in {".cmake", ".cpp", ".h"}:
            fingerprint.update(path.name.encode())
            fingerprint.update(path.read_bytes())
    fingerprint.update(soc_version.encode())
    fingerprint.update(str(cann).encode())
    fingerprint.update(str(backend_root).encode())
    _update_version_metadata(fingerprint, cann)
    _update_version_metadata(fingerprint, backend_root)
    fingerprint.update(str(torch.__version__).encode())
    fingerprint.update(str(getattr(torch_npu, "__version__", "unknown")).encode())
    build_fingerprint = fingerprint.hexdigest()[:16]
    cache_dir = Path(get_default_build_root()) / "omni_ulysses_symmetric_memory_fused" / build_fingerprint
    cache_dir.mkdir(parents=True, exist_ok=True)
    build = cache_dir / "ascendc"
    library = build / "lib/libulysses_symmetric_memory_fused.so"
    with FileLock(str(cache_dir / "build.lock")):
        if not library.is_file():
            # Legacy CANN links device objects in place. Retry failed builds
            # from a fresh, private directory rather than incrementally.
            if build.exists():
                shutil.rmtree(build)
            env = dict(os.environ, ASCEND_HOME_PATH=str(cann))
            log_path = cache_dir / "build.log"
            try:
                with log_path.open("w") as log:
                    subprocess.run(
                        [
                            "cmake",
                            "-S",
                            str(source),
                            "-B",
                            str(build),
                            "-DCMAKE_BUILD_TYPE=Release",
                            f"-DSHMEM_ROOT={backend_root}",
                            f"-DSOC_VERSION={soc_version}",
                        ],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                    subprocess.run(
                        ["cmake", "--build", str(build), "-j", os.environ.get("MAX_JOBS", "2")],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            except subprocess.CalledProcessError as exc:
                # An unsuccessful CMake link must never become a cache hit.
                library.unlink(missing_ok=True)
                raise RuntimeError(f"AscendC Ulysses build failed; see {log_path}") from exc
            if not library.is_file():
                raise RuntimeError(f"AscendC build produced no library; see {log_path}")
    npu = Path(torch_npu.__file__).parent
    load(
        name=f"omni_ulysses_symmetric_memory_fused_bridge_{build_fingerprint}",
        sources=[str(source / "bridge.cpp")],
        extra_include_paths=[str(npu / "include"), str(npu / "include/third_party/acl/inc"), str(cann / "include")],
        extra_ldflags=[str(library), f"-Wl,-rpath,{library.parent}", f"-L{npu / 'lib'}", "-ltorch_npu", "-ldl"],
        extra_cflags=["-O2"],
        with_cuda=False,
        is_python_module=False,
        verbose=False,
    )


class NPUUlyssesSymmetricMemory:
    """Single-stream symmetric workspace using fused AscendC exchanges.

    Inference-only, one stream, WORLD group, in one HCCS domain. Head-shard
    rows must fit the 32 KiB UB tile. Every group size uses the same
    tile-major peer-interleaved MIX kernel with two AIV workers per peer.
    """

    def __init__(self, capacity_bytes: int) -> None:
        if not dist.is_initialized() or dist.get_world_size() < 2:
            raise RuntimeError("Symmetric memory exchange requires an initialized WORLD with at least two ranks")
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("NPU symmetric memory Ulysses requires eager execution; graph capture is unsupported")
        self.device = torch.device("npu", torch.npu.current_device())
        if symm_mem.get_backend(self.device) != "NPUSHMEM":
            raise RuntimeError("TorchNPU with the NPUSHMEM symmetric-memory allocator is required")
        self.world = dist.get_world_size()
        self.rank = dist.get_rank()
        self.capacity_bytes = capacity_bytes
        self.stream = torch.npu.current_stream(self.device)
        ensure_a2a_permute_available()
        symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)
        self.allocation = symm_mem.empty(capacity_bytes, dtype=torch.uint8, device=self.device)
        self.handle = symm_mem.rendezvous(self.allocation, dist.group.WORLD.group_name)
        if len(self.handle.buffer_ptrs) != self.world or any(ptr == 0 for ptr in self.handle.buffer_ptrs):
            raise RuntimeError("All WORLD peers must be directly accessible in one HCCS domain")
        self.peer_ptrs = torch.tensor(self.handle.buffer_ptrs, dtype=torch.int64, device=self.device)
        self._qkv_staging_key = None
        self._qkv_staging_views = None
        self._closed = False

    def _check_stream(self) -> None:
        if self._closed:
            raise RuntimeError("Symmetric memory workspace is closed")
        if torch.npu.current_device() != self.device.index:
            raise RuntimeError("Symmetric memory workspace must be used on its original NPU device")
        if torch.npu.current_stream(self.device) != self.stream:
            raise RuntimeError("Symmetric memory workspace must be used on a single NPU stream")
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("NPU symmetric memory Ulysses requires eager execution; graph capture is unsupported")

    def _row_bytes(self, tensor: torch.Tensor, reverse: bool = False) -> int:
        """Validate inputs and return the bytes copied per head-shard row."""
        if tensor.device != self.device:
            raise ValueError("Input must be on the workspace NPU device")
        if tensor.requires_grad:
            raise ValueError("NPU symmetric memory Ulysses is inference-only")
        if tensor.ndim != 4 or any(tensor.shape[i] <= 0 for i in (0, 2, 3)):
            raise ValueError("Input must be [B, S, H, D] with positive batch, heads and head size")
        if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError("Supported dtypes: bfloat16, float16, float32")
        if reverse:
            if tensor.shape[1] % self.world:
                raise ValueError("Global sequence length must be divisible by WORLD size")
        elif tensor.shape[2] % self.world:
            raise ValueError("Head count must be divisible by WORLD size; pad/repeat GQA heads upstream")
        row_bytes = tensor.shape[2] * tensor.shape[3] * tensor.element_size()
        if not reverse:
            row_bytes //= self.world
        if row_bytes % 32:
            raise ValueError("Head shard rows must be 32-byte aligned for the tiled NPU copy")
        if row_bytes > 32768:
            raise ValueError("Fused NPU exchange supports at most 32 KiB per head shard row")
        return row_bytes

    def __call__(self, tensor: torch.Tensor, *, reverse: bool = False) -> torch.Tensor:
        self._check_stream()
        row_bytes = self._row_bytes(tensor, reverse)
        batch, seq, heads, dim = tensor.shape
        seq_local = seq // self.world if reverse else seq
        shape = (
            (batch, seq_local, heads * self.world, dim)
            if reverse
            else (batch, seq * self.world, heads // self.world, dim)
        )
        required_bytes = tensor.numel() * tensor.element_size()
        if required_bytes > self.capacity_bytes:
            raise ValueError(f"Input requires {required_bytes} bytes; capacity is {self.capacity_bytes}")
        output = tensor.new_empty(shape)
        # Strict Ulysses requires matching shapes on every rank. Joint-only
        # CFG prefill therefore has no sequence payload to exchange anywhere.
        if seq_local == 0:
            return output
        staging = self.allocation[:required_bytes].view(tensor.dtype).view(tensor.shape)
        # copy_ also accepts row-strided Q/K/V views from packed projections.
        staging.copy_(tensor)
        torch.ops.omni_ulysses_symmetric_memory_fused.pull(
            self.peer_ptrs, output, self.rank, self.world, batch, seq_local, row_bytes, reverse
        )
        return output

    def qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Exchange Q/K/V together, including unequal Q and KV head counts.

        All three inputs use disjoint symmetric regions. Both device barriers
        surround the entire group, so no peer can reuse a region prematurely.
        """
        self._check_stream()
        tensors = (query, key, value)
        rows = [self._row_bytes(tensor) for tensor in tensors]
        for tensor in tensors:
            if tensor.dtype != query.dtype or tensor.shape[:2] != query.shape[:2]:
                raise ValueError("Q/K/V must share dtype and batch/sequence dimensions")
        sizes = [tensor.numel() for tensor in tensors]
        required_bytes = sum(sizes) * query.element_size()
        if required_bytes > self.capacity_bytes:
            raise ValueError(f"Q/K/V require {required_bytes} bytes; capacity is {self.capacity_bytes}")
        batch, seq = query.shape[:2]
        outputs = tuple(
            tensor.new_empty((batch, seq * self.world, tensor.shape[2] // self.world, tensor.shape[3]))
            for tensor in tensors
        )
        if seq == 0:
            return outputs
        offsets = (0, sizes[0], sizes[0] + sizes[1])
        staging_key = (query.dtype, tuple(tensor.shape for tensor in tensors))
        if staging_key == self._qkv_staging_key:
            staging_views = self._qkv_staging_views
        else:
            staging = self.allocation[:required_bytes].view(query.dtype)
            staging_views = tuple(
                staging[offset : offset + size].view(tensor.shape)
                for tensor, offset, size in zip(tensors, offsets, sizes)
            )
            # Retain only the last configuration; views alias the fixed
            # allocation, while every call still copies fresh input data.
            self._qkv_staging_key = staging_key
            self._qkv_staging_views = staging_views
        for tensor, staging_view in zip(tensors, staging_views):
            staging_view.copy_(tensor)
        torch.ops.omni_ulysses_symmetric_memory_fused.qkv(
            self.peer_ptrs, *outputs, self.rank, self.world, batch, seq, *rows
        )
        return outputs

    def close(self) -> None:
        """Finish queued exchanges before releasing symmetric memory."""
        self._check_stream()
        # Every fused exchange ends with a cross-rank device barrier, which
        # guarantees peers have finished reading the staging workspace.
        torch.npu.synchronize(self.device)
        self._qkv_staging_views = None
        self._qkv_staging_key = None
        self.peer_ptrs = None
        self.handle = None
        self.allocation = None
        self._closed = True


def _get_workspace(tensors: tuple[torch.Tensor, ...], group_name: str, world_size: int) -> NPUUlyssesSymmetricMemory:
    """Share one workspace across layers; grow collectively for larger requests.

    Strict Ulysses already requires matching tensor shapes across ranks. The
    resulting byte capacity and growth decisions must consequently match too.
    The group must contain every WORLD rank in the same order, since the
    device barrier is collective over WORLD.
    """
    global _WORKSPACE, _WORKSPACE_GROUP
    group = _resolve_group(group_name, world_size)
    if tensors[0].device.type != "npu":
        raise ValueError("NPU ulysses_a2a_permute requires NPU tensors")
    if _WORKSPACE_GROUP is not None and group is not _WORKSPACE_GROUP:
        raise RuntimeError(
            "NPU symmetric memory Ulysses supports a single process group; clear it before reinitialization"
        )
    if _WORKSPACE is None:
        if dist.get_process_group_ranks(group) != list(range(dist.get_world_size())):
            raise ValueError("NPU symmetric memory Ulysses requires the Ulysses group to match WORLD rank order")
    required = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    if _WORKSPACE is not None:
        if required <= _WORKSPACE.capacity_bytes:
            return _WORKSPACE
        _WORKSPACE.close()
        _WORKSPACE = None
    # The first exchange may be a joint-only CFG prefill with zero image
    # tokens. Rendezvous still needs a positive symmetric allocation.
    capacity = max(64 << 20, ((required + (64 << 20) - 1) // (64 << 20)) * (64 << 20))
    _WORKSPACE = NPUUlyssesSymmetricMemory(capacity)
    _WORKSPACE_GROUP = group
    logger.info(
        "NPU symmetric memory Ulysses initialized (rank=%d, world=%d, capacity_bytes=%d)",
        dist.get_rank(),
        dist.get_world_size(),
        capacity,
    )
    return _WORKSPACE


def _resolve_group(group_name: str, world_size: int):
    group = _resolve_process_group(group_name)
    if dist.get_world_size(group) != world_size:
        raise ValueError("world_size must match the Ulysses process group")
    return group


def clear_ulysses_symmetric_memory_workspace() -> None:
    """Collectively release the workspace before destroying process groups."""
    global _WORKSPACE, _WORKSPACE_GROUP
    if _WORKSPACE is not None:
        _WORKSPACE.close()
        _WORKSPACE = None
    _WORKSPACE_GROUP = None


def ulysses_qkv_fwd(x: torch.Tensor, group_name: str, world_size: int) -> torch.Tensor:
    return _get_workspace((x,), group_name, world_size)(x)


def ulysses_qkv_fwd_group(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    group_name: str,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _get_workspace((query, key, value), group_name, world_size).qkv(query, key, value)


def ulysses_o_rev(y: torch.Tensor, group_name: str, world_size: int) -> torch.Tensor:
    return _get_workspace((y,), group_name, world_size)(y, reverse=True)
