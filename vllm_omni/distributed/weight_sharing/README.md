# Persistent device weight sharing

The backend shares finalized, read-only weight storages between paired processes
on the same physical accelerator. It is a standalone building block; model
binding, worker setup, authenticated metadata transport and process supervision
remain the caller's responsibility. This does not enable weight sharing in a
serving configuration by itself.

## Ascend NPU codec

Use the existing backend with an explicit codec on **both** peers:

```python
from vllm_omni.distributed.weight_sharing import (
    TorchCudaIpcBackend,
    TorchNpuIpcCodec,
)

backend = TorchCudaIpcBackend(codec=TorchNpuIpcCodec())
```

`TorchCudaIpcBackend` retains its original name and defaults to CUDA. The NPU
codec overrides every device operation and reuses only device-independent view
reconstruction and storage leases. Importing the package does not load torch or
torch_npu. The NPU codec uses torch_npu's reducer/rebuilder, with a distinct
`npu-ipc-v1` descriptor ID; CUDA descriptors cannot be consumed as NPU handles.

Requirements:

- Linux, Ascend NPU and torch_npu with the 15-field tensor IPC reduction ABI.
  The tested stack is PyTorch 2.10.0 and torch_npu 2.10.0.post4 on Ascend 910B1.
- Matching model/checkpoint/layout specifications and TP rank identities.
- A physical device identity of `NPU-<uuid>`, where `<uuid>` is the canonical
  UUID returned by `torch_npu.npu.get_device_properties(local_device_index).uuid`.
  Each process supplies its own local ordinal in `SessionContext.device_index`;
  ordinal equality is not a physical-device check.
- Provider weights in the native caching allocator, outside private pools,
  allocated with `PYTORCH_NPU_ALLOC_CONF=expandable_segments:False`. Set this
  before starting the provider. The consumer need not use the same allocator.
- Ordinary ND/NCHW tensors only. Internal formats such as FRACTAL_NZ, imported
  allocations, tensor subclasses, sparse and quantized tensors are unsupported.
  Unsupported storage is rejected; the codec does not repack or copy it.

Shape, dtype, stride, storage offset, shared storage and exact aliases are
preserved. Empty storages are recreated locally without an IPC handle.

## Ownership and control plane

The existing backend lifecycle also applies to NPU:

1. Provider calls `prepare_export`, then `claim_export` to obtain metadata bytes.
2. Consumer calls `open_import` with matching identity/spec and relays its
   `receipt` to the provider's `confirm_import`.
3. Consumer drops external views/Parameters and calls `close_import`. `HELD`
   means external storage references remain; retry after dropping them.
4. After `CLOSED`, relay the release receipt to the provider's `confirm_release`,
   then call `release_export`.

All device and resource operations run on the backend's owner thread. Export
and import close synchronize the local NPU. An unclaimed export can be cancelled
exactly once. Claimed exports stay alive until confirmed peer release. Lost
receipts, partial import failures and uncertain delivery must be reported with
`mark_failed`; a poisoned sharing group requires caller-managed process teardown.
Never replay an import into a replacement consumer or mutate shared weights.

## Tests

CPU ABI/rejection tests: `test_torch_npu_ipc_codec.py`. The common contract and
backend tests continue to cover state transitions, failures and owner threads.

On one reserved NPU, run `tests/distributed/weight_sharing/test_npu_ipc.py` for
sibling-process IPC, aliases/views, external-reference lifetime, repeated
teardown, allocation measurement with a clone control, unclaimed cancellation,
and unsupported-format/allocation rejection. Tests use the parent process only
to relay control metadata; it never transfers weight tensors.
