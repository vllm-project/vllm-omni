# Optional FlashInfer PCIe/RDMA transport

Strict Ulysses can use FlashInfer's registered PCIe transport when explicitly
selected. It requires a FlashInfer build containing the PCIe
`UlyssesCommunicator` API from
[FlashInfer #4876](https://github.com/flashinfer-ai/flashinfer/pull/4876), plus
libibverbs/libmlx5 development headers and a supported local GPU/NIC topology.
That dependency is not yet part of this change or installed automatically.

```bash
export VLLM_OMNI_ULYSSES_A2A_BACKEND=flashinfer-pcie
export VLLM_OMNI_FLASHINFER_ULYSSES_REQUIRE_RDMA=1
export FLASHINFER_ULYSSES_PCIE_ROUTE=rdma
# Size for the largest per-rank attention operand in the chosen workload.
export VLLM_OMNI_FLASHINFER_ULYSSES_MAX_BYTES=268435456
# Add to the existing server command:
# --ulysses-a2a-permute
```

RDMA is required by default for this explicit backend. An unavailable RDMA
route or an operand exceeding capacity raises an error. Setting
`VLLM_OMNI_FLASHINFER_ULYSSES_REQUIRE_RDMA=0` permits PCIe routes and a logged
NCCL fallback for cold initialization failure or larger operands. Kernel
execution errors are propagated. The normal Ulysses transport remains the
default when this backend is not selected.

Registered Q/K/V/gate and reverse-output slots remain separate and are reused
across layers. Process-group teardown collectively closes the communicator
before releasing its registered storage. Selection does not enable model
projection overlap, attention quantization, or a different sparsity algorithm.

The RDMA route supports batch size 1. Required-RDMA mode rejects larger batches
before entering the native collective; optional mode uses NCCL for those batches.
