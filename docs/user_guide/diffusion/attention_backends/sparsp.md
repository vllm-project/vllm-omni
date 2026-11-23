# SparSP demand routing for LTX-2

SparSP is an experimental CUDA attention backend for native LTX-2 video
self-attention with eager, single-node, pure Ring sequence parallelism.
Every routing policy uses the same Q128/K64 selector and NCCL process group.
K/V payloads preserve the input BF16 or FP16 bits.

| Setting | Options | Effect |
| --- | --- | --- |
| `routing` | `full_ring`, `selective_direct` | Relay full K/V or send only the requested head/block pairs. |
| `demand_planner` | `torch`, `manifest`, `per_hop` | When direct demand is exchanged and how many hops are posted ahead of local attention. |
| `placement` | `none`, `auto`, `interval_periodic` | Use contiguous ownership or assign whole query tiles by recurring spatial bands. |

Direct demand exchanges a per-head K/V block mask for each attention call.
Both endpoints derive the same head/block order and payload length. Each
sender packs those blocks into contiguous K/V packets and the receiver
places them in the layout used by sparse attention. The union over local
queries determines traffic savings.

- `torch` exchanges every hop's demand in one collective, then keeps one
  hop in flight ahead of compute.
- `manifest` exchanges every hop's demand in one collective, derives all
  packed indices, and posts every hop before local attention.
- `per_hop` is the on-demand control. It posts every hop before local
  attention, but each hop first exchanges its own demand and reads its
  packed indices back to the host. That round trip queues behind the
  previous hop's payload on the same NCCL communicator.

The default `placement: none` leaves video ownership contiguous.
`placement: auto` uses periodic ownership when each rank can own an
equal number of complete Q128 tiles, and `interval_periodic` requires
that geometry. The input permutation moves positions and token-wise
conditioning with their video tokens, preserves K64 block membership,
and is inverted after gather. At 313 frames on four ranks, `auto` uses
contiguous ownership at 384×768 and periodic ownership at 768×1536.

## Enablement

Set `parallel_config.ring_degree` to the number of ranks and
`enforce_eager` to true. In `diffusion_attention_config`, configure the
LTX-2 video self-attention role.

```yaml
per_role:
  ltx2.video_self:
    backend: SPARSP_ATTN
    sparsp:
      routing: selective_direct
      demand_planner: manifest
      cdf: 0.90
```

The backend requires complete 64-token K/V blocks per rank, matching FP16
or BF16 Q/K/V, and one CUDA device per rank. Set Ulysses, AllGather-KV,
and tensor parallel degrees to one. Use eager execution with no attention
schedule, padding, external masks, or joint video attention.

## Optional provider

Build the pinned Apache-2.0
[SpargeAttn revision](https://github.com/thu-ml/SpargeAttn/tree/bdad543f15cd8e437852fbf10fa6bcf73d923e02)
against Omni's PyTorch and CUDA installation. The patch builds a separate
`spas_sparsp` package with caller-owned LSE and current-stream launches.
Set the architecture to match your GPU.

```bash
git clone --branch return_lse https://github.com/thu-ml/SpargeAttn.git
cd SpargeAttn
git checkout bdad543f15cd8e437852fbf10fa6bcf73d923e02
git apply --unidiff-zero /path/to/vllm-omni/tools/sparsp/spargeattn_lse_output.patch
TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4 pip install --no-build-isolation .
```

## Compare routing and planning

On four PCIe GPUs with peer access, run the fixed-input comparison.

```bash
CUDA_VISIBLE_DEVICES=1,4,5,6 NCCL_P2P_LEVEL=SYS NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=1 \
  torchrun --standalone --nproc_per_node=4 -m benchmarks.diffusion.bench_sparsp \
  --qkv /path/to/canonical-bshd-qkv.pt --cdf 0.9 --warmup 10 --iterations 200 \
  --output /tmp/sparsp.json
```

The archive must contain `q`, `k`, and `v` in canonical BSHD token order.
The comparison brackets the demand planners with full-ring NCCL controls
and compares the planners at identical K/V payload size. It verifies
bitwise output equality and reports the NCCL version, attention-call
latency, and bytes sent across all ranks. Add `--placement-grid 40 24 48`
for the 46,080-token periodic refinement capture.
