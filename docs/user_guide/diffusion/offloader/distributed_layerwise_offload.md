# Distributed Layerwise Offloading

Distributed layerwise offloading (DLO) extends block streaming to multi-device
deployments. With AllGather enabled, each rank supplies roughly `1 / dp_size`
of a block from persistent host memory and reconstructs it at runtime. Without
AllGather, each rank streams a complete block independently.
Compatible TP1 deployments can share mmap-backed host pages among processes on
the same node; otherwise DLO streams the ordinary loader's rank-local tensors.

See the [DLO feature design](../../../design/feature/offloader/distributed_layerwise_offload.md)
for the implementation contract and compatibility matrix.

## Execution model

DLO overlaps three operations with a fixed two-block device buffer:

```text
Compute stream:  [Layer N]          [Layer N+1]        [Layer N+2]
H2D stream:      [H2D shard N+1]    [H2D shard N+2]
AllGather:       [AG N+1]           [AG N+2]
Slots:           slot 0: Layer N    slot 1: Layer N+1
```

AllGather communicates only request-independent weight shards, so data-
parallel ranks may process different requests concurrently.

## Usage

```bash
# Four ranks with sharded host weights and AllGather
vllm serve /path/to/model --omni \
  --enable-distributed-layerwise-offload \
  --data-parallel-size 4

# Standard-loader rank-local weights, without DLO AllGather
vllm serve /path/to/model --omni \
  --enable-distributed-layerwise-offload \
  --data-parallel-size 4 \
  --dlo-no-use-allgather

# Sequence parallel deployment
vllm serve /path/to/model --omni \
  --enable-distributed-layerwise-offload \
  --usp 4
```

```python
from vllm_omni import Omni

omni = Omni(
    model="/path/to/model",
    enable_distributed_layerwise_offload=True,
    dlo_use_allgather=True,
)
```

## Flags

| Flag | Meaning | Default |
| --- | --- | --- |
| `--enable-distributed-layerwise-offload` | Enable DLO | `false` |
| `--data-parallel-size N` | DP ranks and AllGather weight-sharding group | `1` |
| `--dlo-use-allgather` | Shard host weights and reconstruct with AllGather | `true` |
| `--dlo-no-use-allgather` | Stream complete rank-local blocks without a DLO weight collective | `false` |
| `--dlo-resident-layers N` | Keep N leading main-DiT blocks on device; requires no-AllGather and model-declared resident paths | `0` |
| `--host-weight-runtime-mode {disabled,preferred,required}` | HWR policy: disabled, prefer an exact artifact with fallback, or require an exact artifact | `disabled` |
| `--host-weight-runtime-root PATH` | Writable node-local HWR store shared by workers in one storage domain; required for `preferred` and `required` | unset |
| `--dlo-host-registration-limit-gib N` | Optional per-worker ceiling for registering an HWR mmap; zero adds no ceiling | `0` |

## Host-weight loading

The diffusion loader chooses host storage before DLO is enabled. It first
attempts to build a complete, validated direct-checkpoint mmap plan. If names,
coverage, shape, dtype, topology, or loader-callback compatibility cannot be
proven, it runs the ordinary model loader instead. DLO consumes that result and
does not make a second checkpoint-compatibility decision.

The shared-mmap optimization in this phase is supported only with TP1. TP
greater than one falls back before model mutation to ordinary TP-aware loading.
DLO may still consume those TP-local tensors, but this is a compatibility path:
it does not share checkpoint-backed runtime weights across DP replicas and
provides no shared-mmap host-memory guarantee.

The mmap plan skips only dedicated DiT weight sources. Other component sources,
such as a text encoder loaded through the shared diffusion loader, continue to
use their ordinary component loader. A checkpoint source that mixes DiT and
non-DiT weights falls back completely rather than leaving an unplanned
component uninitialized.

With direct checkpoint mmap, the loader:

1. saves non-persistent buffers such as RoPE frequencies;
2. moves the normally created transformer to the meta device;
3. loads checkpoint tensors as mmap views backed by the shared OS page cache;
4. applies any loader-owned bounded layout adapters while packing blocks;
5. restores saved non-persistent buffers; and
6. preserves `post_load_weights()` and `validate_loaded_weights()` lifecycle
   hooks.

For AllGather with a group larger than one, each process copies only its
persistent shard and then releases the source mapping. For no-AllGather, each
process keeps the mapping open and packs complete blocks through two bounded
pinned staging slots. Processes mapping the same files on one node share the
immutable pages; no-AllGather still performs a complete-block H2D copy in each
process.

When the effective DLO group size is one, `dlo_use_allgather=True` does not
perform a collective and uses the same rank-local transfer behavior.

### BF16 Host Weight Runtime

HWR is opt-in for models that declare the BF16 final-layout artifact contract.
The current AllGather promotion target is MiniMax H3 on DP2/TP1.
FLUX.2-klein-4B declares and validates the same artifact contract, but its
AllGather transport remains outside the promoted end-to-end scope until
model-specific CUDA evidence is added. FLUX.2-klein-9B, FLUX.2-dev, online FP8,
HSDP, LoRA/adapted weights, and non-default load formats remain outside the
validated scope.

```bash
vllm serve /path/to/model --omni \
  --enable-distributed-layerwise-offload \
  --host-weight-runtime-mode preferred \
  --host-weight-runtime-root /var/cache/vllm-omni/hwr
```

Both DLO transports consume the same finalized HWR artifact:

| DLO transfer | Exact-hit behavior |
| --- | --- |
| AllGather with more than one rank | Copy one persistent padded CPU shard per rank, release the artifact mapping during setup, and reconstruct complete blocks with the existing collective |
| No-AllGather | Restore the artifact and retain its mapping as the host master through service |

The policy is identical for both transports:

- `preferred` restores an exact hit; on a miss it uses a complete checkpoint
  plan for the current startup while publishing the artifact in parallel,
  falling back to canonical loading and post-load publication only when direct
  production is unavailable;
- `required` consumes only an existing exact artifact and fails the complete
  cohort if any rank misses or cannot validate/restore it.

Populate every node-local HWR storage domain with a matching preferred startup
before switching to required:

```text
preferred cold startup -> checkpoint plan + parallel final-layout publication
required warm startup  -> exact artifact hit -> restore -> DLO transport
```

Every AllGather rank must agree on eligibility, artifact identity and content,
restore planning, commit, warm finalization, and final transport layout. A
rank-local failure is reported at the same rendezvous by every peer, so no rank
returns early while another enters a collective.

Registered direct H2D and `--dlo-host-registration-limit-gib` apply only to the
no-AllGather HWR transport. AllGather closes the artifact lease after extracting
its persistent rank-local CPU shard during backend setup.

## Declarative topology

Models may declare an `OffloadPlan` instead of embedding offload logic:

```python
from vllm_omni.diffusion.offloader import OffloadPlan


class MyPipeline(nn.Module):
    _dit_modules = ["transformer"]
    _offload_plan = OffloadPlan(
        block_attrs={"transformer": ("blocks",)},
        offload_submodules={"context_encoder": "layers"},
    )
```

When no plan exists, discovery falls back to
`_layerwise_offload_blocks_attrs` and then heuristic attribute lookup.

## Data-parallel concurrency

With `data_parallel_size > 1` and AllGather enabled, the scheduler can process
up to `dp_size` requests per denoising step. Every concurrent request must set
the same explicit `num_inference_steps`; `None` is rejected because every rank
must enter each collective.

## Limitations

- Direct checkpoint mmap currently requires TP1. TP greater than one is
  outside the Phase A shared-mmap support scope and falls back before model
  mutation to the ordinary TP-aware loader. DLO can stream that runtime layout,
  and eligible BF16 configurations may use HWR final-layout artifacts,
  but direct checkpoint mmap provides no shared-mmap host-memory guarantee for
  TP greater than one.
- HSDP plus AllGather is rejected to avoid double sharding. HSDP without
  AllGather has limited end-to-end validation.
- Per-tensor online FP8 linears use the ordinary loader and can run with either
  DLO transfer path. With AllGather, every rank temporarily materializes the
  complete FP8 model in host memory before DLO retains only its shard. Other
  online quantization methods require no-AllGather until their runtime layouts
  are validated.
- Resident leading layers require `--dlo-no-use-allgather` and a model
  `OffloadPlan` that declares eligible `resident_dit_paths`.
- DP concurrency requires an explicit, identical inference-step count.

Sharing quantized or otherwise unvalidated transformed layouts through a
normalized HWR producer is a follow-up design in
[RFC #6195](https://github.com/vllm-project/vllm-omni/issues/6195), not part of
the current BF16 final-layout path.

See the [Cosmos3 DistOffload recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/cosmos3/Cosmos3-DistOffload.md)
for an end-to-end example.
