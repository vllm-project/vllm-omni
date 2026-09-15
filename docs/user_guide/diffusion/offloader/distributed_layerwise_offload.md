# Distributed Layerwise Offloading

Distributed layerwise offloading (DLO) extends block streaming to multi-device
deployments. Each selected component chooses its transfer independently. With
`allgather`, each rank stores roughly one shard per member of the effective DLO
group (DP, or SP when DP is one) and reconstructs each layer at runtime. With
`rank-local`, each rank streams a complete loader-produced block independently.
Compatible TP1 DiT deployments can share checkpoint-backed host pages among
processes on the same node; otherwise DLO streams the ordinary loader's
rank-local tensors.

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

The public API describes intent in one nested configuration:

| Level | Setting | Question answered |
| --- | --- | --- |
| Granularity | `mode` | move a complete component (`module`) or stream its blocks (`layer`) |
| Selection | `components` list | whether `dit` and/or `text_encoder` may move to CPU |
| Per-component layer options | `layer_options` | choose rank-local or AllGather weight transfer and optionally retain leading DiT blocks |

`weight_transfer` applies only to model weights. It does not change encoder
activations or the encoder's compute parallelism. `rank-local` copies each
rank's loader-produced block to its device. `allgather` stores one shard of
that block per rank, copies each shard to its device, and reconstructs the
complete block with a device-side AllGather before compute.

The historical `enable_cpu_offload`, `enable_layerwise_offload`, and
`enable_distributed_layerwise_offload` fields remain compatibility aliases
for existing callers and model-specific stage lifecycles. Existing
combinations retain their priority (`distributed layerwise` > `layerwise` >
`module`). A compact config rejects conflicting compatibility strategies and
non-default legacy DLO tuning.
`distributed-layerwise` remains an internal backend name; users select
`mode="layer"`, and the runtime chooses the capable backend.

## Usage

```bash
# Four SP ranks; Wan declares replicated text-encoder weights, so both
# selected components can use AllGather safely.
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --omni \
  --diffusion-offload-config \
  '{"mode":"layer","components":["dit","text_encoder"],"layer_options":{"dit":{"weight_transfer":"allgather"},"text_encoder":{"weight_transfer":"allgather"}}}' \
  --usp 4

# Mix transfers independently: sharded DiT, loader-produced encoder layout
vllm serve /path/to/model --omni \
  --diffusion-offload-config \
  '{"mode":"layer","components":["dit","text_encoder"],"layer_options":{"dit":{"weight_transfer":"allgather"},"text_encoder":{"weight_transfer":"rank-local"}}}' \
  --usp 4

# Full-topology rank-local DLO for an existing model-specific lifecycle
vllm serve /path/to/model --omni \
  --enable-distributed-layerwise-offload \
  --dlo-no-use-allgather \
  --usp 4
```

A compact layer config whose selected components are all `rank-local` and
whose `resident_layers` is zero resolves to the ordinary layerwise backend. It
uses the ordinary loader and does not request DLO's direct-checkpoint mmap.
Keep the compatibility DLO flags for a model-specific full-topology rank-local
lifecycle; do not add resident layers solely to force backend selection.

```python
from vllm_omni import Omni

omni = Omni(
    model="/path/to/model",
    diffusion_offload_config={
        "mode": "layer",
        "components": ["dit", "text_encoder"],
        "layer_options": {
            "dit": {
                "weight_transfer": "rank-local",
                "resident_layers": 20,
            },
            "text_encoder": {
                "weight_transfer": "allgather",
            },
        },
        "pin_memory": True,
    },
)
```

## Flags

| Setting | Meaning | Default |
| --- | --- | --- |
| `diffusion_offload_config.mode` | `module` or `layer` granularity | required |
| `diffusion_offload_config.components` | Non-empty list containing `dit`, `text_encoder`, or both | required |
| `layer_options.NAME.weight_transfer` | `rank-local` or `allgather` | `rank-local` |
| `layer_options.dit.resident_layers` | Leading main-DiT blocks kept on device; requires `rank-local` and model-declared resident paths | `0` |
| `diffusion_offload_config.pin_memory` | Pin streamed host memory for faster H2D copies | `true` |
| `--data-parallel-size N` | DP ranks; DP is the DLO group when greater than one, otherwise SP is used | `1` |
| `--host-weight-runtime-mode {disabled,preferred,required}` | HWR policy: no interaction, populate on a miss, or require an exact hit | `disabled` |
| `--host-weight-runtime-root PATH` | Writable node-local HWR store shared by workers in one storage domain; required for `preferred` and `required` | unset |
| `--dlo-host-registration-limit-gib N` | Optional per-worker ceiling for registering an HWR mmap; zero adds no ceiling | `0` |

HWR and host-registration tuning remain on the compatibility DLO interface and
cannot be combined with `diffusion_offload_config` in this release.

## Component and weight-transfer matrix

| Component | `allgather` | `rank-local` |
| --- | --- | --- |
| DiT | Shard each loader-visible block across the DLO DP group, or the SP group when DP is one | Stream each rank's complete loader-produced block |
| Text encoder | Same shard + AllGather path when the model declares identical encoder weights across the DLO group | Stream each rank's complete encoder block, including encoder-TP shards |

The text encoder's compute group and weight-transfer group are separate
concepts. An encoder TP group owns different parameter shards, so it is not a
valid AllGather offload group. Models opt in through
`OffloadPlan.encoder_dlo_weight_replication`; otherwise multi-rank encoder
AllGather fails at startup with guidance to use `rank-local`. This makes SP
AllGather available to replicated encoders without silently corrupting an
encoder-TP layout.

`resident_layers` remains a DiT-only setting and currently requires the DiT
transfer to be `rank-local`.

## Host-weight loading

When the distributed backend is selected by AllGather, resident DiT layers, or
the compatibility DLO flag, the diffusion loader chooses host storage before
DLO is enabled. It first
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

For a DiT using `allgather` with a group larger than one, each process copies
only its persistent shard and then releases the source mapping. For a DiT using
`rank-local`, each process keeps the mapping open and packs complete blocks
through two bounded pinned staging slots. Processes mapping the same files on
one node share the immutable pages; rank-local transfer still performs a
complete-block H2D copy in each process.

When the effective DLO group size is one, `weight_transfer="allgather"` does not
perform a collective and uses the same rank-local transfer behavior.

### Final-layout Host Weight Runtime

HWR is an opt-in startup optimization for models that declare the final-layout
BF16 restore contract. Use it only when the selected DiT uses `rank-local`:

The validated BF16 model contracts currently cover MiniMax H3 and
`black-forest-labs/FLUX.2-klein-4B`. FLUX.2-klein-9B shares the same model
class but has not been validated against this contract. FLUX.2-dev, online
FP8, HSDP, LoRA/adapted weights, and non-default load formats remain outside
the validated scope.

```bash
vllm serve /path/to/model --omni \
  --enable-distributed-layerwise-offload \
  --dlo-no-use-allgather \
  --host-weight-runtime-mode preferred \
  --host-weight-runtime-root /var/cache/vllm-omni/hwr
```

#### Choosing a mode

| Mode | Exact local artifact hit | Miss or recoverable artifact/store problem | Intended use |
| --- | --- | --- | --- |
| `disabled` | HWR is not consulted | Use the existing checkpoint-mmap or ordinary-loader path | Default compatibility path |
| `preferred` | Restore the final-layout artifact | Load canonically, serve with those tensors, and attempt to publish an artifact for the next startup | Normal deployment and store population |
| `required` | Restore the final-layout artifact | Fail startup without canonical DiT fallback or publication | Enforce a pre-populated store in controlled rollouts or CI |

Both enabled modes still fail on non-retryable configuration, identity, or
compatibility errors. `preferred` is a fallback policy for a cache miss or a
recoverable cache problem; it does not hide an invalid deployment.

#### Populating a store for `required` mode

`required` is deliberately consume-only. PR2 does not include a separate
prewarm command, so populate each node-local storage domain with one matching
`preferred` producer cohort:

1. Choose a persistent, writable root visible to every diffusion worker in the
   storage domain. Do not use a process-private temporary directory.
2. Start the deployment with `preferred`, using the exact model revision,
   dtype, TP size, SP configuration, and other weight-layout settings intended
   for serving. No inference request is needed; publication happens during
   model startup. Wait for the engine to become healthy, then shut it down
   cleanly.
3. Restart with the same arguments and root, changing only the mode to
   `required`. A successful startup proves that every worker acquired a valid
   artifact; a miss, corrupt artifact, or incompatible identity fails startup.
4. Repeat the population start on every node or storage domain because this
   store is node-local.

For example:

```bash
# First startup: canonically load and populate the exact artifacts.
vllm serve /path/to/model --omni \
  --enable-distributed-layerwise-offload \
  --dlo-no-use-allgather \
  --host-weight-runtime-mode preferred \
  --host-weight-runtime-root /var/cache/vllm-omni/hwr

# After a healthy startup and clean shutdown, enforce cache hits.
vllm serve /path/to/model --omni \
  --enable-distributed-layerwise-offload \
  --dlo-no-use-allgather \
  --host-weight-runtime-mode required \
  --host-weight-runtime-root /var/cache/vllm-omni/hwr
```

Include the same TP and SP layout flags in both commands. DP rank and DP size
are excluded from artifact identity, so the population and serving DP sizes may
differ and equivalent DP replicas share artifacts. TP rank is included, so a
TP-N deployment normally needs N rank-specific artifacts in each storage
domain; launching the matching TP cohort creates that set. If the model
revision or layout changes, run `preferred` again for the new identity before
returning to `required`.

On a cold start, the canonical loader remains authoritative and publishes a
validated final-layout artifact for later workers. A warm start restores the
DiT final tensors without ordinary DiT materialization. DLO then attempts to
register the immutable mapping for direct H2D; if registration is unavailable,
it streams through the same two bounded host staging slots. The artifact
identity includes the TP rank/size and SP layout, so TP1, TP2 rank-local shards,
and distinct SP layouts do not alias one another. Publication failure remains
separate from a `preferred` serving startup; the next `required` startup
provides the explicit artifact-availability check.

For local canonical checkpoints, the first eligible worker may hash source
shards to establish immutable identity. HWR caches those digests in the same
node-local storage domain and validates file metadata before reuse, so later
workers normally avoid repeating that read. Cold BF16 publication also hashes
ordered payloads as they are written and overlaps payload durability work with
later shards; this changes startup work only, not artifact contents or runtime
H2D behavior.

#### Registered direct H2D

Registration is attempted automatically only for an eligible warm HWR hit when
the existing `pin_cpu_memory` policy is enabled. A successful path is:

```text
shared read-only HWR mmap -> existing rotating HBM block buffer -> GPU kernel
```

It removes the recurrent mmap-to-private-staging CPU copy and does not allocate
the two host staging slots. It does not reduce the H2D payload or make GPU
kernels access host memory directly.

CUDA requires read-only host-registration capability for the immutable HWR
mapping. Unsupported capability, a positive registration limit smaller than
the complete page-aligned mapping, or a safely rolled-back registration error
falls back to bounded staging. Programmatic compatibility configurations can
set `pin_cpu_memory=False` to disable registration explicitly.
A successful registration locks the complete mapped range in host memory for
that worker's lifetime, so use
`--dlo-host-registration-limit-gib` when an operator-enforced ceiling is
required.

Shutdown drains H2D work, releases hook/source references, unregisters every
range, and only then closes the HWR lease. AllGather and direct checkpoint mmap
retain their existing transfer paths.

When HWR is disabled, DLO is disabled, or the DiT uses `allgather`, the loader
does not resolve HWR sources or construct its store.

## Declarative topology

Models may declare an `OffloadPlan` instead of embedding offload logic:

```python
from vllm_omni.diffusion.offloader import OffloadPlan


class MyPipeline(nn.Module):
    _dit_modules = ["transformer"]
    _encoder_modules = ["prompt_model"]
    _offload_plan = OffloadPlan(
        block_attrs={"transformer": ("blocks",)},
        offload_submodules={"context_encoder": "layers"},
        encoder_component_types={"prompt_model": "text_encoder"},
        encoder_block_attrs={"prompt_model": ("encoder.layers",)},
        encoder_dlo_weight_replication=frozenset({"prompt_model"}),
        on_demand_component_paths=frozenset({"prompt_model"}),
    )
```

`encoder_component_types` maps arbitrary encoder paths to the public
`text_encoder` selector. `encoder_dlo_weight_replication` is the explicit
safety declaration for reusing the DiT DLO group. On-demand encoder non-block
state is loaded and offloaded by its pipeline phase.

When no plan exists, DiT discovery falls back to
`_layerwise_offload_blocks_attrs` and then heuristic attribute lookup;
undeclared auxiliary components remain resident.

## Data-parallel concurrency

With `data_parallel_size > 1` and AllGather enabled, the scheduler can process
up to `dp_size` requests per denoising step. Concurrent requests must resolve
to the same denoise schedule, so every request must provide the same explicit
`num_inference_steps`. Pipeline defaults are rejected because different request
modes can resolve the same `None` value to different schedules.

## Limitations

- Direct checkpoint mmap currently requires TP1. TP greater than one is
  outside the Phase A shared-mmap support scope and falls back before model
  mutation to the ordinary TP-aware loader. DLO can stream that runtime layout,
  and eligible DiT rank-local configurations may use HWR final-layout artifacts,
  but direct checkpoint mmap provides no shared-mmap host-memory guarantee for
  TP greater than one.
- HSDP with `allgather` on any selected component is rejected to avoid double
  sharding. HSDP with rank-local transfers has limited end-to-end validation.
- With data parallelism, `allgather` cannot be combined with TeaCache or
  Cache-DiT on any selected component. Prompt-embedding cache is also
  incompatible when the text encoder uses `allgather`. Rank-local cache hits
  could otherwise make ranks enter different weight collectives.
- Per-tensor online FP8 linears use the ordinary loader and can run with either
  DiT transfer path. With DiT `allgather`, every rank temporarily materializes
  the complete FP8 model in host memory before DLO retains only its shard.
  Other online quantization methods require rank-local transfer for the
  affected component until their runtime layouts are validated.
- Resident leading layers require DiT `rank-local` transfer and a model
  `OffloadPlan` that declares eligible `resident_dit_paths`.
- DP concurrency requires an explicit, identical inference-step count.

Sharing quantized or otherwise unvalidated transformed layouts through a
normalized HWR producer is a follow-up design in
[RFC #6195](https://github.com/vllm-project/vllm-omni/issues/6195), not part of
the current BF16 final-layout path.

See the [Cosmos3 DistOffload recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/cosmos3/Cosmos3-DistOffload.md)
for an end-to-end example.
