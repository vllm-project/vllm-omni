# Host Weight Runtime

The Host Weight Runtime is a loader-facing feature for reusing immutable,
runtime-ready model weights across workers on one host. It lets a model
integration replace repeated checkpoint loading and transformation with an
exact lookup of a previously published host representation, while keeping the
canonical model source authoritative.

This feature is shared infrastructure. Diffusion, autoregressive, speech, and
future model integrations may depend on the top-level
`vllm_omni.host_weight_runtime` package without depending on one another's
loader or execution stack. The internal component and filesystem contracts are
specified in the [Host Weight Runtime module design](../module/host_weight_runtime.md).

## Status

The implementation provides contracts and a CPU local-filesystem store. The
diffusion consumer additionally defines typed,
representation-independent final-layout identity/restoration mechanics plus a
concrete BF16-with-preserved-FP32 policy for MiniMax H3 and
`black-forest-labs/FLUX.2-klein-4B`. The opt-in no-AllGather DLO integration
selects, publishes, restores, and transfers these artifacts.

V1 includes:

- exact runtime-weight identity and immutable manifests;
- coordinated local lookup and one-producer publication;
- descriptor-backed safetensors mmap leases;
- preferred and required resolution policy;
- explicit, separately reported post-load publication;
- validation, deny, quarantine, cleanup, and capacity controls; and
- typed reports for every terminal resolution outcome.

V1 does not include:

- default-on activation or consumers outside no-AllGather DLO;
- online FP8, quantized, merged-adaptation, or additional model producers;
- HWR interaction with DLO AllGather;
- a remote artifact provider or cross-node coordination;
- automatic eviction; or
- a change to DLO collective or execution behavior.

## Motivation and use cases

Model loading can create the same final host representation repeatedly. This
is especially expensive when loading performs checkpoint decoding, tensor
renaming, TP slicing, quantization, packing, or scale construction before GPU
transfer. Independent workers may then retain private CPU copies of identical
runtime weights.

The Host Weight Runtime is useful when:

- multiple same-node workers request semantically identical final weights;
- constructing those weights is materially more expensive than validating and
  mapping a local artifact;
- the runtime layout can be identified exactly and reproduced deterministically;
- the workers can share immutable file-backed pages through the OS page cache;
  and
- canonical loading remains available when policy permits fallback.

Typical consumers include same-node diffusion replicas, DLO host-weight
sources, and future transformed or quantized model loaders. Request routing and
replica orchestration remain outside this feature.

This feature is not zero-copy GPU execution. A lease exposes stable CPU tensor
views and mapped ranges. A separate transport still decides whether to use
registered mmap, private pinned staging, synchronous copies, or asynchronous
H2D transfer.

## Resolution behavior

The loader resolves the immutable canonical source and computes the exact
requested identity before asking the runtime for a representation.

```mermaid
flowchart TD
    L["Loader resolves canonical source"] --> I["Compute exact WeightArtifactIdentity"]
    I --> M{"Runtime mode"}
    M -->|"disabled"| C["Canonical loader"]
    M -->|"preferred or required"| A{"Validated local artifact?"}
    A -->|"yes"| R["Return HostWeightLease"]
    A -->|"no"| X["Remote exact artifact (future)"]
    X --> P{"Registered producer available?"}
    P -->|"yes"| B["Build, validate, and publish atomically"]
    B --> R
    P -->|"no or recoverable failure"| F{"Policy permits fallback?"}
    F -->|"preferred"| C
    F -->|"required"| E["Fail startup"]
    P -->|"nonretryable failure"| E
    C --> W{"Post-load publication enabled?"}
    W -->|"yes"| B2["Publish final model through POST_LOAD_ONLY producer"]
    B2 --> CL["Close validated publication lease"]
    CL --> R2["Return separate publication report"]
    R2 --> D
    W -->|"no"| D["Keep canonical model"]
```

The modes are:

| Mode | Observable behavior |
| --- | --- |
| `disabled` | Use canonical loading directly without probing storage, identity, credentials, or topology. |
| `preferred` | Return an exact lease when available; otherwise use canonical fallback only for a miss, invalid cache entry, or typed retryable failure. |
| `required` | Fail when an exact lease cannot be acquired. |

During pre-load resolution, an unsupported capability, semantic identity
collision, producer failure, or publication failure is nonretryable and remains
visible even in preferred mode. A post-load publication failure is likewise
visible in its own report, but cannot revise the canonical-fallback outcome.
Storage policy must never disguise a semantic or configuration error as a cache
miss.

## Loader and restoration sequence

The model integration owns the end-to-end loading transaction. The runtime
does not construct a model or invoke the canonical loader.

```mermaid
sequenceDiagram
    participant L as Loader adapter
    participant R as HostWeightRuntime
    participant S as HostWeightStore
    participant P as WeightProducer
    participant X as WeightRestorer
    participant T as GPU transport

    L->>L: Resolve canonical revision and exact identity
    L->>R: resolve(identity, producer)
    R->>S: lookup(exact identity)
    alt validated hit
        S-->>R: HostWeightLease
        R-->>L: lease and LOCAL_HIT report
        L->>X: plan_restore(model, lease)
        X-->>L: validation-only restore plan
        L->>X: commit() once
    else miss with allowed producer
        R->>S: get_or_build(identity, producer)
        S->>P: produce(store-scoped writer)
        P-->>S: final-layout tensors and metadata
        S-->>R: validated HostWeightLease
        R-->>L: lease and LOCAL_PRODUCTION report
        L->>X: plan_restore(model, lease)
        X-->>L: validation-only restore plan
        L->>X: commit() once
    else policy permits canonical fallback
        R-->>L: CANONICAL_FALLBACK
        L->>L: Run canonical loader
        opt explicit post-load publication enabled
            L->>R: publish_after_load(identity, POST_LOAD_ONLY producer)
            R->>S: get_or_build(identity, producer)
            S-->>R: validated HostWeightLease or typed failure
            R->>R: close publication lease
            R-->>L: separate publication report
        end
    else required or nonretryable failure
        R-->>L: FAILED report
    end
    L->>T: consume final CPU tensors and any lease
    T-->>L: transfer teardown complete
    L->>L: close lease when one was acquired
```

`plan_restore()` must not mutate the model or lease. `commit() -> None` is the
sole one-shot model mutation. If planning fails, canonical fallback may reuse
the untouched model. If commit begins and fails, the partially hydrated model
must be discarded and canonical fallback must construct a fresh model.

`publish_after_load()` is synchronous in V1 and accepts only a
`POST_LOAD_ONLY` producer. Its report is separate from the terminal resolution,
so a publication failure cannot rewrite a successful canonical fallback. A
successful publication closes the store-returned lease inside the runtime and
warms only future startups. It does not restore, rebind, or otherwise mutate the
canonically loaded model serving the current startup.
`allow_local_build` gates producers during pre-load resolution, while
`allow_post_load_publish` independently gates this explicit post-load path.

## Exact representation identity

The store performs exact lookup. It never silently converts one representation
into another or substitutes the first backing that responds.

Identity includes:

- immutable model revision and source fingerprint;
- model component and ownership boundary;
- representation name, dtype, and format metadata;
- final tensor layout and semantic parallel coordinates;
- static adaptation identity; and
- producer implementation, manifest, and restorer schema versions.

The requested representation is selected by the loader. The store chooses
where to obtain that exact representation: validated local artifact, future
remote materialization, or a registered producer.

### Parallelism

Parallel coordinates are included only when they change bytes, shape, layout,
or component ownership.

| Dimension | Identity and sharing rule |
| --- | --- |
| DP | Exclude DP rank for replicated weights so same-node replicas can acquire the same artifact. Include a coordinate only if DP changes weight ownership. |
| TP | Include TP size and rank when each rank owns different tensor slices or packed layouts. Each distinct TP shard is a separate exact artifact. |
| SP | Include SP size and backend when they change the weight layout. Exclude SP rank when all SP ranks consume identical bytes. |
| PP | Encode component ownership or stage-local layout when PP changes which weights belong to the consumer. |
| EP | Encode expert ownership and layout whenever ranks own different expert weights. |

The runtime does not infer these rules from process-group topology. The model
adapter is responsible for constructing a correct identity.

### Quantization and adaptations

Runtime BF16, FP8, packed quantized weights, and future formats are distinct
representations. Conversion is performed by an explicitly versioned producer,
never by cache coercion. A producer must publish the final layout consumed by
its matching restorer.

Dynamic LoRA overlays are not part of a reusable base-weight artifact. A
statically merged adapter is cacheable only as a separate identity containing
the adapter fingerprint and merge semantics.

### Initial diffusion final-layout contract

The shared diffusion contract covers complete final-layout DiT parameters and
persistent buffers. Text encoders, VAEs, non-persistent derived state, and other
pipeline components remain outside the artifact. One explicit representation
policy selects allowed dtypes, tensor roles, physical layout identity, producer
ABI, manifest schema, and restoration schema.

The contract is intentionally separate from loader activation:

- `FinalLayoutRequest` contains typed loader identity/configuration fingerprints,
  TP coordinate, and conservative SP semantics. It has no open metadata bag,
  DP coordinate, SP rank, device identity, DLO transfer mode, registration
  policy, or store path.
- `FinalLayoutArtifactSpec` binds one `WeightRepresentation` and runtime-layout
  name to explicit producer/restorer schemas and a canonical, versioned
  implementation ABI descriptor. Compatibility never depends on reflective
  source inspection.
- `PreparedWeightSource` snapshots immutable revisions or exact local file
  content plus a typed checkpoint-adapter identity before ordinary
  materialization. Source replacement before or during production fails
  publication. A hash-looking symlink basename is trusted only for an explicit
  Hugging Face Hub source whose repository ID and
  `models--.../snapshots/<revision> -> blobs/<hash>` topology validate; every
  local or otherwise unverified symlink target is content-hashed.
- the tensor ownership digest records exact runtime names, kinds, shapes,
  semantic roles, dtypes, and strides from a CPU or meta model skeleton;
- `FinalLayoutTensorRestorer` accepts only an exact lease identity, validates
  complete policy-defined coverage without mutation, and returns a one-shot
  commit plan;
- each model declares one dtype-neutral `FinalLayoutModelContract` with an
  explicit implementation version and a post-commit validator; and
- `FinalLayoutBF16Producer` accepts only the matching identity context and a
  finalized CPU model. It is `POST_LOAD_ONLY` and `SINGLE_PROCESS` per exact TP
  coordinate. Its BF16 policy preserves model-declared FP32 parameters and
  buffers, revalidates MiniMax H3 mixed-precision invariants, and revalidates
  FLUX.2-klein's two block stacks, packed QKV mapping, and BF16 base layout.

Other representations reuse source identity, typed parallel identity, tensor
ownership, and exact restoration only when their policy proves those semantics.
For example, runtime FP8 needs a separate policy/producer for generated scales,
quantization metadata, and Cutlass physical layouts; it is not enabled by
changing a dtype string on the BF16 producer.

This stage makes no startup, sharing, or DLO performance claim. A following
consumer PR owns disabled/preferred/required precedence, mixed-component loader
transactions, warm-hit restoration, and transactional lease handoff. A TP2
prewarm deployment will require a matching TP2 producer cohort to populate both
TP-coordinate identities even though the store coordinates each artifact
independently.

## Host sharing and GPU transport

Every process receives its own virtual mappings and `HostWeightLease`, but
workers mapping the same local artifact can share physical file-backed pages
through the OS page cache. This avoids independent private tensor allocations;
it does not guarantee that process PSS reports exactly one model copy.

The ownership boundary is:

- the store owns immutable artifact files, publication, and lifecycle;
- the kernel owns page-cache residency and placement;
- the lease owns process-local mappings and the shared artifact lock; and
- transport owns page registration, page locking, private staging, H2D copies,
  streams, device buffers, and lease release ordering.

DLO AllGather and no-AllGather remain transport and execution choices. A DLO
integration may consume a lease, but the Host Weight Runtime does not choose a
parallel collective or orchestrate DP requests. See the
[DLO feature design](offloader/distributed_layerwise_offload.md).

## Locality and NUMA policy

The V1 backend is node-local. It accepts an allowlist of known local disk
filesystems plus tmpfs/ramfs, records the detected kernel filesystem type, and
rejects known remote or unknown filesystems. NFS, CIFS, Lustre, Ceph, and other
cross-node mounts cannot silently satisfy the local backend because their
page-cache and advisory-lock behavior does not provide the required node-local
contract.

A NUMA-specific root creates a separate storage domain but does not by itself
place page-cache pages on that NUMA node. A topology-aware integration must
also pin workers, establish local first touch or prefault, perform transport
registration in the same domain, and collect residency evidence.

Tmpfs artifacts consume host memory and may consume swap. They must be
accounted as memory-backed storage rather than ordinary disk capacity.

## Concurrency and timeouts

One process per exact identity owns a build; other workers wait and then acquire
leases for the published artifact. Publication is invisible until all payloads
and metadata are validated, hashed, fsynced, and atomically renamed.

`coordination_timeout_seconds` bounds filesystem lock acquisition. It does not
cancel synchronous validation, a producer that has already started, or atomic
publication. A hung in-process producer therefore blocks its owning process and
must be handled by external process supervision. Enforceable producer
cancellation requires a future process-isolated producer contract.

Leases keep a shared artifact lock for their lifetime. A forked child may close
its inherited descriptors but cannot unlock the parent process's lease. Cleanup
uses a nonblocking exclusive lock and reports an active lease instead of
removing live mappings.

## Failure and fallback contract

| Condition | Preferred mode | Required mode |
| --- | --- | --- |
| Exact local hit | Return lease | Return lease |
| Local miss | Try later backing or canonical fallback | Try later backing or fail |
| Invalid or denied artifact | Quarantine/rebuild when possible; otherwise canonical fallback | Quarantine/rebuild or fail |
| Retryable lock, domain, or capacity failure | Canonical fallback | Fail |
| Unsupported producer or backend | Fail | Fail |
| Identity collision | Fail | Fail |
| Pre-load producer or atomic publication failure | Fail | Fail |
| Post-load publication failure after canonical fallback | Keep canonical model; report publication failure | Not reached through required resolution |
| Restore planning failure | Canonical fallback may reuse untouched model | Fail |
| Restore commit failure | Discard model; fallback requires a fresh instance | Fail |

Cache failures should normally reduce performance rather than availability,
but only when the typed failure explicitly permits fallback. Canonical source,
authentication, configuration, and semantic errors remain visible.

## Consumer integration requirements

A consumer PR must:

1. Resolve an immutable canonical model source before cache lookup.
2. Construct the exact final representation identity, including relevant
   parallel, quantization, and adaptation semantics.
3. Register a deterministic producer for that identity or support lookup-only
   operation.
4. Provide a validation-only restorer with a one-shot commit.
5. Keep GPU transport outside the producer, store, and restorer contracts.
6. Handle preferred fallback and required failure without reusing a model after
   a failed restore commit.
7. Retain the lease until every transport operation that may access mapped
   memory has completed.
8. Emit and test the terminal resolution report and any separate post-load
   publication report.

Consumer validation must prove:

- output parity with canonical loading;
- exact warm-hit work avoidance;
- correct TP/SP/PP/EP ownership and layout identity;
- correct BF16, FP8, quantized, or adaptation semantics;
- shared-backing evidence when host-memory savings are claimed;
- clean unmapping, unregistration, and artifact-lock release; and
- startup, host-memory, and transport effects for the claimed deployment.

## Rollout

The rollout is intentionally staged:

1. Land the neutral contracts and local filesystem store.
2. Add one model-specific producer/restorer integration with parity evidence.
3. Connect eligible DLO or other transport paths without moving transport
   ownership into the runtime.
4. Add additional final-layout and quantized producers independently.
5. Introduce remote materialization only through the same local lease contract.

The architectural decisions and deferred work are tracked in
[RFC #6414](https://github.com/vllm-project/vllm-omni/issues/6414).
