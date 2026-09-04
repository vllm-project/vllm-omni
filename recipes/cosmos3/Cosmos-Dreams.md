# Cosmos-Dreams (Cosmos3-Interactive)

Cosmos-Dreams is the causal action-conditioned Cosmos3 world model. This port
targets the distilled `interact_8b_tfdcm_chunk4_agibot` checkpoint pinned to
`iter_000001600`: four SDE denoise steps, `[1, 4, 4, ...]` latent chunks, and
per-frame interleaved action/video tokens.

## Artifact contract

Use the two-stage imaginaire4 exporter with a completed
`cosmos_dreams_manifest.example.json`. The manifest must include the DCP content
hash and the real action-normalizer statistics/provenance. Stage 1 validates the
selected EMA/student tensor inventory; Stage 2 writes
`CosmosDreamsPipeline` into `model_index.json`, embeds the causal fields in
`transformer/config.json`, and emits `cosmos_dreams_artifact.json` with tensor
headers and per-shard SHA-256 hashes.

Raw DCP is intentionally not accepted by Stage 2.

Schema-v1 artifacts may also contain the frozen pre-unification YAM datasets
exported by imaginaire4: `abc_yam`, `molmoact2_yam`, and `xdof_yam`. vLLM-Omni
validates their 20D left-arm/right-arm layout, domain 16, and separate legacy
normalizer sources before serving them. The unified `yam_dataset` classes have
different coordinate semantics and are not interchangeable with these legacy
contracts.

From the `imaginaire4` checkout, materialize the checkpoint locally and copy
`packages/cosmos3/cosmos3/scripts/cosmos_dreams_manifest.example.json` to a
working manifest. Compute the hash over the same local `model/` directory that
Stage 1 will load, then replace every template value, including the complete
numeric normalizer arrays:

```bash
python -c 'from pathlib import Path; from cosmos3.scripts.cosmos_dreams_export import sha256_checkpoint; print(sha256_checkpoint(Path("/checkpoints/iter_000001600/model")))'

python -m cosmos3.scripts.export_model \
  --checkpoint-path /checkpoints/iter_000001600/model \
  --config-file /runs/interact_8b_tfdcm_chunk4_agibot/config.yaml \
  --cosmos-dreams-manifest /manifests/cosmos_dreams.json \
  --student-only-checkpoint-metadata \
  -o /exports/cosmos-dreams-hf

python -m cosmos3.scripts.convert_model_to_diffusers \
  --checkpoint-path /exports/cosmos-dreams-hf \
  --cosmos-dreams-manifest /manifests/cosmos_dreams.json \
  -o /exports/cosmos-dreams-diffusers
```

Stage 1 rejects remote DCP paths for causal exports: download the complete tree
first so `checkpoint_hash` can be verified. It compares the selected
`net_ema.*` namespace against the export target before DCP loading, preventing
missing tensors from surviving as random initialization. Stage 2 then compares
the serialized tensor names, shapes, and dtypes against its strict remap
inventory. Keep `cosmos_dreams_artifact.json` with the model directory.

## Deployment

Start from [`cosmos_dreams.yaml`](../../vllm_omni/deploy/cosmos_dreams.yaml).
The default is eager, batch size one, 720×1280, one resident session, and a
96-latent-frame window. Each request resolves an aligned canvas and selects a
matching paged KV pool; the maximum permitted geometry is validated at load.

Checkpoint identity, hash, domain map, and normalizer data are deliberately not
template defaults: they must be read from `transformer/config.json`. Startup
rejects a deploy override that contradicts those embedded artifact fields.

The KV startup allocation is a hard floor. At TP=1/BF16, one 720p frame across
36 layers is about 133 MiB; the configured window, scratch reservation, and
the 512-token text pool are all counted before model
execution. If the manager reports an insufficient memory budget, use a
checkpoint artifact exported with a smaller `window_frames`; runtime overrides
are rejected. The physical window does not cap the logical session at that
length: evicted positions remain logical placeholders while generation rolls
forward. A sliding window changes long-horizon semantics relative to the
unbounded reference and needs a separate quality evaluation. Engine-level
window/sink/reset overrides would make the paged path diverge from the
manifest-driven dense oracle.

## Request modes

- Full rollout: provide all actions, `reset=true`, and preferably
  `close_session=true`. The globally last latent frame is not committed because
  it has no downstream reader.
- Tick session: use `ARDiffusionSession` with an `ARDiffusionOmniTickConsumer`
  and carry AgiBot actions as a `robot_action.v1` control inside the typed
  `ARDiffusionTickRequest`. The first tick covers the singleton prefix and one
  four-frame chunk; later ticks commit one four-frame chunk. Request, event,
  session, and chunk identities are validated against the returned
  `metadata.ar_diffusion` envelope. Reset, close, and disconnect release the
  owning worker through `ARDiffusionWorkerLifecycle`; no inference request is
  submitted solely for cleanup.
- Dense oracle: select the default `DiffusionEngine` with the same checkpoint.
  It maintains dense per-layer history and is the numerical reference for the
  gathered-paged path.

Attention runs one joint softmax over `[real text | committed history |
current]`. The paged path uses the fused `paged_write_attn` operator and carries
the real text K/V in the auxiliary scratch slots; the dense path concatenates the
same three spans. Compile/CUDA graphs, RF/CFG checkpoints, and multi-session
serving remain follow-up work.

See the [offline parity runner](../../examples/offline_inference/cosmos_dreams/README.md)
for jsonl/pickle input and latent-output examples.
