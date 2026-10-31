# Adding an RL Rollout Model

This guide explains how to add a model to the vLLM-Omni rollout side of an RL
training workflow. It covers model-specific parameter scopes, live weight
reload, stage routing, and validation. It does not define an RL algorithm or a
new weight-transfer wire format.

The runtime contract is specified in the [weight-transfer design
document](../../design/feature/weight_transfer.md). The user-facing
configuration and HTTP API are documented in the [weight-transfer user
guide](../../user_guide/weight_transfer.md).

## Ownership boundaries

Keep the implementation split between the rollout runtime and the training
framework:

```text
Training framework                         vLLM-Omni rollout
-------------------                        ------------------
trainable parameter scope                  pipeline/model loading contract
canonical names and name remapping         supports_weight_reload validation
FSDP/TP materialization                    upstream WTE receiver lifecycle
manifest and version bookkeeping           stage routing and worker errors
reward and trajectory collection           inference and model application
native WTE sender and payload encoding     upstream layerwise reload
```

The rollout side receives the payload produced by an upstream vLLM trainer WTE.
Do not add a model-specific IPC format, manually construct CUDA IPC handles in
the model pipeline, or make the worker know the trainer's topology.

## Integration path

For a new model, use this sequence:

```text
model pipeline
  -> validate the upstream reload contract
  -> match trainer parameter names to the model's existing loaders
  -> enable the configured stage WTE
  -> test real parameter application
  -> run the trainer-to-rollout RL E2E
```

A model that already satisfies the loading contract can use the common WTE
path without an additional adapter or helper class. Keep any necessary
model-specific loading changes in that model's existing implementation.

## 1. Define the training boundary

Before changing Omni code, write down the exact parameter and output boundary
for the recipe:

- Which components are trainable: for example, `transformer.*`, a shared
  `model.*` backbone, an AR head, or a diffusion output head.
- Which components remain frozen: commonly the VAE, text encoder, vision
  encoder, and preprocessing modules.
- Which canonical names the trainer will publish.
- Whether AR and diffusion stages share parameters or only share a checkpoint.
- Which rollout data the trainer needs: tokens and token log probabilities for
  AR, or images, latents, timesteps, and denoising log probabilities for
  diffusion.

The trainer owns parameter selection, FSDP gathering, dtype conversion, name
remapping, manifest validation, and IPC payload creation. Selection should
preserve the source iterable's laziness without materializing or copying the
tensors.

## 2. Match canonical names to the existing model loader

Inspect the model's existing `load_weights` implementation before adding code.
For diffusion pipelines, model-specific behavior belongs under:

```text
vllm_omni/diffusion/models/<model_name>/
```

AR models retain their existing implementation under
`vllm_omni/model_executor/models/` and reuse the upstream worker lifecycle.
Shared diffusion reload preparation belongs in
`vllm_omni/diffusion/model_loader/weight_reload.py`.

For example, a QwenImage transformer-only trainer publishes names such as
`transformer.img_in.weight`; the pipeline's `AutoWeightsLoader` dispatches them
to the transformer. HunyuanImage3 uses `model.*` for its shared backbone and
recipe-specific head names such as `lm_head.*` or `patch_embed.*`. Its
`transformer` attribute aliases `model`, so do not export both namespaces as
separate copies of the same backbone.

Keep recipe-specific selection and source name remapping in the training
framework. Add a target-side loader mapping only when the model's loading
contract requires one, and reuse it for initial loading and reload. Do not add
model-name conditionals to the common worker or wrap the WTE lifecycle just to
select parameters. Name matching alone does not prove reload support.

## 3. Validate the pipeline reload contract

Before setting `supports_weight_reload`, verify the complete loading path. The
shared integration point is
[`diffusion/model_loader/weight_reload.py`](../../../vllm_omni/diffusion/model_loader/weight_reload.py).

The pipeline must satisfy all of the following:

1. Its loading layout and parameter metadata can be captured after pipeline
   construction but before external checkpoint loading or postprocessing
   changes the parameter structure.
2. `load_weights` routes incoming tensors through the pipeline's parameter
   loaders, including fused, renamed, and sharded parameters.
3. The upstream layerwise reload lifecycle can run the required post-load
   transformations.
4. Repeated updates work with multiple chunk boundaries.
5. The original parameter storage, device, dtype, and untouched buffers remain
   valid after an update.
6. Generation is quiescent from `start_weight_update` through
   `finish_weight_update`.

Do not use direct `.data.copy_()` for a reload path. It bypasses the metadata
and completion tracking used by upstream WTE. A capability flag alone is not
evidence that a model, quantization mode, offload mode, or cache configuration
is supported.

Start with an unquantized, non-offloaded configuration. Validate quantization,
HSDP, inference offload, and cache acceleration separately before enabling any
of those combinations.

## 4. Reuse the native WTE lifecycle

The rollout worker exposes the upstream four-phase lifecycle:

```python
await engine.init_weight_transfer_engine(init_info)
await engine.start_weight_update()
for update_info in trainer_encoded_chunks:
    await engine.update_weights(update_info)
await engine.finish_weight_update()
```

The transport can be RPC or HTTP control. With IPC, the actual tensor storage
is still referenced through CUDA IPC handles on the same node; HTTP carries the
control request and backend metadata. Model-specific loading code must not
wrap or replace these calls.

The trainer must keep every source buffer alive until the receiver acknowledges
the corresponding update. This matters for packed senders that reuse one GPU
buffer: the sender must send chunk A and wait before overwriting it with chunk
B.

## 5. Configure stage routing

Enable WTE only on the stage that receives updates:

```yaml
stages:
  - stage_id: 0
    # AR stage configuration, if it is trained in this recipe.
  - stage_id: 1
    weight_transfer_config:
      backend: ipc
```

For a pipeline-wide configuration, use the top-level
`weight_transfer_config`. `AsyncOmni` routes lifecycle calls to configured
stages and propagates worker failures.

For AR+DiT models, explicitly decide whether the recipe updates:

- AR-only parameters;
- diffusion-only parameters;
- a shared backbone with an AR head; or
- a shared backbone with a diffusion head.

The trainer manifest and target loader must use the same canonical names.
Stage selection does not by itself synchronize two workers that share a logical
backbone. That requires a separate multi-stage validation covering routing,
replica consistency, cache invalidation, and request quiescence.

## 6. Add rollout outputs required by the recipe

Weight reload and rollout data are separate contracts. Add output conversion
only when the training algorithm needs information not already exposed by the
rollout backend:

- AR recipes generally need response token IDs, response boundaries, and token
  log probabilities. The reward may still be computed from a later image.
- Diffusion recipes may need the generated image, conditioning, selected
  denoising latents, timesteps, sigmas, and SDE log probabilities.

Keep reward computation, advantage calculation, replay, gradient computation,
and optimizer steps in the training framework. Omni's responsibility is to
produce valid rollout results and apply the next published model snapshot.

## 7. Test in increasing scope

### CPU contract tests

Use a small mock only for deterministic interface behavior:

- trainable and frozen parameter scopes;
- lazy selection and canonical names;
- valid lifecycle order;
- duplicate, incomplete, shape, and dtype-invalid updates;
- generation blocked during an active update;
- consecutive publication versions.

A CPU mock does not prove CUDA IPC, upstream layerwise reload, stage
coordination, cache behavior, or real image quality.

### GPU reload regression

Use a tiny checkpoint and the real pipeline to verify:

1. The initial generation succeeds.
2. A trainer-shaped native WTE payload changes a selected live parameter.
3. Frozen parameters and buffers remain unchanged.
4. The pipeline generates successfully after the update.
5. Restoring the original snapshot works across a second publication.

Check the live tensor values directly; a tiny random checkpoint may not produce
a visibly different image after a transformer-only perturbation.

### RL E2E

Run the actual training framework with the selected recipe:

```text
rollout -> reward/trajectory collection -> replay and backward
        -> optimizer step -> native WTE publication
        -> rollout reload -> next rollout
```

Record the Omni and trainer revisions, trainable manifest, publication/sample
versions, selected parameter hashes, frozen-parameter checks, and the number of
transferred chunks. Test both native RPC + IPC and HTTP control + IPC when both
control paths are supported.

## Pull request checklist

- [ ] The parameter scope and frozen components are documented.
- [ ] Canonical names and any trainer-side name remap are documented.
- [ ] Model-specific changes stay in the existing model loading path.
- [ ] No private IPC payload or model-name branch was added to the common worker.
- [ ] `supports_weight_reload` is enabled only after the full loading contract
      passes.
- [ ] CPU lifecycle and selector tests pass.
- [ ] A real GPU reload test passes.
- [ ] The RL E2E records versions, manifests, hashes, and frozen-state checks.
- [ ] Unsupported quantization, offload, cache, and multi-stage combinations
      are stated explicitly.

## Related documents

- [Adding a diffusion model](adding_diffusion_model.md)
- [Weight-transfer design](../../design/feature/weight_transfer.md)
- [Weight-transfer user guide](../../user_guide/weight_transfer.md)
