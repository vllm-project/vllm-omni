# MiniMax H3 disaggregated encoder stage

This opt-in topology runs the Qwen3-VL text encoder, video VAE encoder, and
audio WVAE encoder together in a vLLM stage. It sends their conditioning
payload over NIXL to an encoder-free diffusion stage. The standard MiniMax H3 recipes
remain single-stage and continue to run all components inside the diffusion
pipeline.

## Full encoder payload

Both the standard and Turbo builtin deployments use `NixlConnector` with the
UCX backend for the stage 0 → stage 1 edge. Stage 0 is the sender and stage 1
is the receiver. The producer hook `encoder2diffusion_full_payload` validates
the complete conditioning and packages it under `encoder_output`; the diffusion
stage declares `stage_input_payload_keys: [encoder_output]`.

This is not a text-only handoff. The payload includes:

- `hidden_states.output`: BF16 text embeddings.
- `meta.token_role_ids`: INT64 token roles aligned with the text embeddings.
- `embed.embedding`: FP32 visual conditioning latents when present.
- `embed.speech_feat`: FP32 audio conditioning latents when present.
- `kv_metadata.minimax_h3_encoder_layout`: packed layout metadata describing
  the task, dimensions, condition shapes/lengths, keyframes, and reference order.

Text-only requests still carry the full encoder schema, with empty optional
media tensors. The H3 layout field is model conditioning metadata, **not**
native paged-KV transfer. Stage 1 does not request native KV-cache reception.
All participating encoder ranks complete their component collectives before
the producer sends. The diffusion group leader receives the payload and the
generic stage-payload path distributes it to the participating diffusion ranks.
After a successful connector send, the orchestrator carries no duplicate inline
conditioning; original image/video/audio inputs and temporary encoder media
buffers are removed from the diffusion prompt. Stage 1 consumes the received
`encoder_output` without re-encoding text or media.

## Start the server

Choose the topology explicitly and load its deployment defaults:

```bash
vllm-omni serve MiniMaxAI/MiniMax-H3 \
  --omni \
  --deploy-config vllm_omni/deploy/minimax_h3_disaggregated.yaml
```

The default deployment assigns stage 0 to GPUs 0-1 with tensor parallel size
2 and `max_num_seqs: 1`. Stage 1 uses GPUs 2-5 with diffusion tensor parallel
size 1, Ulysses degree 4, and VAE patch parallel size 4. Adjust the
`devices`, `tensor_parallel_size`, and stage 1 `parallel_config` values in a
deployment override for the available hardware. Diffusion quantization,
layerwise offload, distributed layerwise offload, VAE parallelism, and USP
settings use the same stage 1 options documented in [MiniMax-H3.md](MiniMax-H3.md).

Stage 0's `tensor_parallel_size` defines one shared rank group for all three
encoder roles; it does not choose each role's execution strategy. The required
`hf_overrides.minimax_h3_encoder_components` mapping declares those strategies
explicitly: `text_encoder` uses tensor parallelism, `video_vae` uses patch
parallelism, and `audio_vae` runs on the group leader. All three role entries
are mandatory. They share the stage's `max_num_seqs` scheduler and batch limit;
the configuration does not create independent role schedulers or world sizes.

For a six-GPU TP2 encoder → TP4 diffusion topology, retain the builtin
deployment and override only stage 1's TP and Ulysses degrees. VAE patch
parallelism stays at 4. `--stage-overrides` keeps parallelism scoped to its
owning stage rather than broadcasting an override to both:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
vllm-omni serve MiniMaxAI/MiniMax-H3 \
  --omni \
  --deploy-config vllm_omni/deploy/minimax_h3_disaggregated.yaml \
  --stage-overrides '{"1":{"tensor_parallel_size":4,"ulysses_degree":1}}'
```

The same override is supported with the Turbo builtin deployment. Device IDs
in the deployment are relative to `CUDA_VISIBLE_DEVICES`; choose visible
devices and per-stage `devices` overrides for the available hardware. These
commands describe configuration, not a claim of hardware validation for this
integrated encoder implementation.

For memory-constrained deployments, start from the CPU-offload or distributed
layerwise-offload profiles in [MiniMax-H3.md](MiniMax-H3.md). Apply memory and
quantization options only to Stage 1 with `--stage-overrides`; retain the
encoder's BF16 configuration and the video/audio VAEs' FP32 precision. Select
one offload strategy per deployment:

```bash
# Stage 1 full-topology model offload, including MiniMax-H3's VAEs. Keep the
# compatibility alias because the compact selector covers only DiT/text encoder.
--stage-overrides '{"1":{"enable_cpu_offload":true}}'

# Stage 1 distributed layerwise offload. Tune resident layers for available RAM.
--stage-overrides '{"1":{"enable_distributed_layerwise_offload":true,"dlo_use_allgather":false,"dlo_resident_layers":20}}'

# Stage 1 online FP8 quantization of the DiT only.
--stage-overrides '{"1":{"diffusion_quantization_config":"{\"transformer\":{\"method\":\"fp8\"}}"}}'
```

The Stage 1 VAE patch-parallel options remain independent of offload and
quantization. See [MiniMax-H3.md](MiniMax-H3.md) for memory requirements and
hardware-qualified profiles before combining these options.

Stage 1 sets `model_loaded.text_encoder: false` and
`model_loaded.vae_encoder: false`. It loads only the VAE decoders and skips
tokenizer, processor, and text-encoder downloads. It requires complete text and
media conditioning from Stage 0, with no fallback to local encoding.

This H3 topology explicitly keeps its single-replica
diffusion stage inline, avoiding serialization of decoded video through a
subprocess. The builtin placement describes a single-host deployment. NIXL's
transport design can support remote endpoints, but **cross-node H3 execution
has not been validated** here; these examples are not a tested multi-node
launch procedure. CPU wiring and payload-contract tests do not establish native
NIXL, GPU, Turbo-weight, or cross-node execution success.

## Optional SharedMemory configuration

For same-host use without NIXL, both builtin deployment files also declare
`shared_memory_connector` (`SharedMemoryConnector`). In a local copy of the
chosen deployment, change **both** edge references:

```yaml
# Merge these fields into the existing stages; retain their other settings.
stages:
  - stage_id: 0
    output_connectors:
      to_stage_1: shared_memory_connector
  - stage_id: 1
    input_connectors:
      from_stage_0: shared_memory_connector
```

Pass that complete deployment through `--deploy-config`. Do not change only
one end of the edge. The encoder roles, `model_loaded` flags, hooks, and full
`encoder_output` schema stay unchanged. The processor also retains support for
an inline full-conditioning handoff when supplied; stage 1 still does not fall
back to local encoders. SharedMemory requires same-host shared-memory access
and is not a cross-node transport.

The `/v1/videos` request schema and `extra_params.task` values (`t2va`,
`fl2va`, and `ref2va`) are unchanged from the single-stage recipe.

## Turbo LoRA

The deploy config below carries the four-step 768p contract -- five sigma
points, `flow_shift=6`, `audio_flow_shift=3` -- so requests that omit sampling
controls do not inherit the 50-step base schedule:

```bash
vllm-omni serve MiniMaxAI/MiniMax-H3 \
  --omni \
  --lora-path /path/to/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors \
  --deploy-config vllm_omni/deploy/minimax_h3_disaggregated_turbo.yaml
```

`--lora-path` must name one artifact, or a directory holding exactly one; the
Turbo repository holds several and a directory of them is rejected as
ambiguous. Serving a different artifact means overriding those defaults with
that artifact's own contract -- see the
[Turbo LoRA table](MiniMax-H3.md#turbo-lora) for every published row.

The base deployment intentionally retains 50 inference steps for non-LoRA
quality. The `fl2v` artifacts serve T2VA and FL2VA; the `ref2v` artifacts need
a `--task-type ref2va` server. Standard and Turbo configuration coverage does
not imply that Turbo weights have been exercised on this integrated topology.
