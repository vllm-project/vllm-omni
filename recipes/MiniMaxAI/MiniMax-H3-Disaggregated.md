# MiniMax H3 disaggregated text encoder

This opt-in topology runs the Qwen3-VL text encoder as a vLLM stage and sends
its hidden states and token-role metadata to an encoder-free diffusion stage.
The standard MiniMax H3 recipes remain single-stage and continue to load the
text encoder inside the diffusion pipeline.

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

For example, this five-GPU topology assigns one GPU to the encoder and four
to the diffusion stage. `--stage-overrides` keeps placement and parallelism
scoped to the owning stage rather than broadcasting an override to both:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
vllm-omni serve MiniMaxAI/MiniMax-H3 \
  --omni \
  --deploy-config vllm_omni/deploy/minimax_h3_disaggregated.yaml \
  --stage-overrides '{"0":{"devices":"0","tensor_parallel_size":1},"1":{"devices":"1,2,3,4","tensor_parallel_size":1,"ulysses_degree":4,"vae_patch_parallel_size":4}}'
```

For memory-constrained deployments, start from the CPU-offload or distributed
layerwise-offload profiles in [MiniMax-H3.md](MiniMax-H3.md). Apply memory and
quantization options only to Stage 1 with `--stage-overrides`; retain the
encoder's BF16 configuration and the video/audio VAEs' FP32 precision. Select
one offload strategy per deployment:

```bash
# Stage 1 model-level CPU offload.
--stage-overrides '{"1":{"enable_cpu_offload":true}}'

# Stage 1 distributed layerwise offload. Tune resident layers for available RAM.
--stage-overrides '{"1":{"enable_distributed_layerwise_offload":true,"dlo_use_allgather":false,"dlo_resident_layers":20}}'

# Stage 1 online FP8 quantization of the DiT only.
--stage-overrides '{"1":{"diffusion_quantization_config":"{\"transformer\":{\"method\":\"fp8\"}}"}}'
```

The Stage 1 VAE patch-parallel options remain independent of offload and
quantization. See [MiniMax-H3.md](MiniMax-H3.md) for memory requirements and
hardware-qualified profiles before combining these options.

Stage 1 sets `model_loaded.text_encoder: false`; it must not load or download
text-encoder weights. This H3 topology explicitly keeps its single-replica
diffusion stage inline, avoiding serialization of decoded video through a
subprocess. It expects both stages to run in one deployment; cross-node payload
transport is outside this configuration.

The `/v1/videos` request schema and `extra_params.task` values (`t2va`,
`fl2va`, and `ref2va`) are unchanged from the single-stage recipe.

## Turbo LoRA

MiniMax-H3 Turbo uses five sigma points, `flow_shift=6`, and
`audio_flow_shift=3`. Start Turbo deployments with the dedicated defaults so
requests that omit sampling controls do not inherit the 50-step base schedule:

```bash
vllm-omni serve MiniMaxAI/MiniMax-H3 \
  --omni \
  --lora-path /path/to/MiniMax-H3-Turbo \
  --deploy-config vllm_omni/deploy/minimax_h3_disaggregated_turbo.yaml
```

The base deployment intentionally retains 50 inference steps for non-LoRA
quality. Turbo LoRA supports T2VA and FL2VA requests, not Ref2VA.
