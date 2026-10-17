# Matrix-Game 4.0 in vLLM-Omni

A native geometry-to-video integration for **one CUDA GPU** at 864×480. It has
no SM120-only model layers or architecture gate. The current qualification
device is one **RTX PRO 5000 72GB Blackwell**; other GPUs, including the RTX 5090
deployment target, require their own capacity, quality and speed measurements.

This is an experimental Python chunk API. It is not a released HTTP game server.
The game engine supplies low-poly RGB frames; no GPT-6 or other coding-agent API
is required while playing. A coding agent can help author the external game.

See the [deployment recipe](../../../../recipes/Skywork/Matrix-Game-4.0.md)
for the recommended deployment configuration, pinned resources, native game
input generation, request configuration and measured throughput. Omni model
inference does not depend on the external Matrix-Game package. Interactive
gameplay still needs an input producer: the upstream C++ game uses OpenGL/EGL
to supply low-poly RGB frames, while the application handles controls and media.

## Framework reuse

| Component | Implementation |
| --- | --- |
| DiT weights and modules | Existing `WanTransformer3DModel`, with strict native-name mapping |
| Attention | Omni `Attention`; `TORCH_SDPA` reference path, opt-in `CUDNN_ATTN` / `FLASH_ATTN` |
| Projection and FFN parameters | Existing vLLM parallel Linear layers and Wan feed-forward blocks |
| Text encoder | Shared Wan checkpoint loader and Transformers `UMT5EncoderModel`; also used by ABot-World |
| Repeated captions | Runner-owned prompt-embedding cache with a configurable bounded LRU |
| Request execution | Existing `DiffusionEngine` and `DiffusionRequestBatch` |
| Video presentation | Typed `DiffusionMediaOutput`; runner-owned float-to-uint8 conversion and transport |
| Matrix-specific behavior | 148-channel conditioning, incremental flow sampling, sink and temporal-cache policy |
| Codec | Shared Wan TAEW2 encoder/decoder; explicit encoder/decoder state per session |
| Graph replay | Platform-selected vLLM graph wrapper, with session-owned stable inputs and codec history |
| Experimental FP8 | Existing `quantization_config="fp8"`, online weight loading and vLLM kernel selection |

Reusing a component does not automatically qualify every backend or optimization
it offers. Multi-GPU execution, paged KV and concurrent sessions are not enabled.

FP8 uses the existing `DiffusionFp8Config` and vLLM online linear method through
the standard `quantization_config` entry point. vLLM owns weight conversion,
activation scaling and kernel selection. No Matrix-specific quantizer, scale
floor, partitioned linear method or GPU architecture check is added. The packed
QKV projection follows the standard backend's scaling and GEMM policy. Optional
`ignored_layers` names use Wan runtime prefixes such as `blocks.0.attn1.to_qkv`.
Only the plain online FP8 config is qualified here; serialized checkpoints,
per-component maps and other methods are rejected before loading.

The remaining model-specific normalization, RoPE and session code preserves the
released checkpoint's rounding and causal state. Existing per-head fused RoPE
and generic diffusion caching have different semantics and are not substitutes.

## Quality contract

Reference quality is a gate, not a best-effort target. Compare the same checkpoint,
geometry frames, caption, seed, timestep grid, codec and precision. BF16 reference
comparisons must pass before enabling lossy quantization or a different attention
backend. Check both the first chunk and generation beyond the 21-latent window.

The correctness path keeps three separate Q/K/V GEMMs, the original BF16 RMSNorm
rounding, the shared Conv3d convolution path, unfused adaptive normalization and
chronological KV order. Packed QKV GEMM and convolution-to-GEMM produced measured
numerical differences on SM120; those optimizations cannot be presumed lossless.

A 241-frame castle comparison through the actual Omni engine matched all decoded
RGB bytes exactly at one step with TAEHV. This does not establish parity for a
full Wan VAE, other attention backends, quantization, hardware or arbitrary long
sessions. All six 121-frame scene/step pairs and three 1,201-frame two-step gameplay clips
also matched exactly. These correctness measurements use BF16 and do not
qualify the recommended FP8 deployment configuration.

## Acceleration switches

Graph replay defaults to `False` and `quantization_config` defaults to `None`.
Keep `enforce_eager=True` for the outer engine; the model controls capture of its
stateful, fixed-shape operations.

```python
model_config={"scene": "castle", "cuda_graph": True},
quantization_config="fp8",
```

`cuda_graph` captures the steady DiT window and continued TAEW2 chunks using the
existing platform graph wrapper. Online FP8 quantizes supported block linears;
this acceleration path also compiles normalization/residual chains with PyTorch
and uses a ring history buffer. Embeddings, output head and codec remain BF16.
`model_config.fp8` is rejected; quantization has one configuration entry point.

The FP8 path changes scaling and GEMM policy compared with the former custom
adapter. It is **not pixel-identical** to BF16 or the reference optimized FP8
runtime. Retain the default BF16 path for exact reference reproduction. The
recipe documents the recommended configuration and current measured throughput.
A bounded 481-frame Racing run with standard online FP8 measured 76.37 generated
FPS on one PRO 5000 at 864×480, one step, graph replay and SDPA. This is not a
visual-quality parity claim or an RTX 5090 measurement.

## Dynamic Racing captions

Set `model_config.prompt_mode="dynamic"` for Racing and pass each input chunk's
engine-generated caption in `prompt["prompt"]`. The upstream game derives this
text from visible objects and driving state; no language-model or coding-agent
call is needed during play. Castle and Ancient Egypt retain fixed captions.

Enable the existing `enable_prompt_embed_cache=True` and
`prompt_embed_cache_size=8` configuration to reuse repeated text. Dynamic mode
keeps the shared BF16 UMT5 encoder resident; `cuda_graph=True` also captures its
fixed 512-token encoding. New captions refresh cross-attention KV in place,
preserving the DiT graph addresses, temporal KV, sink, random generators and
codec history. Empty request text retains the session's current caption; a new
session starts from the configured scene caption. Captions are bounded to 8192
characters and tokenized with the released 512-token truncation policy.

Send one caption with each opening/four-frame chunk. For a complete clip request,
one caption applies to the entire clip. Rejected ownership, frame-offset and
step-count requests leave the active session unchanged. Fixed mode remains the
default and releases T5 after startup. Dynamic mode costs additional memory and
encoding time; fixed-caption throughput does not describe changing-caption play.

The existing benchmark accepts `--captions captions.jsonl` for either native or
reference execution. Each row must contain zero-based `frame_start`, inclusive
`frame_end` and `prompt`, covering frame 0, then frames 1–4, 5–8, and so on.
Use actual captions aligned with the same geometry recording.
