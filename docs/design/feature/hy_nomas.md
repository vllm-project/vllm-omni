# Hy-Nomas integration proposal

> **Status: proposal.** This document records the initial integration scope for
> [#9327](https://github.com/vllm-project/vllm-omni/issues/9327). It does not add
> model support or claim validated inference, performance, or hardware coverage.

## Objective

Establish reproducible offline multi-segment Hy-Nomas generation in vLLM-Omni
before extending it to interactive serving. The intended users are developers
deploying and evaluating controllable video world models.

## Upstream references

- [Hy-Nomas source and inference examples](https://github.com/Tencent-Hunyuan/Hy-Nomas/tree/911cea6458a89c2663513ff7099561ec190dd491)
- [Model weights and model card](https://huggingface.co/tencent/Hy-Nomas)
- [Upstream inference configuration](https://github.com/Tencent-Hunyuan/Hy-Nomas/blob/911cea6458a89c2663513ff7099561ec190dd491/configs/experiment/inference.yaml)

The upstream model generates video autoregressively by segment, with text and
camera conditioning and attention history across segments. Its distilled DiT
produces 32-channel DCAE-V latents decoded with a fine-tuned PixelUMM decoder.
The README also documents a Diffusers integration on a dedicated development
branch. That branch is a reference, not an assumed installed dependency or a
drop-in vLLM-Omni pipeline.

## Initial scope

The first implementation should accept an ordered sequence of structured text
segments in one offline request, preserve history within that request, and
return decoded video with explicit frame-count and frame-rate metadata.

Start with a compatible floating-point distilled DiT and PixelUMM checkpoint
pair. Record the exact text encoder, tokenizer, checkpoint revisions, inference
configuration, and upstream code revision before establishing a baseline.
Do not transfer the upstream low-memory single-GPU claim to this baseline:
that upstream profile uses quantized weights and offload.

Camera controls, persistent online sessions, streaming transport, prompt
enhancement, LoRA loading, quantization, and acceleration are follow-up scope.
Unsupported controls should fail clearly rather than be silently ignored.

## Integration boundaries

Follow the [diffusion model integration contract](../module/diffusion/diffusion_model_integration.md)
and [adding a diffusion model guide](../../contributing/model/adding_diffusion_model.md).

| Component | Proposed responsibility |
| --- | --- |
| Registry and loader | Select the Hy-Nomas pipeline; validate configuration and checkpoint layouts. |
| Model adapter | Implement checkpoint-compatible DiT attention, text conditioning, denoising, and latent conventions. |
| Decoder integration | Load the matching PixelUMM decoder and preserve segment/frame alignment. |
| Request processing | Validate structured segment prompts and dimensions before execution. |
| Runtime | Own admission, execution lifecycle, cancellation, and resource release. |

Use `vllm_omni/diffusion/models/hy_nomas/` for genuine model-specific code and
the existing diffusion registry for selection. Keep request adaptation in the
model integration or `model_extras` boundary. Reuse shared task runners and
document model commands in task documentation or recipes; do not introduce a
model-specific Python example runner.

LingBot-World and ABot-World are references for causal generation and state
lifecycles. HunyuanVideo-1.5 is a reference for video pipeline structure.
None establishes interchangeable weights, attention masks, or decoder state.

## Segment state and memory

For the offline milestone, history belongs to one request and must not leak
into another request. Track segment position, prompt-derived conditioning,
attention history, and any decoder state explicitly. Release them after
completion, cancellation, or failure. A prompt update must invalidate derived
conditioning caches whose inputs changed.

Preserve upstream history semantics in the correctness baseline. Do not silently
replace full history with a sliding window. If bounded history is introduced,
make its policy explicit and validate it separately. Report memory against
segment count; output video FPS is not a generation-throughput measurement.

For later persistent sessions, evaluate the existing
[AR-diffusion capability](../ar_diffusion_pipeline_capability.md) and
[session lifecycle](realtime_ar_diffusion.md). Resolve compatibility between
Hy-Nomas segment lengths/history semantics and runner-managed KV geometry
before advertising that capability. Session identity, eviction, and scheduling
must remain runtime concerns. Account for decoder state and non-KV buffers in
addition to managed KV. Failure recovery must not reuse partially updated state.

## Validation and acceptance

These checks are planned; none is claimed as completed model validation.

- Loading: verify registry selection, configuration validation, checkpoint key
  mapping, and actionable failures for incompatible components.
- Inputs: validate ordered segments, timestamps or latent lengths, dimensions,
  and unsupported controls according to the chosen upstream baseline.
- Single segment: compare finite latents, output shape, frame count, and visual
  quality against upstream with identical inputs and sampling settings.
- Multiple segments: compare at least two consecutive segments, including a
  changed text instruction, and inspect continuity at the boundary.
- Isolation and cleanup: generate unrelated requests sequentially and exercise
  failure/cancellation paths so previous history cannot contaminate the next
  generation or retain request-owned memory.
- Reproducibility: record GPU model/count, driver, CUDA, package versions,
  checkpoint revisions, seed, resolution, segment lengths, denoising steps,
  guidance, attention backend, commands, and actual results.
- Performance: after correctness, measure warm and cold latency, per-segment
  latency, and peak memory against the same upstream workload. Report multiple
  trials without claiming a speedup before measurement.

Before advertising support, add the relevant unit and GPU functionality tests,
document a runnable shared-runner command, and update the supported-model table
only for capabilities verified by those tests.

## Open decisions

1. Which upstream implementation and checkpoint layout should be the initial
   compatibility target: the native inference path or its Diffusers conversion?
2. Can the existing AR-diffusion capability represent Hy-Nomas history without
   changing its semantics, especially for variable segment lengths?
3. Which dependency and attention backend combination is compatible with the
   project's supported environment?
4. What hardware and numerical/visual acceptance criteria should be used for
   the first reproducible GPU comparison?

Implementation and GPU results remain tracked in the linked issue. This
proposal does not change an existing runtime contract or the supported-model
list.
