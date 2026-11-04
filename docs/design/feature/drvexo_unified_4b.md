# DRVEXO unified 4B serving integration

**Status: draft; checkpoint-reader groundwork only. No registered serving support.**

## Outcome and scope

Provide a native vLLM-Omni path for a complete merged `drvexo-4b-unified`
BF16 package: text, images, speech input, speech output with reference voices,
explicit candidate decisions and streaming. Preserve existing architectures and
request isolation. The separate DRVEXO Speech 8B contribution does not establish
support for this checkpoint.

The local source uses `drvexo_speech_qwen35_4b` configuration, Qwen3.5 language
and vision parameters, 32 language layers with linear/full attention, a trained
temporal audio connector, Talker/Codec, speaker conditioning and a general head.
The v12 adapter has already been merged into 248 language projections. The
complete package has 2,131 BF16/FP32 tensors in eight components; no runtime
PEFT branch or additional base-model directory is required.

## Implemented boundary

`UnifiedBF16Checkpoint` checks the complete component set, per-component tensor
counts, unmerged/quantized rejection, single component ownership, safe shard
paths, full header/index coverage and retained BF16/FP32 precision. It exposes
component-relative names and a language-plus-vision iterator. It does not copy,
convert, merge, quantize or rewrite the source package.

The file-format unit tests import only this engine-independent reader. They
can run with `pytest --noconftest`; this is not a substitute for package import,
native model construction or a live serving test.

## Remaining integration

- Use the native Qwen3.5 hybrid language/vision architecture and its processor,
  positions and cache policy. Do not substitute the 8B Qwen3 dense decoder.
- Add the temporal audio connector and audio-input processor without changing
  its trained shapes or BF16 Conv1d / retained FP32 parameters.
- Bind native Talker/Codec and reference-speaker inputs to the unified component
  views. Handle the Qwen3.5 and Talker tokenizer boundary explicitly; the old
  generated-token mapping cannot be assumed complete for a different vocabulary.
- Establish a fresh general-head binding to the merged backbone and evaluate
  candidate scoring; the reference-method config is not a native binding.
- Register config, model and staged pipeline only with executable tests. Keep
  admission validation, disconnect cleanup and session ownership explicit.
- Add a local BF16 recipe with actual environment and matching-head evidence.

## Acceptance

Validate independent loading from this package, text and long-context extraction,
JSON and tool-call text, image questions, ASR, default and reference voice output,
candidate decisions, committed multi-turn context and cancellations. Record
TTFT, first audio, sustained text/audio rate and chunk gaps separately. Compare
the same inputs with the merged reference runtime; old 8B or unmerged v12 scores
do not transfer. Distinguish a synthetic recording, a live microphone, protocol
completion and semantic correctness.

The current upstream baseline requires vLLM 0.33; validation on the historical
0.32 environment does not establish compatibility. No new quality or performance
claim, public model download, deployment or production replacement is made by
this draft.
