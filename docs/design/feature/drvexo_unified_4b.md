# DRVEXO unified 4B serving integration

**Status: experimental native integration; bounded GB10 smoke validation.**

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

Malformed JSON objects, missing component rows/counts/prefixes and missing
decision-head metadata fail with `ValueError`, rather than leaking key lookup
errors. Header validation is deliberately repeated at independent loading
boundaries; this contribution does not introduce a cross-load metadata cache
or claim a startup I/O improvement.

The file-format unit tests import only this engine-independent reader. They
can run with `pytest --noconftest`; this is not a substitute for package import,
native model construction or a live serving test.

## Native family integration

The family shares the strict stage-view loader, request transport, speech
stages and opt-in endpoints. It retains different Thinkers: 8B uses the dense
Qwen3 decoder, while 4B uses upstream Qwen3.5 hybrid attention and native vision.
The frozen audio tower feeds the trained temporal residual connector. Temporal
convolutions are applied separately to each utterance; images use native mRoPE
positions and audio retains ordinary sequential positions.

The 4B processor reads `assets/qwen35`. Device-side image normalization is
disabled for this architecture because vLLM resolves that configuration from
the checkpoint root. The packaged HF processor performs rescaling and
normalization exactly once. Leaving the inherited device-normalization flag
enabled produced unrelated image answers in a real request; image OCR recovered
after this correction. Images are limited to one per request, audio to one
utterance of at most 390 encoded frames, and video is unsupported.

Generated text crosses the Thinker/Talker vocabulary boundary explicitly. The
source tokenizer must decode the generated IDs to exactly the transmitted text;
the same text is then tokenized for Talker. Source token IDs are never reused
under a different vocabulary. The old hidden-state bridge is not executed.
Explicit speech requests use the caller-text entry. A supplied reference voice
must take precedence over the packaged default speaker, without changing the
next request's speaker.

The general head loads six CPU FP32 tensors and validates their shapes against
the 2560-dimensional backbone. Its final-prefill capture requires explicit
scheduler offsets across chunks, eager BF16 execution, one sequence, no prefix
cache and a chunk budget of 2048. The 4B endpoint retains the training protocol's
4096/384 prefix/candidate budgets; 8B retains 1536/384. A successful candidate
request is not acceptance of the head paired with the newly merged backbone.
This complete deployment always binds the head, including for chat-only
requests; startup errors name each incompatible scheduler/model setting.
Invalid prefill/scoring results retain their explicit failed status. Malformed
client prefill metadata logs at debug level; scoring failures, including
nonfinite numerical results, log at warning level. Neither path logs user text.

Shared chat sampling preserves JSON Schema on the comprehension stage. Forced
and required DRVEXO tool calls use the configured parser's native extraction
when it explicitly does not support the plain JSON tool format, for full and
streaming responses. Other model families retain their existing named/required
JSON path; a parser without a capability flag uses that existing path as well.
Speech stage sampling remains independent of text formatting constraints.
The same comprehension-only constraint rule is covered for Qwen3-Omni; its
downstream Talker/Codec defaults stay unchanged. The DRVEXO speech adapter is
covered for both 4B and 8B caller-text/reference-audio requests and takes its
sampling slice from the declared Talker entry-stage ID. It requires the full
stage-0-indexed sampling list, including Talker and Codec; pre-sliced lists are
rejected. A single-stage Thinker-only list instead reports that speech output
is unavailable in that deployment and requires Talker and Codec. Standard
speech dispatch tests pin the complete-pipeline input contract for both models.
Both standard `/v1/audio/speech` and sentence-speech admission use the local
`talker/` tokenizer to reject control tokens or changed text before engine
submission. Standard speech prompt-length estimation and precomputed text IDs
use the same tokenizer; the root Thinker vocabulary can segment text differently.
The API startup shares the tokenizer loaded off the event loop across these
routes. CPU regressions cover rejection followed by a healthy request and both
streaming and non-streaming prompt modes for both Talker architectures.

## Acceptance

Validate independent loading from this package, text and long-context extraction,
JSON and tool-call text, image questions, ASR, default and reference voice output,
candidate decisions, committed multi-turn context and cancellations. Record
TTFT, first audio, sustained text/audio rate and chunk gaps separately. Compare
the same inputs with the merged reference runtime; old 8B or unmerged v12 scores
do not transfer. Distinguish a synthetic recording, a live microphone, protocol
completion and semantic correctness.

The current upstream baseline requires vLLM 0.33. The local complete merged 4B
package was loaded in a fresh three-stage GB10 process. Bounded requests exercised
arithmetic, committed conversation history, constrained JSON, forced tool calls,
schedule OCR, official DEV audio transcription and candidate scoring. The
schedule returned 09:05, 15 minutes and the stop/report instruction. Default
speech produced a finite 24 kHz WAV and native Code2Wav chunks.

These are integration smoke checks, not broad quality acceptance. One product
identity request still answered “Qwen3.5, developed by Yujie Intelligence”; do not
attribute this to the adapter or claim it fixed without a matched reference
comparison. Photograph details, full merged-head quality, reference-voice
fidelity, live microphones, broad capability retention and load performance
still require separate evaluation. No weights, production deployment or public
download are introduced by this code contribution. See the
[4B model guide](../../models/drvexo_unified_4b.md) for launch details.
