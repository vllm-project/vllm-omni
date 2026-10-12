# Chatterbox full request batching

## Goal and current priority

Optimize measured end-to-end performance while preserving audio correctness.
64 active requests remain the desired load, but a fixed 64-row neural batch
and identical physical batch sizes throughout are not requirements. If a lower
concurrency is necessary for real-time performance, measure and report that
limit explicitly. Never equate admission capacity with real-time performance.

The user's latest clarification supersedes the mandatory full-batch contract
in the earlier design. The remaining sections describe candidate changes,
not required rewrites. Keep a candidate only if profiling and controlled
benchmarks establish a benefit. In particular, do not modify shared HiFT or
CAMPPlus for uniform batching before establishing that this addresses a
measured bottleneck. Internal batch limits are acceptable when their measured
performance justifies them; an unmeasured hardcoded limit is not.

Profile conditioning, T3, flow, vocoding, CPU launch overhead, GPU contention,
and transport separately. Sweep neural batch sizes 1, 2, 4, 8, 16, 32, and 64
where memory permits, at client concurrency 1, 8, 16, 32, and 64. Record
first-audio p50/p95, per-request real-time factor, streaming gaps, throughput,
actual stage occupancy, and peak GPU memory. Compare warmed runs separately
from cold initialization. Choose stage batch sizes from these measurements;
larger batches can increase latency, padding work, and memory use.

Preserve existing mixed-length and per-request state correctness. Test Turbo
and Original independently: Original's ten-step guided flow and paired T3
have different costs. Treat the newly removed cap as an experimental change
until it beats the prior four-row implementation on equivalent GPU workloads.
Use the same quality checks, texts, references, and output budgets throughout.

## Current implementation and first change

The decoder previously split flow and HiFT into four-row calls. Commit
`a859a0227` removes those loops. Flow now takes all scheduled rows, but HiFT
still groups equal mel lengths and splits first/continuing streams. Conditioning
still caps batches at eight, voice encoding splits partial windows at 32, and
CAMPPlus groups equal waveform lengths. These are profiling candidates; remove or tune them only when measurements
justify the change.

Historical H100 results used the old four-row decoder. They do not validate
full-batch memory use or performance. Preserve that distinction in the recipe.

## Candidate: shared capacity and admission

Use stage 1's `max_num_seqs` as the Chatterbox request-capacity source. Validate
stage 0 against it: Turbo requires that many rows, Original with CFG requires
twice that many sequence slots. Pass the same capacity to the API conditioning
worker rather than maintaining a conditioning-specific neural batch constant.
Keep existing cache singleflight and cancellation ownership: only cache misses
need encoding, and an encoder call retains its admission permits until complete.

The conditioning worker drains ready misses up to request capacity. It must not
wait for a full batch. Stage scheduling remains asynchronous; the contract does
not force conditioning and decoding to operate on the same request IDs at the
same instant. Reject inconsistent capacity configuration at initialization.

## Candidate: length-aware conditioning

Retain input lengths explicitly through all reference encoders. S3 tokenization
already accepts padded mel rows and returns lengths. Remove recursive reference
splitting. Voice encoding must process every selected partial window in one
neural invocation and reduce windows back to their owning request, without the
existing 32-window subdivision. Partial windows are a model dimension, not a
second independently tuned request limit.

CAMPPlus needs length-aware intermediate masking and temporal reductions.
Padding only its input is insufficient: convolution and normalization can
populate padded positions, and its context statistics and attention currently
include them. Mask each affected intermediate at the valid temporal extent;
compute means, standard deviations, segment contexts, and pooling over valid
frames only. Keep the no-length calling contract for existing users unchanged.
Remove Chatterbox's equal-waveform-length grouping once this path is verified.

CPU resampling and feature preparation may iterate over reference clips. The
neural encoders must receive all eligible rows together. Cache hits are returned
directly rather than re-encoded to manufacture a fixed physical batch size.

## Candidate: one flow and vocoder batch

Flow uses the existing per-row prompt lengths, masks, and position-indexed
noise. Preserve row identity through final/non-final chunks and reference
length differences. Validate that padded encoder and flow intermediates cannot
affect valid output frames; do not assume an input mask alone establishes this.

Build one padded mel tensor for HiFT, with per-row valid mel lengths, cache
lengths, and first/final flags. Continuing rows prepend their cached mel;
first rows retain their original coordinate origin. Pass a padded source cache
and a per-row cache mask so first rows do not have their newly generated source
overwritten with zeros. Invoke HiFT once for the whole scheduled request batch.
Apply trimming, overlap-add, and saved-state extraction at each row's valid end.

Extend HiFT and its F0 predictor with optional lengths rather than adding a
second copy of their networks. Noncausal convolutions require zeroing invalid
activations at layer boundaries. Transposed convolutions require propagated
output lengths. Reflection padding and source STFT need each row's true
boundary, and ISTFT must preserve the valid waveform's boundary normalization.
Implement these with batched masks and index gathers, not loops that run a
network for each length. Unspecified lengths preserve existing model behavior.

Do not replace boundary handling with ordinary zero-padding followed by output
truncation: that changes short rows near their ends. Preserve weight names and
checkpoint loading. No changes to checkpoint architecture or diffusion steps.

## Verification for any selected candidate

Add batch-contract tests at conditioning, flow, and HiFT entry points: a
64-request invocation must not become smaller repeated neural calls. Test mixed
reference lengths, unequal chunk lengths, first/continuing rows, final chunks,
and cancellation. Keep distinct speakers, noise positions, and cached source
rows observable so row mixing fails explicitly.

Compare actual neural outputs against standalone calls for mixed lengths.
For stochastic source generation, supply identical per-row random inputs in
the test; changing RNG draw layout is not evidence of padding contamination.
Check both variants, audio seams, tails, and speaker embeddings. Preserve
upstream reference parity where the old implementation established it.

Run affected shared HiFT and speaker-model regressions, Chatterbox adapter and
decoder tests, deployment validation, and pre-commit. Commit and sync through
the existing host scripts. First validate correctness and memory on the
available A5000 at its supported capacity; full 64-stream validation needs a
fresh H100 and must report peak activation memory and OOMs explicitly.

On H100 measure cold references, warm references, and built-in voices, with
short sentence inputs and the existing long capacity probe separately. Record
first audio, per-request real-time factor, streaming gaps, stage timings, and
actual stage occupancy. Long-input capacity success is not narration-quality
success. Compare with the old revision using the same inputs and hardware.
Do not call the implementation real-time until sustained per-stream production
keeps up with playback at 64 active requests.
