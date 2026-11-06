# Qwen duplex history prototype

The opt-in history strategy keeps a bounded text memory of earlier dialogue.
A private Thinker request reads the previous memory and original text/audio in
one inference. It does not transcribe audio in a separate request or route its
output to Talker. The default `recent` policy keeps existing behavior.

This prototype supports text and native audio. Selecting `incremental` or
`bounded_recent` disables image input and rejects tool configuration. Session
records retain their existing query, deletion and playback semantics. Their
total storage is not bounded by this feature.

## Deployment

Create an overlay outside the repository, using an absolute base path:

```yaml
base_config: /path/to/vllm-omni/vllm_omni/deploy/qwen3_omni_duplex.yaml
duplex_session:
  model_options:
    history_policy: incremental
    history_limits:
      recent_messages: 8
      chunk_messages: 8
      summary_tokens: 512
      max_audio_items: 64
      max_audio_bytes: 33554432
      max_context_tokens: 2048
```

Start the server through your GPU scheduler:

```bash
vllm serve /path/to/Qwen3-Omni-30B-A3B-Instruct \
  --omni --deploy-config /path/to/history.yaml --port 8091
```

For the paired clipping control change only `history_policy` to
`bounded_recent`. Both policies receive the same model context limit and
media retention allowance. Clipping selects recent complete turns up to the
available input capacity, without the legacy 16-message restriction. Each
request still obeys the model's audio count limit. The summary policy incurs
additional inference; equal input limits do not imply equal compute.

`max_context_tokens` is an experimental cap below the actual model limit, not
a model-capacity claim. If omitted, the resolved model limit applies. Set the
same output allowance for both policies. Confirm that the earliest fact has
actually left the control's input before interpreting a long-history result.
The 2048-token overlay is a starting experiment, not a tuned recommendation.

The 32 MiB bound counts encoded unique PCM sources held by the strategy,
including preparation references. It represents about 393 seconds of 16 kHz
float32 audio after Base64 encoding. It is not a limit on process RSS: Session
records, foreground buffers, decoded arrays, feature extraction, and engine
resources have separate lifetimes. A batch admits at most four audio payloads;
feature extraction runs on CPU and can temporarily hold decoded copies.
Oversized individual audio is not sliced. Unavailable sources and input that
cannot fit a summary request produce distinct omissions; subsequent turns
continue. Failed model completions retain progress and have bounded short
retries. Audio retained solely for processed sources is released.

## Frozen evaluation inputs

`cases.jsonl` contains 10 development scenarios and 30 evaluation scenarios.
Every evaluation scenario has a query about an early fact; ten additionally
exercise persistent instructions and ten exercise corrections. A scenario,
not an individual turn or its audio/text rendition, is the comparison unit.

The user script never changes in response to generated answers. Source texts,
question positions and expected answers are fixed before evaluation. The
script's initial facts and corrected values use distinct development topics.

Optional CPU-only synthetic speech preparation:

```bash
uv run --no-sync python -m vllm_omni.benchmarks.duplex_history audio \
  --manifest benchmarks/duplex_history/cases.jsonl --audio-dir /tmp/history-audio
```

This requires `espeak-ng` with Mandarin support. Synthetic speech must be
listened to and its key facts, negations and corrections checked before a
voice result is treated as reliable. Generated WAV files are not checked in.
The runner resamples them to 16 kHz PCM16 for the wire protocol.

After warming up the deployment, run the same manifest against each policy:

```bash
uv run --no-sync python -m vllm_omni.benchmarks.duplex_history run \
  --manifest benchmarks/duplex_history/cases.jsonl --split eval \
  --url 'ws://localhost:8091/v1/realtime?duplex=1' --model MODEL_NAME \
  --policy incremental --modality text --idle-seconds 3 \
  --output /tmp/history-incremental-text.jsonl
```

For native speech use `--modality audio --audio-dir /tmp/history-audio`.
Repeat against the clipping deployment with `--policy bounded_recent` and a
new output path. The policy flag labels results; it does not change server
configuration. Keep the deployment overlays with the results.

Use the development split with `--limit 2` for connectivity checks before a
full run. The main 30-scenario comparison costs about 2,400 foreground
responses across two input modes and two policies at 20 turns per scenario,
plus development, warmup, background summaries, and latency experiments.

Normal runs wait for response completion and simulated playback, then use a
fixed idle interval. `--idle-seconds 0` measures back-to-back interaction with
no dedicated consolidation wait. This is not a concurrent-user or deliberate
mid-speech interruption benchmark. CPU state tests cover interruption,
source invalidation and close/late-result behavior separately.

```bash
uv run --no-sync python -m vllm_omni.benchmarks.duplex_history report \
  /tmp/history-bounded-text.jsonl /tmp/history-incremental-text.jsonl
```

The report contains per-category counts, paired improvements/regressions,
unresolved judgments, and P50/P95 client first-text/first-audio latency from
response submission (audio commit for voice input). It imposes no score or
performance threshold. `needs_review` results remain visible, not counted as
correct by substring matching. Review them against the frozen answer rules
and keep the original response and adjudication evidence.

Record software/model revisions, hardware, overlays, manifest hash, actual
audio bytes and durations, foreground latency, memory RSS, and background
request cost. With `VLLM_LOGGING_LEVEL=DEBUG`, Qwen memory logs expose retained encoded bytes, high-water
marks, completed summaries, omissions and invalidations. Real model results
are required before making recall, latency or resource-efficiency claims.

Each successful foreground prompt build also emits one debug record:
`Qwen history foreground degraded=... trimmed_turns=... summary_dropped=...`.
`degraded=1` means context fitting removed at least one offered old turn or the
existing summary; sum this field to count degraded prompt builds. `trimmed_turns`
counts removed whole turns and `summary_dropped` is 0 or 1. These are preparation
counts, including a plan subsequently cancelled before engine admission; they
are not counts of completed replies or model errors. A current input that cannot
fit raises an error instead of producing a degradation record. These fields do
not count sources already missing before rendering; media eviction and summary
omission counters are separate.

Media consolidation starts when the smaller configured-pool/foreground-cache
item limit is reached, or the byte limit cannot hold another utterance as large
as the largest retained one, independently of the text-message threshold.
Still-valid foreground-cache audio can supply a summary after strategy eviction;
selected sources are charged to the strategy limit until feature preparation
finishes, including cancellation. If a source exists but cannot fit that limit,
the omission reason is `summary_capacity`, not `source_unavailable`.

## Replacing the policy

Set `history_policy: your_package.policy:YourPolicy`. The server imports the
class with a no-argument constructor. Clients cannot supply this import path.
Inherit the `QwenHistoryPolicy` abstract base class in `duplex/history_policy.py`
and implement `select(state, limits)` and `next_batch(state, limits)`. Incomplete
implementations are rejected when the server loads the strategy. The built-in clipping and
incremental classes are executable examples of the interface.

A policy chooses ordered whole-turn source ranges. The Qwen plugin retains
responsibility for input construction, context bounds, media correspondence,
source validity and result application. State is per Session. This prototype
interface has no compatibility guarantee and does not promise arbitrary
retrieval stores or third-party summarization services.

Policy decisions may be cached while the Session history revision, memory
generation and retained-media revision are unchanged. A policy must be stateless
and must not depend on elapsed wall time or mutate state while selecting ranges.

## Local state and regression tests

Use a virtual environment compatible with this branch (vLLM 0.33 and the
repository Transformers requirements). These tests require no GPU:

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m pytest -o addopts='' \
  tests/model_executor/models/qwen3_omni/test_history_policy.py \
  tests/benchmarks/test_duplex_history.py -q

CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m pytest -o addopts='' \
  -m 'core_model and cpu' \
  tests/model_executor/models/qwen3_omni \
  tests/model_executor/models/joyai_vl_interaction/test_duplex_memory.py -q
```
