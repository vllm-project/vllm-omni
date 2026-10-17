# Parity and Validation

Climb the ladder in order. Each rung isolates one stage or one runtime
feature, so a failure points to a small area.

## Contents
- Parity ladder
- Common failure causes
- Evidence to attach to the PR

## Parity ladder

| Rung | How | Pass criterion |
|------|-----|----------------|
| 1. Thinker text | `"modalities": ["text"]`, `temperature: 0`, same prompt and media as HF | Token-identical, or first divergence explained (kernel numerics) and downstream answer unchanged |
| 2. Talker codec ids | Feed the same thinker conditioning; fix the talker `seed` | Identical ids for greedy; for sampling, identical under the same seed and sampler |
| 3. Code2wav waveform | Feed HF codec ids directly | Waveform within tolerance (report max abs diff, SNR or cosine) and same length |
| 4. Full pipeline, `--no-async-chunk` | Default deploy, one request | Audio matches rung 3 output for the same codes |
| 5. Full pipeline, `--async-chunk` | Same request | Same duration; audio equal apart from chunk boundaries; no clicks, gaps, truncation, or repetition |
| 6. Concurrency | 2 and 4+ different prompts at once | Each output equals its single-request output |
| 7. Graph check | Same request with `--enforce-eager` | Matches graph-mode output |

Then run one accuracy dataset that covers the claimed modalities, for example
Daily-Omni or Video-MME for understanding and Seed-TTS WER for speech, through
`vllm bench serve --omni` (see
`.claude/skills/vllm-omni/references/benchmarking.md`).

## Common failure causes

| Symptom | Check first |
|---------|-------------|
| Thinker text diverges early | Chat template, multimodal placeholder expansion, MRoPE positions, custom ops (Qwen3-Omni forces `+rotary_embedding` on CUDA) |
| Thinker fine, speech nonsense | Wrong hidden layer index handed to the talker; embedding versus hidden state swapped; missing TTS BOS/EOS embeddings |
| Codec ids off by a constant | Codec id offset from the text vocabulary; codebook-major versus frame-major layout |
| Speech never stops | Talker `stop_token_ids` in `sampling_constraints`; codec EOS id |
| Audio wrong length or pitch | Hop length / upsample product; sample rate in output metadata |
| Streaming audio repeats or truncates | Cumulative versus delta emission; left context handled twice; `is_finished` not flushing the tail |
| Clicks at chunk boundaries | `codec_left_context_frames` too small or not trimmed after decode |
| Only fails under concurrency | Per-request state not keyed by request id; shared graph input buffers |
| Graph-only mismatch | Stale static buffers; autocast context differs between capture and replay (MiniCPM-o code2wav regression) |
| Stage 1 waits forever | Full-payload hook returned `None`; `requires_full_payload_input` set without a producer |

Use `.claude/skills/vllm-omni/references/debugging.md` for runtime symptoms
(hangs, OOM, connector stalls).

## Evidence to attach to the PR

- Pinned reference: repository URL and commit, package versions, checkpoint.
- Hardware, driver, vLLM and vLLM-Omni commits, deploy YAML, exact commands.
- Parity table for rungs 1-7 with the comparison metric for each.
- One accuracy result on the same build.
- Smoke performance: TTFT, AUDIO_TTFP, AUDIO_RTF at concurrency 1 and a
  moderate concurrency, labeled as smoke numbers unless a full A/B was run.
- Audio samples (or links) for reviewers: non-streaming and streaming.
