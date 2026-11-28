# Qwen3-Omni (and Qwen2.5-Omni)

Code: `vllm_omni/model_executor/models/qwen3_omni/`. Recipe: `recipes/Qwen/Qwen3-Omni.md`.
Design: `docs/design/qwen3_omni_mrv2_performance.md`,
`docs/design/qwen3_omni_tts_performance_optimization.md`.

## Stages

| Stage | `model_stage` | Type | Module | Output |
|-------|---------------|------|--------|--------|
| 0 | `thinker` | `LLM_AR` | `qwen3_omni_moe_thinker.py` | Text (final) + hidden states |
| 1 | `talker` | `LLM_AR` | `qwen3_omni_moe_talker.py` (+ `qwen3_omni_moe_code_predictor_mtp.py`) | RVQ codec codes |
| 2 | `code2wav` | `LLM_GENERATION` | `qwen3_omni_code2wav.py` | Waveform (final) |

Other files: `pipeline.py` (topology; switches to thinker-only when the HF config
disables audio output), `first_frame_decoder.py` (talker first-audio path),
`vision_encoder_cudagraph.py`, `duplex/`. Hand-off functions:
`stage_input_processors/qwen3_omni.py` (`thinker2talker_*`, `talker2code2wav_*`).

## Deploy YAMLs (`vllm_omni/deploy/`)

| YAML | Use |
|------|-----|
| `qwen3_omni_moe.yaml` | Default MRv2 on CUDA/ROCm/NPU/XPU; MUSA retains an unsupported V1 pin. Thinker on GPU 0 (0.9), talker + code2wav on GPU 1 (0.6 / 0.1). Prefix caching off, `async_chunk: true`, `talker_first_audio`, `[1, 2, 4, 8, 16, 25]` codec chunk ramp, grouped graph buckets and CUDA fused Snake |
| `qwen3_omni_moe_h200.yaml` | MRv2 with all stages on one H200 (0.60 / 0.22 / 0.10) plus code-predictor KV and sampling optimizations |
| `qwen3_omni_moe_thinking.yaml` | Thinker-only pipeline, TP=2, prefix caching on |
| `qwen3_omni_duplex.yaml` | `session_mode: duplex` |
| `qwen3_omni_moe_mori_intranode.yaml` | MI300X with Mori connector, one GPU per stage, all eager |

## Known quirks

- The default deploy disables prefix caching on Thinker, Talker and Code2Wav.
  The duplex and H200 profiles inherit that setting. Thinker caching can be
  enabled explicitly on stage 0. Read the effective profile before tuning.
- Talker first-frame and fused MTP fast paths require Talker prefix caching
  off, independently of Thinker caching. Eligibility lives in
  `vllm_omni/model_executor/models/common/talker_first_audio.py` and
  `vllm_omni/worker_v2/model_states/mtp_model_state.py`.
- `moe_backend: triton` on thinker and talker.
- CUDA forces `custom_ops: ["+rotary_embedding"]` on the thinker; fused MRoPE
  changed audio output.
- Talker uses a fixed `seed: 42` for reproducible speech.
- Code2wav uses its inner CUDA graph on CUDA (`enforce_eager: false`); ROCm
  forces it eager because MIOpen `conv_transpose1d` is not capture-safe. The TTS
  optimization doc's eager example is older than the YAML.
- MRv2 output is not bitwise identical to v1; first-frame decoding needs CUDA,
  a single-rank talker, `talker_first_audio`, and a one-frame first chunk. Duplex
  performance under MRv2 is not established.
- Some realtime routes are unavailable with `async_chunk` on (see recipe).
- The orchestrator is event-driven by default for every pipeline;
  `VLLM_OMNI_EVENT_DRIVEN_ORCH=0` selects the legacy 1 ms poll loop.

## Tuning levers

- First audio packet latency: `initial_codec_chunk_frames`, MRv2 `talker_first_audio`
  and `codec_chunk_ramp`.
- Throughput: talker / code2wav `max_num_seqs`, CUDA graphs, talker replicas.
- Smoothness: `codec_chunk_frames` and `codec_left_context_frames`; verify with
  `audio_continuity_ok_rate`.

## Benchmarks and tests

- E2E bench commands: `docs/cli/bench/serve.md` (`benchmarks/qwen3_omni/` only has a
  thinker embedding harness).
- Tests: `tests/model_executor/models/qwen3_omni/`,
  `tests/model_executor/stage_input_processors/test_qwen3_omni_*.py`,
  `tests/e2e/{offline_inference,online_serving}/test_qwen3_omni*.py`,
  `tests/e2e/accuracy/qwen3_omni/`.

## Qwen2.5-Omni

`vllm_omni/model_executor/models/qwen2_5_omni/`, same thinker/talker/code2wav
shape, processors in `stage_input_processors/qwen2_5_omni.py`. Default
`qwen2_5_omni.yaml`: `async_chunk: false`, thinker + code2wav on GPU 0, talker on
GPU 1. On CUDA, thinker and code2wav are eager; talker uses CUDA graphs
(`enforce_eager: false`). Prefix caching is off. Platform sections override the
graph policy; check the selected backend. No recipe yet.
