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
| `qwen3_omni_moe.yaml` | Default. 2×H100: thinker on GPU 0 (0.9), talker + code2wav on GPU 1 (0.6 / 0.1). `async_chunk: true`, shared-memory connector with `initial_codec_chunk_frames: 4`, `codec_chunk_frames: 25`, `codec_left_context_frames: 25` |
| `qwen3_omni_moe_mrv2.yaml` | MRv2 runners on CUDA; `talker_first_audio`, `codec_chunk_ramp`, Code2Wav graph batch sizes, fused snake. Non-CUDA platforms fall back to v1 |
| `qwen3_omni_moe_mrv2_h200.yaml` | MRv2 with all stages on one H200 (0.60 / 0.22 / 0.10) plus code-predictor KV and sampling optimizations |
| `qwen3_omni_moe_thinking.yaml` | Thinker-only pipeline, TP=2, prefix caching on |
| `qwen3_omni_duplex.yaml` | `session_mode: duplex` |
| `qwen3_omni_moe_mori_intranode.yaml` | MI300X with Mori connector, one GPU per stage, all eager |

## Known quirks

- `enable_prefix_caching: false` on all stages of the full pipeline; the MRv2
  first-frame path also requires it off.
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
GPU 1, all CUDA AR stages eager, prefix caching off. No recipe yet.
