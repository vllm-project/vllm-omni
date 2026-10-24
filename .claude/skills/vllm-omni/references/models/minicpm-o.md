# MiniCPM-o 4.5

Code: `vllm_omni/model_executor/models/minicpmo_4_5/`. Recipe:
`recipes/OpenBMB/MiniCPM-o-4_5.md`. Design:
`docs/design/minicpm_o45_mrv2_performance.md`,
`docs/design/minicpmo_encoder_cuda_graph.md`, `docs/design/fullduplex.md`.

MiniCPM-o 5.0 has no tracked sources, registry entry, or deploy YAML in this
tree; a `minicpmo_5_0/` directory with only `__pycache__` is a leftover.

## Stages

| Stage | `model_stage` | Arch | Main files | Output |
|-------|---------------|------|-----------|--------|
| 0 | `llm` | `MiniCPMO45OmniForConditionalGeneration` | `minicpmo_4_5_omni_llm.py`, vision/audio encoders | Text (final) + latents |
| 1 | `tts` | same unified class | `minicpmo_4_5_omni_tts.py` | Codec tokens |
| 2 | `code2wav` | `MiniCPMO45Code2Wav` (separate arch) | `minicpmo_4_5_code2wav.py`, `batched_token2wav.py`, `minicpmo_4_5_token2wav.py` | Waveform (final) |

Stage names differ from Qwen3-Omni (`llm` / `tts`, not `thinker` / `talker`).
Hand-off: `stage_input_processors/minicpmo_4_5_omni.py` (`llm2tts`,
`tts2code2wav_*`). Chat template:
`vllm_omni/transformers_utils/chat_templates/minicpmo45_native.jinja`.
Duplex plugin: `minicpmo_4_5/duplex/plugin.py`.

## Deploy YAMLs (`vllm_omni/deploy/`)

| YAML | Use |
|------|-----|
| `minicpmo_4_5.yaml` | Default: **duplex**, all stages on GPU 0 (0.55 / 0.15 / 0.18), `max_sessions: 16`, shared-memory connector with `codec_chunk_frames: 25`, `codec_left_context_frames: 3` |
| `minicpmo_4_5_2gpu.yaml` | Recommended production: thinker on GPU 0, tts + code2wav on GPU 1 |
| `minicpmo_4_5_3gpu.yaml` | One GPU per stage |
| `minicpmo_4_5_{3,4}gpu_stage1_replicas.yaml` | Experimental tts replicas (2 or 3) |
| `minicpmo_4_5_8x4090.yaml` (+ `_stage1_replicas`) | Thinker TP=4, `max_model_len: 4096`, mostly eager |
| `minicpmo_4_5_duplex_h200.yaml` | Duplex with incremental fbank, fused vision, duplex audio-encoder graphs, stage-2 TF32 and fused CFM |
| `minicpmo_4_5_turn.yaml` | `session_mode: turn` for ordinary HTTP chat |
| `minicpmo_4_5_turn_mrv2.yaml` (+ `_h200`) | Turn mode on MRv2; tts/code2wav `max_num_seqs` 8 (16 on H200) |

## Known quirks

- The default YAML is duplex. HTTP chat benchmarks and Daily-Omni runs need a turn
  YAML.
- Disable thinking in `--extra-body` for Daily-Omni, or the answer budget is spent
  on reasoning; use `--daily-omni-pack-mode minicpm-interleave`.
- Stage 2 outer runner is always eager; CFM / HiFT / flow-encoder graphs are
  enabled through connector or engine extras, not `enforce_eager`.
- Thinker encoder CUDA graphs (`encoder_cuda_graph.py`): exact-shape graphs for
  vision and stateless audio, captured on the second call, at most 4 per encoder
  by default, configurable via `--hf-overrides`. CPU, grad, autocast, and stateful
  duplex audio stay eager. Capture failure is fatal. `--enforce-eager` disables them.
- Stage-2 flow-encoder graph replay must run inside the same autocast context as
  capture (`batched_token2wav.py`); replay outside it changed numerics. Regression
  test: `test_decode_replays_encoder_graph_in_capture_autocast` in
  `tests/model_executor/models/minicpmo_4_5/test_code2wav_batching.py`.
- MRv2 applies to turn mode only; duplex stays on v1. Earlier MRv2 performance
  numbers were invalidated; re-measure. TF32 for CFM is opt-in (HiFT stays FP32).

## Benchmarks and tests

- E2E: `vllm bench serve --omni` with a turn deploy; duplex via
  `--backend openai-realtime-duplex --dataset-name omniinteract`.
- Tests: `tests/model_executor/models/minicpmo_4_5/` (incl. `duplex/`),
  `tests/model_executor/stage_input_processors/test_minicpmo_4_5*.py`,
  `tests/e2e/offline_inference/test_minicpmo_4_5.py`,
  `tests/e2e/online_serving/test_minicpmo_4_5*.py`.
