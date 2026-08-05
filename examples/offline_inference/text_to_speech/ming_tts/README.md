# Ming-omni-tts Offline Inference

`end2end.py` runs Ming dense 0.5B end to end with vLLM-Omni. It uses the in-repo Ming prompt assembly helper directly, so the example request shape matches the real integration instead of a simplified wrapper.

## Files

| File | Purpose |
|---|---|
| `end2end.py` | Driver: CLI, case loading, prompt construction, orchestration (~150 lines) |
| `cases.yaml` | All 11 built-in case definitions (prompt, text, instruction, ref-audio flags, flow controls) |
| `runner.py` | Engine management and audio output (streaming + blocking paths) |

## Model Overview

Ming dense 0.5B is exposed here as a two-stage offline pipeline:

- **Stage 0**: Qwen2-based AR generation with Ming prompt formatting and inline flow controls
- **Stage 1**: audio VAE decode to mono 44.1 kHz waveform

`config_ming_tts.py` adapts the checkpoint's HuggingFace config fields
(LLM, DiT, aggregator, AudioVAE, token ids). `vllm_omni/deploy/ming_tts.yaml`
selects the vLLM-Omni pipeline and stage runtime topology, including
connectors, async chunking, memory limits, and sampling defaults.

The example supports both:

- **Blocking eager** via `vllm_omni/deploy/ming_tts.yaml`
- **Async chunk eager** via `vllm_omni/deploy/ming_tts.yaml` (default `async_chunk: true`)

## Setup

Install vLLM-Omni with the platform requirements for your accelerator:

```bash
uv pip install -e .
```

The Ming offline example does not require a separate upstream Ming package.
Reference-audio cases use the repo dependencies for audio loading,
resampling, and CampPlus speaker extraction, including `soundfile`,
`torchaudio`, and `onnxruntime-rocm` in the documented ROCm environment.

The tested ROCm environment is summarized in the
[Ming recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/inclusionAI/Ming-omni-tts-0.5B.md).

## Supported Cases

These cases cover the upstream dense 0.5B cookbook surface that maps cleanly onto the current vLLM-Omni example:

- `style`: zero-speaker style-conditioned speech
- `ip`: zero-speaker IP voice generation
- `bgm`: music-only generation
- `tta`: text-to-audio event generation with FlowLoss controls
- `emotion`: reference-audio speech with emotion control
- `basic`: reference-audio speech with speed / pitch / volume control
- `dialect`: reference-audio speech with dialect control
- `zero_shot`: reference-audio cloning with explicit transcript
- `podcast`: multi-reference dialogue generation with automatic speaker embedding extraction
- `speech_bgm`: speech with background music conditioning
- `speech_sound`: speech with environmental sound conditioning

## Quick Start

Run the zero-speaker style example:

```bash
python examples/offline_inference/text_to_speech/ming_tts/end2end.py \
    --case style \
    --deploy-config vllm_omni/deploy/ming_tts.yaml \
    --enforce-eager
```

Run zero-shot cloning with a transcript:

```bash
python examples/offline_inference/text_to_speech/ming_tts/end2end.py \
    --case zero_shot \
    --ref-audio /path/to/10002287-00000094.wav \
    --ref-text "在此奉劝大家别乱打美白针。" \
    --deploy-config vllm_omni/deploy/ming_tts.yaml \
    --enforce-eager
```

Run emotion-controlled speech:

```bash
python examples/offline_inference/text_to_speech/ming_tts/end2end.py \
    --case emotion \
    --ref-audio /path/to/emotion_prompt.wav \
    --deploy-config vllm_omni/deploy/ming_tts.yaml \
    --enforce-eager
```

Run podcast generation with two reference clips:

```bash
python examples/offline_inference/text_to_speech/ming_tts/end2end.py \
    --case podcast \
    --ref-audio-paths /path/to/CTS-CN-F2F-2019-11-11-423-012-A.wav /path/to/CTS-CN-F2F-2019-11-11-423-012-B.wav \
    --deploy-config vllm_omni/deploy/ming_tts.yaml \
    --enforce-eager
```

The script automatically extracts one 192-d speaker embedding per reference WAV using the Ming model's `campplus.onnx`.

If you already have precomputed multi-speaker embeddings, you can override extraction with:

```bash
--speaker-embedding /path/to/podcast_speaker_embeddings.json
```

where the JSON is a list of speaker embeddings, one 192-d vector per speaker.

Run text-to-audio event generation:

```bash
python examples/offline_inference/text_to_speech/ming_tts/end2end.py \
    --case tta \
    --deploy-config vllm_omni/deploy/ming_tts.yaml \
    --enforce-eager
```

Use async_chunk streaming:

```bash
python examples/offline_inference/text_to_speech/ming_tts/end2end.py \
    --case basic \
    --ref-audio /path/to/10002287-00000095.wav \
    --streaming \
    --deploy-config vllm_omni/deploy/ming_tts.yaml \
    --enforce-eager
```

`--streaming` uses `AsyncOmni` and the async_chunk stage config. It currently
supports one prompt per process invocation; use blocking mode for
`--num-prompts > 1`.

Collect runtime stats and a manifest:

```bash
python examples/offline_inference/text_to_speech/ming_tts/end2end.py \
    --case style \
    --deploy-config vllm_omni/deploy/ming_tts.yaml \
    --enforce-eager \
    --enable-stats \
    --stats-log-file output_audio/ming_style_pipeline.log \
    --metadata-json output_audio/ming_style_manifest.json
```

## Reference Fixtures

The upstream Ming cookbook uses these public audio fixtures from `inclusionAI/Ming-omni-tts/data/wavs`:

- `10002287-00000094.wav` for zero-shot cloning
- `10002287-00000095.wav` for `basic`
- `emotion_prompt.wav` for `emotion`
- `yue_prompt.wav` for `dialect`
- `00000309-00000300.wav` for `speech_bgm` and `speech_sound`
- `CTS-CN-F2F-2019-11-11-423-012-A.wav` and `CTS-CN-F2F-2019-11-11-423-012-B.wav` for `podcast`

## Validation Matrix

The repo-facing example is intended to cover the same dense TTS workflows used
by the local Ming validation script:

| Case | Blocking `deploy/ming_tts.yaml` | Async chunk `deploy/ming_tts.yaml` | Extra inputs |
|---|---:|---:|---|
| `style` | Yes | Optional smoke test | none |
| `ip` | Yes | Optional smoke test | none |
| `bgm` | Yes | Optional smoke test | none |
| `tta` | Yes | Optional smoke test | none |
| `emotion` | Yes | Yes | `--ref-audio emotion_prompt.wav` |
| `basic` | Yes | Yes | `--ref-audio 10002287-00000095.wav` |
| `dialect` | Yes | Yes | `--ref-audio yue_prompt.wav` |
| `zero_shot` | Yes | Yes | `--ref-audio 10002287-00000094.wav --ref-text ...` |
| `podcast` | Yes | Yes | two `--ref-audio-paths` |
| `speech_bgm` | Yes | Yes | `--ref-audio 00000309-00000300.wav` |
| `speech_sound` | Yes | Yes | `--ref-audio 00000309-00000300.wav` |

## Validated Outputs

The following measurements are retained from an earlier L4 CUDA validation;
they are not ROCm benchmark results. Default async_chunk matched blocking
output frame counts and Stage-1 patch counts for every case:

| Case | Blocking frames / patches / sec | Async chunk frames / patches / sec |
|---|---:|---:|
| `style` | 409248 / 29 / 9.28 | 409248 / 29 / 9.28 |
| `ip` | 183456 / 13 / 4.16 | 183456 / 13 / 4.16 |
| `bgm` | 1326528 / 94 / 30.08 | 1326528 / 94 / 30.08 |
| `tta` | 465696 / 33 / 10.56 | 465696 / 33 / 10.56 |
| `emotion` | 324576 / 23 / 7.36 | 324576 / 23 / 7.36 |
| `basic` | 211680 / 15 / 4.80 | 211680 / 15 / 4.80 |
| `dialect` | 239904 / 17 / 5.44 | 239904 / 17 / 5.44 |
| `zero_shot` | 409248 / 29 / 9.28 | 409248 / 29 / 9.28 |
| `podcast` | 437472 / 31 / 9.92 | 437472 / 31 / 9.92 |
| `speech_bgm` | 296352 / 21 / 6.72 | 296352 / 21 / 6.72 |
| `speech_sound` | 352800 / 25 / 8.00 | 352800 / 25 / 8.00 |

## Key Arguments

| Argument | Description |
|---|---|
| `--model` | Hugging Face repo or local Ming checkpoint path |
| `--deploy-config` | Deploy config YAML. Use `vllm_omni/deploy/ming_tts.yaml` |
| `--case` | Built-in demo case |
| `--ref-audio` | Single reference wav path for cloning-style cases |
| `--ref-audio-paths` | Multiple reference wav paths, used by `podcast` |
| `--ref-text` | Reference transcript. Required for `zero_shot` |
| `--instructions` | Free-form Ming instruction string |
| `--instruction-json` | Structured Ming instruction JSON |
| `--speaker-embedding` | JSON file containing a 192-d speaker embedding |
| `--extract-speaker-embeddings` | Force CampPlus speaker extraction from the provided reference audio paths |
| `--max-decode-steps` | Override `ming_max_decode_steps` |
| `--num-prompts` | Repeat the same case N times. Outputs are indexed when `N > 1` |
| `--streaming` | Use `AsyncOmni` and async_chunk transport |
| `--enforce-eager` | Recommended for Ming dense; non-eager is out of scope |
| `--enable-stats` / `--log-stats` | Enable vLLM-Omni per-request stats logging |
| `--stats-log-file` | Optional path for the stats log |
| `--metadata-json` | Optional path for the run manifest JSON |
| `--stage-init-timeout` | Per-stage initialization timeout in seconds |
| `--init-timeout` | Total initialization timeout in seconds |
| `--batch-timeout` | Batch timeout in seconds |
| `--worker-backend` | `multi_process` or `ray` |
| `--ray-address` | Ray cluster address when using `--worker-backend ray` |

## Output

- The script writes one mono 44.1 kHz WAV file per run
- Default output directory: `output_audio/`
- Default filename: `ming_<case>.wav`
- When `--num-prompts > 1`, outputs are indexed as `ming_<case>_00000.wav`, `..._00001.wav`, etc.
- When stats are enabled, the script can also write:
  - a stats log file such as `ming_style_pipeline.log`
  - a manifest JSON with per-output metadata, stage durations, peak memory info,
    and streaming client latency metrics when `--streaming` is used

## Notes

- `style` and `ip` are zero-speaker paths and do not require a reference clip
- `emotion`, `basic`, `dialect`, `speech_bgm`, and `speech_sound` require one reference clip
- `zero_shot` requires both `--ref-audio` and `--ref-text`
- `podcast` requires at least two reference clips via `--ref-audio-paths`
- `podcast` automatically extracts one speaker embedding per reference clip
- `--speaker-embedding` may contain either one 192-d vector or a list of 192-d vectors
- `--enforce-eager` was used for the validated runs
- The earlier L4 validation used SDPA for the Ming audio VAE instead of
  FlashAttention2, which is the preferred default when available.
