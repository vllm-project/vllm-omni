# AuK speech generation and editing — H100

> Instruction-driven zero-shot TTS, instruct TTS, speech editing, enhancement
> and separation with Tencent's AuK, served as a two-stage vLLM-Omni pipeline
> (Qwen2.5-Omni thinker encoder, then a rectified-flow DiT with a BigVGAN-flow
> VAE) on one H100.

## Summary

- Vendor: Tencent Hunyuan
- Model: `tencent/AuK` (32-step base) and `tencent/AuK-Flash` (distilled 4-step),
  both with `Qwen/Qwen2.5-Omni-3B` as the frozen encoder
- Task: zero-shot TTS, instruct TTS, content and acoustic editing,
  paralinguistic editing, speech enhancement, speaker and music separation
- Mode: offline inference (online chat serving is not yet qualified)
- Hardware: 1x H100 80GB
- Maintainer: Community

## When to use this recipe

Use this recipe to run AuK through vLLM-Omni for any of its instruction-driven
tasks with a single shared request shape: an instruction, an optional source or
reference clip, and a target duration. The task is chosen by the instruction
text, exactly as in the upstream repository's cookbook.

## Supported model contract

| Task family | Instruction | Audio input | Duration |
| --- | --- | --- | --- |
| Zero-shot TTS | "Say the following with the same voice: '...'" | reference clip | `gen_seconds` required |
| Instruct TTS | `Generate speech based on the following description: "{voice description}". The content to speak is: "{text}".` (upstream template; other orderings can make the model speak the description) | none | `gen_seconds` required |
| Content / lyric editing | "Replace '...' with '...'" and similar | source clip | defaults to the source length |
| Acoustic / paralinguistic editing | pitch, speed, volume, emotion, timbre, de-accent, nonverbal, whisper | source clip | defaults to the source length (speed edits scale it) |
| Enhancement / separation | "Keep pure speech voice, remove noise and reverberation." and similar | source clip | defaults to the source length |

| Property | Value |
| --- | --- |
| Input audio | mono or stereo (averaged), any sample rate (resampled to 24 kHz for the VAE and to 16 kHz for the encoder), one clip per request |
| Output | mono float32 waveform at 24 kHz, `ceil(gen_seconds * 50)` latent frames |
| Sequence limit | reference plus target latents up to 65536 frames (about 21 minutes) |
| Sampling knobs | `num_inference_steps` (default 32), `guidance_scale` (default 2.0), `seed`; Flash pins 4 steps, CFG 0 and its own time grid |
| Request knobs | `gen_seconds`, `sway` (default -1.0), `t_grid`, `vae_sample` via `additional_information["auk"]` |
| Concurrency | one request per DiT forward; the encoder stage runs with `max_num_seqs: 1`, prefix caching and chunked prefill off |
| Streaming | none: the ODE runs over the whole target |

## References

- Upstream repository and cookbook: <https://github.com/Tencent-Hunyuan/AuK>
- Technical report: <https://arxiv.org/abs/2609.08936>
- Weights: <https://huggingface.co/tencent/AuK>, <https://huggingface.co/tencent/AuK-Flash>,
  <https://huggingface.co/Qwen/Qwen2.5-Omni-3B>
- Pipeline topology: `vllm_omni/model_executor/models/auk/pipeline.py`;
  deploy defaults: `vllm_omni/deploy/auk.yaml`
- Request helpers: `vllm_omni/model_extras/auk.py`
- Test: `tests/e2e/offline_inference/test_auk.py`

## Hardware

- Accelerator model and per-device memory: NVIDIA H100 80GB
- Number of devices: 1 (both stages share the device)
- Device interconnect: not applicable
- Host memory: 64 GB is ample; the assembled directory is symlinks only
- Qualification scope: offline `Omni` inference for the four cookbook tasks
  below, base and Flash checkpoints, seed-reproducible, four concurrent
  requests bit-identical to sequential runs

## Software environment

- OS: Linux
- Python: 3.12
- Driver / runtime: CUDA 13 driver with the vLLM 0.29.0 wheel (torch 2.13)
- vLLM version: 0.29.0
- vLLM-Omni version or commit: branch `auk-support` on top of `aff7d6494`

## Command

Assemble the checkpoint directory once (symlinks by default):

```bash
hf download tencent/AuK --local-dir ckpts/AuK
hf download Qwen/Qwen2.5-Omni-3B --local-dir ckpts/Qwen2.5-Omni-3B
python tools/prepare_auk_checkpoint.py --auk-dir ckpts/AuK --qwen-dir ckpts/Qwen2.5-Omni-3B --out ckpts/auk-omni
# For AuK-Flash: --auk-dir ckpts/AuK-Flash --out ckpts/auk-omni-flash
```

Offline inference (any task; the instruction selects it):

```python
import soundfile as sf
from vllm.multimodal.media.audio import load_audio

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.model_extras.auk import auk_prompt, auk_sampling_params

omni = Omni(model="ckpts/auk-omni")  # loads vllm_omni/deploy/auk.yaml
wav, sr = load_audio("ref.wav", sr=None)
prompt = auk_prompt(
    "Say the following with the same voice: 'Ladies and gentlemen, welcome.'",
    (wav, sr),
    gen_seconds=4.0,
)
out = omni.generate(prompt, auk_sampling_params(nfe=32, cfg=2.0, seed=0))[0]
sf.write("out.wav", out.multimodal_output["audio"], out.multimodal_output["audio_sample_rate"])
```

Text-only instruct TTS passes `None` for the audio and must set `gen_seconds`.
Editing and enhancement requests pass the source clip and may omit
`gen_seconds` to keep the source length.

## Verification

```bash
VLLM_OMNI_AUK_MODEL_DIR=ckpts/auk-omni python -m pytest tests/e2e/offline_inference/test_auk.py -q
```

The parity suites compare the port against the upstream implementation and
need reference artifacts produced by the upstream package (install it with
`pip install -e /path/to/AuK` in its own environment; it pins its own torch):

```bash
python tools/auk_parity_reference.py --auk-repo /path/to/AuK --ckpt-dir ckpts/AuK \
    --qwen-dir ckpts/Qwen2.5-Omni-3B --out auk-parity
AUK_CKPT_DIR=ckpts/AuK AUK_REF_WAV=/path/to/AuK/assets/demo-input-audio/zero-shot-tts/ref.wav \
    python -m pytest tests/diffusion/models/auk/test_auk_transformer_parity.py tests/diffusion/models/auk/test_auk_vae_parity.py -q
AUK_OMNI_CKPT_DIR=ckpts/auk-omni AUK_PARITY_REF=auk-parity/parity_ref/base \
    python -m pytest tests/diffusion/models/auk/test_pipeline_auk.py -q
```

Measured on one H100 against the upstream implementation with the same seeds
and mean VAE latents (four cookbook cases, base checkpoint): identical
transcripts, speaker similarity to the upstream output 0.993 to 0.999, log-mel
L1 0.11 to 0.47 (the upstream VAE-resampling noise floor is 0.39). Flash:
0.989 to 0.999 and 0.24 to 0.51. Wall per request on the second call: base
1.0 to 2.0 s for 3.5 to 10.9 s of audio, Flash 0.17 to 0.30 s; engine start
about 47 s with a warm page cache.

## Notes

- Memory usage: about 25 GB peak on the device for a 12 s generation with
  both stages resident (deploy defaults: encoder 0.45, diffusion stage 0.35
  of device memory).
- Key flags: `enforce_eager` on both stages (the encoder walks the decoder
  layers itself for the layer fusion). `enable_prefix_caching` must stay off
  for the encoder: a cache hit skips prompt positions that the fused
  condition needs. `enable_chunked_prefill` is off by default: forcing it
  (128-token chunks, so two to three chunks per prompt) reproduces the
  unchunked condition to bf16 rounding (per-token cosine 0.99999) and leaves
  request walls unchanged, because the encoder is about 3% of a base request,
  while concurrent runs stop being bit-identical to sequential ones.
- Known limitations: the encoder's audio tower runs without `flash_attn` in a
  plain vLLM install, which shifts the encoder output on audio token positions
  (per-token cosine 0.96 vs the upstream fp32 fusion; text positions 0.9999)
  with waveform-level effects at the VAE noise floor; installing `flash-attn`
  removes the gap. No streaming. Digits are read the way upstream reads them
  (spell numbers out in the text). Instruct TTS without a reference can leak
  the description into the speech, as upstream does.

## Supported features

| Feature | Status | Guide |
| --- | --- | --- |
| Offline `Omni.generate` | supported (base and Flash) | `docs/getting_started/quickstart.md` |
| Online `/v1/chat/completions` with audio | not yet qualified | `docs/serving/` |
| `/v1/audio/speech` | not supported (needs a TTS adapter) | `docs/contributing/model/adding_tts_model.md` |
| Streaming / async chunk | not supported | `docs/design/feature/async_chunk.md` |
| Batching across requests | one request per DiT forward | `docs/user_guide/diffusion/` |
| Tensor / sequence parallelism | not supported | `docs/configuration/composable_parallel.md` |
