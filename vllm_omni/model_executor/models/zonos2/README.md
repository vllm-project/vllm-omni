# ZONOS2 DAC and speech conditioning

The two-stage pipeline decodes nine DAC codebooks into float32 mono PCM at
44,100 Hz, with 512 samples per frame. The default deployment enables
codec chunks with 16 new decodable frames and a four-frame raised-cosine
overlap. Eight raw delay frames provide right context. Final output stops
at the aligned EOS boundary and flushes the withheld tail.

Both stages require eager execution and synchronous scheduling. Codec chunk
transport is separate from vLLM asynchronous scheduling. For a synchronous
whole-request handoff, set `async_chunk: false` and connector
`codec_streaming: false` in a deployment copy.

## Optional dependencies and offline assets

Install the model-specific packages into the vLLM-Omni environment:

```bash
uv pip install -r requirements/zonos2.txt --overrides requirements/zonos2-overrides.txt
```

Install torchaudio from the same distribution/version as torch.
Reference audio uses the existing serving MediaConnector for data URIs, URLs
and permitted local files. The serving audio backend handles decoding;
ffmpeg must be available for formats requiring it. Invalid media/backend
errors are reported before generation. CI fixtures must be local or inline.

The protobuf override preserves the vLLM/Ray-compatible version (6.33.6).
audiotools 0.7.2 declares an obsolete protobuf `<3.20` dependency; its DAC
inference path is verified with the override. This exception is scoped to
ZONOS2 installation and does not change global package requirements.

Provide assets before launching; the model does not download them:

- `VLLM_ZONOS2_DAC_PATH`: local `weights_44khz_8kbps_0.0.1.pth`.
  Without the variable, lookup checks `<model>/dac_44khz.pth`, then
  `~/.cache/descript/dac/weights_44khz_8kbps_0.0.1.pth`.
- `VLLM_ZONOS2_SPEAKER_PATH`: complete local
  `marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B` snapshot, including its
  configuration and modeling Python files. Without the variable, only
  already cached Hugging Face assets are considered.
- `VLLM_ZONOS2_TN_CACHE_DIR`: writable NeMo grammar cache owned by the operator.
  NeMo 1.2.0 builds its local FARs lazily; it does not download grammars.
- `HF_MODULES_CACHE`: a writable, operator-owned dynamic-module cache when
  reference speaker conditioning uses the trusted local model code.

DAC loads on the first codec decode (including runtime profiling).
The speaker encoder loads on the first reference request and runs on CPU,
so it does not allocate another GPU. There is no silent waveform fallback
for missing codec weights or missing dependencies.

## Speech request fields

`input` is required. `voice` is `default` or omitted. `language` accepts
English, Chinese, French, German, Spanish, Italian, Portuguese, Japanese
and Korean, plus explicit language codes such as `en_us`, `en_gb`, `cmn`,
`fr_fr`, `de`, `es`, `it`, `pt_br`, `ja` and `ko`. Omission uses English;
`Auto` is rejected because this frontend has no language detector.

`speed` uses the official speaking-rate buckets. Omission leaves the original
no-rate prompt unchanged. `extra_params.speaking_rate` (UTF-8 bytes/second)
or `speaking_rate_bucket` can be used instead; the three controls are
mutually exclusive. This changes model conditioning, rather than resampling.

`extra_params.quality_buckets` supplies integer indices and
`quality_values` supplies physical values. Both accept an ordered list or
a feature-name object, and are mutually exclusive. Feature order is
`lufs`, `estimated_snr`, `max_pause`, `estimated_bandlimit_hz`,
`leading_silence_s`, `trailing_silence_s`. Missing quality controls retain
the official `trailing_silence_s=3` bucket default.

`ref_audio` accepts one reference, resolved through the serving media
policy, and produces a 2048D Qwen3 speaker embedding. Alternatively provide
a finite 2048D `speaker_embedding`; both fields are mutually exclusive.
No reference transcript is needed. Optional boolean extra parameters are
`clean_speaker_background`, `accurate_mode`, and `text_normalization`.

`emotion_cfg_scale`/`cfg_scale` only accept 1.0. Emotion/style instructions,
other CFG values and unknown parameters are explicitly rejected. Stage 0
sampling controls and `seed` retain the P3 request-local behavior.

HTTP streaming uses `response_format` PCM or WAV. Non-streaming formats
use the common serving encoders. Native sample rate is 44,100 Hz.

## State and validation

The producer sends cumulative raw-code snapshots; `meta.num_processed_tokens`
is the output frame target and `meta.last_chunk` marks the codec terminal.
These fields survive the framework's runtime metadata rewrite. The decoder
keys overlap state by runner request ID, suppresses replayed chunks, and
clears buffers/tails on finish, cancellation or codec failure.

CPU boundary/dependency tests live in `test_codec.py` and adapter tests in
`test_zonos2_tts_adapter.py`. GPU2 component validation replays the previous
P3 codes against frozen official DAC/OLA and speaker implementations.
Runtime validation uses controlled talker logits and actual DAC weights.
This verifies P4 components; it does not replace P5 end-to-end acceptance
or P6 audio quality evaluation.

The DAC dependency's TorchScript Snake activation can change numerically
after its first profiling call. Exact reference comparisons warm both
implementations first. Conditioning helpers derive from the frozen MIT
ZONOS2 source at `194c0a3a`; the original notice is in `LICENSE.zonos2`.

## Optional single-row router replay

`VLLM_ZONOS2_ROUTER_REPLAY=1` opts into CUDA replay of the Sonic EDA router
for contiguous BF16 single-row inference. It is disabled by default. Only
the router linear/activation/norm/softmax operations are captured; attention,
KV metadata, sampling, RNG, EOS and request state stay on the existing path.
The full AR graph guard still requires `enforce_eager: true`, synchronous
AR scheduling and no prefix cache. Batched/prefill/CPU/training shapes use
the existing implementation. Capture failures propagate. Each router owns
its static inputs/outputs and invalidates captures on device/dtype moves.
Consumers must finish using replay outputs before the next invocation.

```bash
export VLLM_ZONOS2_ROUTER_REPLAY=1
# Then use the existing serve/API/benchmark commands with an idle single GPU.
pytest -sv tests/model_executor/models/zonos2/test_router_replay_gpu.py
```

## C4 numerical sensitivity

The fixed seed42 Chinese `zh_01` cap failure is reproducible on A40 with
128-token chunked prefill. Single-request and C4 histories match through
the first ten frames, then BF16 batched numerical differences change one
codebook draw. Saved-logit replay reproduces that draw; the hidden row map
is intact. The resulting trajectory can remain silent without EOA.
Larger prefill can avoid this cap but did not pass the per-sample CER gate.
Keep the B1 recommendation; do not hide failures by changing the frozen
sampling defaults or by treating cap removal as speech-quality qualification.
Default precision settings and the explicit incomplete-generation error remain.
