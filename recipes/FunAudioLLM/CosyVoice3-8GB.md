# CosyVoice3 with an 8GB GPU

## Summary

| Field | Value |
| --- | --- |
| Vendor / model | FunAudioLLM / `Fun-CosyVoice3-0.5B-2512` |
| Task / mode | Voice-cloned TTS through the online speech API; complete WAV responses |
| Hardware | One CUDA GPU with 8188MiB of device memory, under WSL2 |
| Maintainer | [LinzeShi](https://github.com/LinzeShi) |

## When to use this recipe

Use this profile as a starting point for short English or Chinese speech
requests on an 8GB CUDA GPU, one at a time. It uses the existing two-stage
PyTorch pipeline without quantization, CPU weight offload, or TensorRT.

## Supported model contract

| Item | Contract of this profile |
| --- | --- |
| Entrypoint | `vllm serve <model> --omni`; `POST /v1/audio/speech` |
| Inputs | Text, one reference recording, and its matching transcript; no built-in voice presets |
| Output | Complete mono 24kHz WAV; duration depends on the generated utterance |
| Deployment | Both stages on GPU 0, TP=1, concurrency 1, eager execution; talker context limit 4096 tokens |
| Precision | BF16 engine dtype; the model's HiFT vocoder runs in FP32 |
| Coverage | Three short prompts, submitted twice per run; long-form and higher concurrency are not qualified |

The pipeline streams codec chunks internally. This recipe only measures
complete HTTP responses; it does not qualify streaming playback continuity.

## References

- [Canonical model card](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- [Upstream CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- [Omni CosyVoice3 guide at the measured revision](https://github.com/vllm-project/vllm-omni/blob/a96e3bcb90c81aeb2fd187dc59644b367cb4a32e/docs/user_guide/examples/online_serving/text_to_speech.md#cosyvoice3)
- [Existing speech client](../../examples/online_serving/text_to_speech/cosyvoice3/speech_client.py)
- [Consumer-GPU qualification RFC, A1](https://github.com/vllm-project/vllm-omni/issues/10008)

## Hardware

| Component | Tested value |
| --- | --- |
| GPU | 1x CUDA GPU, 8188MiB; no inter-GPU communication |
| Host RAM | 32GB installed; approximately 20GiB assigned to WSL |
| Scope | One server, sequential requests; other GPUs and multiple resident models are not qualified |

The measurements below describe this setup, not performance guarantees for
other GPUs with the same memory capacity.

## Software environment

| Component | Tested value |
| --- | --- |
| OS | Windows 11 Home build 26200; WSL 2.7.12.0, kernel 6.18.33.2; Ubuntu 22.04.4 |
| Driver / Python | NVIDIA 616.92; Python 3.12.4 |
| Runtime | vLLM 0.33.0; Torch 2.14.0+cu130; Transformers 5.19.0; cuDNN 9.25.1 |
| Audio dependencies | ONNX Runtime 1.31.1; s3tokenizer 0.3.0 |
| Omni revision | `a96e3bcb90c81aeb2fd187dc59644b367cb4a32e` |
| Model revision | `29e01c4e8d000f4bcd70751be16fa94bf3d85a18` |

Use a fresh environment with the matching
[CUDA source installation](../../docs/getting_started/installation/gpu.md#build-wheel-from-source),
including its pinned cuDNN runtime exports. Run commands from the repository
root at the measured revision above. Keep the checkout and cache in WSL's
Linux filesystem. WSL uses the Windows GPU driver; do not install a Linux
display driver inside WSL.

## Command

Download only the weights used by this pipeline and the tokenizer metadata.
The alternative RL weights, the base Qwen weights, and TensorRT estimator
files are not needed for this configuration.

```bash
export HF_HUB_DISABLE_XET=1
COSY_MODEL=$(hf download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --revision 29e01c4e8d000f4bcd70751be16fa94bf3d85a18 \
  --include 'config.json' --include 'configuration.json' \
  --include 'cosyvoice3.yaml' --include 'llm.pt' --include 'flow.pt' \
  --include 'hift.pt' --include 'campplus.onnx' \
  --include 'CosyVoice-BlankEN/*.json' --include 'CosyVoice-BlankEN/*.txt' \
  --max-workers 1 --quiet)
```

The S3 speech tokenizer downloads its own approximately 1GB ONNX checkpoint
to `~/.cache/s3tokenizer` on first use. With s3tokenizer 0.3.0 this comes
from ModelScope, even when the model weights came from Hugging Face. Allow
network access for the first launch. Its expected SHA-256 is
`23236a74175dbdda47afc66dbadd5bcb41303c467a57c261cb8539ad9db9208d`.
Loading and converting this tokenizer also uses host RAM.

Create a local overlay; the generic deploy configuration remains unchanged:

```bash
cat > cosyvoice3-8gb.yaml <<'YAML'
base_config: vllm_omni/deploy/cosyvoice3.yaml
stages:
  - stage_id: 0
    gpu_memory_utilization: 0.3
    max_num_seqs: 1
    max_num_batched_tokens: 4096
    max_model_len: 4096
    enforce_eager: true
  - stage_id: 1
    max_num_seqs: 1
YAML

export COSYVOICE3_TRT=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
vllm serve "$COSY_MODEL" --omni \
  --host 127.0.0.1 --port 8091 --served-model-name cosyvoice3 \
  --deploy-config cosyvoice3-8gb.yaml
```

Both stages inherit `devices: "0"`. Stage 1 retains its 0.2 memory budget
and eager mode. The pipeline keeps its default RAS sampling and 15-frame
codec chunks. `COSYVOICE3_TRT=0` prevents optional TensorRT engines and their
additional downloads; `VLLM_USE_FLASHINFER_SAMPLER=0` avoids sampler JIT
compilation when `nvcc` is unavailable.

The smaller talker budget leaves more room for the codec, reference-audio
processing, CUDA contexts, and the Windows desktop. A stage's budget is not
a cap on the complete pipeline's memory. Check `nvidia-smi` and `free -h`
before launch, and stop other model servers first.

## Verification

In another terminal, wait for health to return HTTP 200, then request audio:

```bash
curl --fail http://127.0.0.1:8091/health
curl --fail-with-body http://127.0.0.1:8091/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "cosyvoice3",
    "input": "Hello, this is a local speech test.",
    "ref_audio": "https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc/asset/zero_shot_prompt.wav",
    "ref_text": "希望你以后能够做的比我还好呦。",
    "response_format": "wav",
    "seed": 42
  }' --output cosyvoice3.wav
```

The public reference file's SHA-256 is
`c7b31d6dbe7cc6a716dded00550db5b50940bf209e424e4ad207b12e657c8ff6`.
Measurements use the same bytes in a base64 data URL to exclude reference
downloads from request latency. The existing
[speech client](../../examples/online_serving/text_to_speech/cosyvoice3/speech_client.py)
also accepts a local reference file.

Activate the serving environment and check the downloaded response:

```bash
python -c 'import soundfile as sf; a, sr = sf.read("cosyvoice3.wav"); assert len(a) > 0 and sr == 24000; print(sr, len(a) / sr)'
```

Listen to the result to assess intelligibility and voice similarity. A valid,
finite, non-silent waveform alone does not establish perceptual quality.

Measured with concurrency 1 and seed 42 at the revision above. The three prompts
below were each submitted twice, in order, with the same reference audio.
All six requests returned HTTP 200 and finite, non-silent, mono 24kHz WAV
data. These are complete-response measurements, not streaming time to first audio.

| Prompt | First pass: request / audio duration | Second pass: request / audio duration |
| --- | --- | --- |
| `Hello, this is a local speech test.` | 3.00s / 2.72s | 1.82s / 2.72s |
| `你好，这是一个本地语音测试。` | 1.86s / 3.08s | 1.67s / 3.08s |
| `The model is running inside Windows Subsystem for Linux on a single graphics card.` | 3.90s / 4.88s | 2.82s / 4.88s |

The table uses the ordinary CLI and the 12 selected model files. Readiness
took 68.2s with model, tokenizer, and earlier kernel caches present. Subsequent
request RTFs (request time / audio duration) were 0.54–0.80. An instrumented
run of the same profile took 154.4s to start and had RTFs of 1.21–1.60.
Cache state and measurement setup differed, so this is not a controlled
speedup comparison or evidence of sustained real-time performance.

## Notes

| Talker budget | Pre-launch GPU usage | Sampled whole-device peak | Minimum available host RAM | Result |
| --- | --- | --- | --- | --- |
| 0.4 | 828MiB | 7724MiB | 1651MiB | 6/6 requests succeeded |
| 0.3, instrumented | 820MiB | 6907MiB | 2266MiB | 6/6 requests succeeded |
| 0.3, ordinary CLI | 820MiB | 6907MiB | 2238MiB | 6/6 requests succeeded |

No OOM occurred. All six corresponding WAV files were byte-identical across
these three runs, which checks these inputs rather than general determinism
or perceptual quality. The 0.4 run included the first S3 tokenizer download
and conversion; its 208.4s startup is not comparable to cached launches.

Device usage includes the desktop baseline and was sampled once per second,
so shorter peaks can be missed. In the instrumented 0.3 run, the talker's
PyTorch peak allocated/reserved memory was 2269.58/2346MiB and code2wav's was
1925.35/2078MiB. These counters were reset after readiness and queried between
requests through a local measurement extension. They exclude startup,
frontend, and non-PyTorch allocations; their peaks need not be simultaneous.
Allocated memory is part of reserved memory, so do not add the two.

The low host-RAM headroom above is an observed limit, not a recommended
reserve. Leave more free RAM before reproducing, especially for initial
S3 tokenizer conversion. The default reference-conditioning cache remains
enabled; no cache, CPU weight offload, or quantization comparison is claimed.

## Supported features

| Feature | Status in this hardware profile | Shared guide |
| --- | --- | --- |
| Voice cloning | Tested with reference audio and transcript | [Speech API](../../docs/serving/speech_api.md) |
| Deployment overlay | Two stages on GPU 0, eager, one request at a time | [Stage configuration](../../docs/configuration/stage_configs.md) |
| HTTP streaming | Not measured; complete WAV responses only | [Speech API](../../docs/serving/speech_api.md) |
| Quantization | Not enabled or qualified | [Quantization](../../docs/user_guide/quantization/overview.md) |
| Sleep/wake | Not qualified | [Sleep mode](../../docs/features/sleep_mode.md) |
