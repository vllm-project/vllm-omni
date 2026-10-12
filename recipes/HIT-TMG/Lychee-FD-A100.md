# Lychee-FD on one NVIDIA A100

## Scope

This recipe starts native Unified Duplex + MRV2 serving for Lychee-FD on one GPU,
with a separate native Token2Wav stage. Functional migration testing used an
A100 80 GiB, eager execution and a local release checkpoint. The public download
instructions below follow the model owner's current distribution layout; that
public checkpoint has not yet been qualified by the migration's GPU runs.

Install this branch of vLLM-Omni using the
[installation guide](../../docs/getting_started/installation/README.md) and a
matching vLLM/PyTorch runtime. The validation environment used vLLM 0.31.0,
Torch 2.13.0+cu132 and a CUDA 12.9 toolkit. The captured development runtime is
not a published, independently reproduced installation recipe. Do not install
the released demo's older patched vLLM into this serving environment.

## Download checkpoints

Choose a model root on your machine. The
[official model card](https://huggingface.co/HIT-TMG/Lychee-FD) now places the
Lychee-FD checkpoint files at the repository root, rather than inside a remote
`lychee_full_duplex/` subdirectory.

```bash
export MODEL_ROOT="${HOME}/models/lychee-fd"
mkdir -p "${MODEL_ROOT}"
hf download HIT-TMG/Lychee-FD \
  --local-dir "${MODEL_ROOT}/lychee_full_duplex"
hf download stepfun-ai/Step-Audio-2-mini \
  --include 'token2wav/*' --local-dir "${MODEL_ROOT}"
```

The resulting layout is:

```text
lychee-fd/
  lychee_full_duplex/
    config.json
    model.safetensors.index.json
    model-00001-of-00006.safetensors
    ...
    tokenizer_config.json
  token2wav/
    flow.yaml
    flow.pt
    hift.pt
    campplus.onnx
    speech_tokenizer_v2_25hz.onnx
```

Follow the model repositories' licenses. Select a mono 24 kHz prompt WAV to
configure the speaker. The owner's demo includes
[a default male prompt](https://github.com/HITsz-TMG/Lychee-FD/blob/main/frontend/public/clone_24k_mono/default_male.wav);
the migration runs used that prompt. Place your selected prompt locally and set
`LYCHEEFD_T2W_PROMPT_WAV` to its absolute path.

## Start the server

Run from the vLLM-Omni checkout with the serving environment activated:

```bash
export CUDA_VISIBLE_DEVICES=0
export LYCHEEFD_TOKEN2WAV_PATH="${MODEL_ROOT}/token2wav"
export LYCHEEFD_T2W_PROMPT_WAV="${MODEL_ROOT}/speaker_24k_mono.wav"
export LYCHEEFD_TTS_VOCODER_HOP_SIZE=10

vllm-omni serve "${MODEL_ROOT}/lychee_full_duplex" \
  --omni --trust-remote-code \
  --deploy-config vllm_omni/deploy/lychee_fd_single_gpu.yaml \
  --host 127.0.0.1 --port 8099
```

CUDA 12.9 is an optional toolchain for reproducing the model-local arithmetic
used by the tested A100 runtime. Ordinary serving uses PyTorch activations when
the matching runtime or default libdevice file is unavailable. To select the
compatible AR audio-kernel toolchain in that runtime, set:

```bash
export CUDA_HOME=/usr/local/cuda-12.9
export LYCHEE_RELEASED_LIBDEVICE_PATH="${CUDA_HOME}/nvvm/libdevice/libdevice.10.bc"
```

An explicitly configured missing libdevice path is a configuration error in the
matching AR runtime; remove the override to permit the default fallback. Kernel
compilation and execution errors still propagate. The F0 decoder's optional
compatibility path uses the tested default CUDA 12.9 location and its own runtime
qualification; it does not consume the AR path override. CPU fallback checks do
not qualify alternative GPU stacks for audio quality or performance. CUDA toolkit
and packaged PyTorch CUDA runtime versions are separate parts of the environment.

The C1 profile keeps the default AR memory-utilization budget of 0.70. Before
sharing the GPU, account for both stages, KV and transient activations. Functional
shared-GPU runs instead used a derived C1 profile with 0.35 utilization and an
explicit 2 GiB AR KV budget; those runs do not qualify the default profile's
resource sizing or realtime performance. C2/C4 profiles use explicit 4 GiB AR KV
budgets and configure both stages for the selected session count. Select them
with `--deploy-config vllm_omni/deploy/lychee_fd_c2_single_gpu.yaml` or the C4
counterpart, and leave eager execution enabled.

## Connect and verify

```bash
curl --fail http://127.0.0.1:8099/health
curl --fail http://127.0.0.1:8099/v1/models
```

Use `ws://127.0.0.1:8099/v1/realtime?duplex=1` and the existing
[Realtime Duplex client and wire contract](../../docs/serving/realtime_duplex_api.md).
Use mono 16 kHz PCM16 input and read mono 24 kHz PCM16 output. The model works
in 400 ms input windows; a client may append smaller audio packets, while the
server assembles complete windows. Continue streaming during assistant speech.
Report actual playback with ACK events and inspect response IDs, chunk order
and final events when validating cancellation or interruptions.

See [the supported surface and validation limits](../../docs/models/lychee_fd.md).
This recipe makes no benchmark, realtime-SLA, public-weight-equivalence or
non-NVIDIA hardware claim. The standard CLI and full-answer checks must use the
checkpoint and profile that a deployment will actually serve.
