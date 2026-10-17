# Qwen3-Omni on NVIDIA DGX Spark (GB10)

> **Qualification revision:** vLLM-Omni `f1a6e7ce` with vLLM 0.29.0
> on one NVIDIA DGX Spark GB10 (aarch64). The pinned deployment YAML and a
> clean-source build from the official vLLM base passed text/audio inference
> checks. This does not qualify Realtime VAD, native Duplex, or other releases.

## Summary

- Vendor: Qwen
- Model: `Qwen/Qwen3-Omni-30B-A3B-Instruct`
- Task: Online text-to-speech and speech-to-speech chat
- Mode: OpenAI-compatible `/v1/chat/completions`, batch response (not a Realtime VAD qualification)
- Hardware: 1x NVIDIA DGX Spark GB10 (aarch64, unified memory)
- Maintainer: Community

## When to use this recipe

Use this profile to serve Qwen3-Omni's three stages (Thinker, Talker and
Code2Wav) together on a **single DGX Spark GB10**. The default multi-GPU
model configuration is not suitable as-is for the single-GPU topology.
The profile has `max_num_seqs: 1` for all three stages and prioritizes
single-request operation over concurrency.

## Supported model contract

| Input | Output | Entry point | Qualification |
| --- | --- | --- | --- |
| Text | Chinese text and 24 kHz speech | `/v1/chat/completions` | Verified, five measured requests |
| 16 kHz mono PCM WAV | Chinese text and 24 kHz speech | `/v1/chat/completions` | Verified, two requests after a clean-profile restart |
| Live microphone / VAD | Streaming voice conversation | `/v1/realtime` | Not qualified by this recipe |
| Image or video | Multimodal answer | `/v1/chat/completions` | Not measured on this GB10 profile |

These are observations for the pinned build and workloads below, not claims
about every model input combination or a current upstream release.

## References

- Model: <https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct>
- Generic Qwen3-Omni recipe: [Qwen3-Omni.md](Qwen3-Omni.md)
- Online serving example: [Qwen3-Omni example](../../examples/online_serving/qwen3_omni/README.md)
- Community recipe tracker: <https://github.com/vllm-project/vllm-omni/issues/2645>

## Hardware

| Component | Validated host |
| --- | --- |
| CPU architecture | aarch64 |
| Accelerator | 1x NVIDIA GB10 (integrated GPU) |
| Memory | 128 GB unified memory nominal; about 121 GiB shown by `free -h` |
| OS | Ubuntu 24.04.4 LTS |
| Driver | 580.159.03 |
| Host CUDA Toolkit | 13.0.88 |

The GB10's GPU and CPU share physical system memory. The GPU memory field
reported by `nvidia-smi` may be `[N/A]`; use system memory measurements and
runtime allocator measurements with their definitions clearly stated.

## Software environment

- Base image: `vllm/vllm-openai:v0.29.0` (aarch64).
- Independently built image: `vllm-omni:gb10-f1a6e7ce-repro-final`
  (local qualification tag; not an upstream-published image).
- Source: clean Git archive of upstream commit `f1a6e7ce`.
- Python: 3.12.3; vLLM: 0.29.0; vLLM-Omni: `0.0.0.dev0+gf1a6e7ce`.
- PyTorch: `2.13.0+cu130`; CUDA runtime: 13.0.
- Transformers: 5.14.1.
- Dependency overlay: 61 pinned Python packages, including two wheels
  locally built from source distributions.

An earlier local image used a preinstalled vLLM-Omni environment and
was not independently reproducible from the public base. The build below
starts from the official vLLM image, exports unmodified pinned upstream
source, installs its dependency overlay offline, and has now passed
text and audio inference on the same GB10 host. It does not claim
bit-for-bit reproducibility of Docker layers or externally sourced wheels.

### Reproduced image build

Use a checkout containing this recipe (the PR branch before merge; the
upstream branch after merge). Docker and the ARM64
`vllm/vllm-openai:v0.29.0` image must be available locally. The model
weights are separate and are **not** part of the image build.

First export the pinned upstream code and prepare the build context:

```bash
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
RECIPE_DIR="$REPO_ROOT/recipes/Qwen"
CTX="$HOME/gb10-repro-build"
PINNED_COMMIT=f1a6e7ce2a8fceb1417ec86b0b49a6386601ac65

# Ensure the historical pinned commit is available even in a shallow clone.
git -C "$REPO_ROOT" cat-file -e "$PINNED_COMMIT^{commit}" || \
  git -C "$REPO_ROOT" fetch origin "$PINNED_COMMIT"
mkdir -p "$CTX/src" "$CTX/wheelhouse" "$CTX/sdists"
git -C "$REPO_ROOT" archive "$PINNED_COMMIT" | tar -xf - -C "$CTX/src"
cp "$RECIPE_DIR/Qwen3-Omni-DGX-Spark-GB10.Dockerfile" "$CTX/Dockerfile"
cp "$RECIPE_DIR/Qwen3-Omni-DGX-Spark-GB10-wheel-pins.txt" "$CTX/wheel-pins.txt"

# Two packages require building wheels from source distributions.
grep -vE '^(#|$|antlr4-python3-runtime==|openai-whisper==)' \
  "$CTX/wheel-pins.txt" > "$CTX/binary-pins.txt"
```

Prepare all 61 wheels using the Tsinghua PyPI mirror, without a proxy.
Downloads happen in a temporary container; this does not change the
installed packages of the base image:

```bash
docker run --rm --pull never --network host \
  -v "$CTX:/work" --entrypoint bash vllm/vllm-openai:v0.29.0 -lc '
    set -euo pipefail
    unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
    unset PIP_PROXY pip_proxy
    export NO_PROXY="*" no_proxy="*"
    python3 -m pip download --only-binary=:all: --no-deps \
      -i https://pypi.tuna.tsinghua.edu.cn/simple \
      -r /work/binary-pins.txt -d /work/wheelhouse
    python3 -m pip download --no-deps --no-binary=:all: \
      --no-build-isolation \
      -i https://pypi.tuna.tsinghua.edu.cn/simple \
      -d /work/sdists \
      antlr4-python3-runtime==4.9.3 openai-whisper==20250625
    python3 -m pip wheel --no-index --no-deps --no-build-isolation \
      --wheel-dir /work/wheelhouse /work/sdists/*.tar.gz
  '
test "$(find "$CTX/wheelhouse" -maxdepth 1 -name '*.whl' | wc -l)" -eq 61
```

Build without network access using the adjacent Dockerfile, whose install
steps use `--no-deps` to preserve vLLM, PyTorch and CUDA from the base:

```bash
docker build --pull=false --network=none \
  -f "$CTX/Dockerfile" \
  -t vllm-omni:gb10-f1a6e7ce-repro-final "$CTX"
# The base image has two known pip-check findings.
# Verify that no additional dependency problems were introduced.
set +e
docker run --rm --pull never --network none --entrypoint python3 \
  vllm-omni:gb10-f1a6e7ce-repro-final -m pip check \
  > "$CTX/pip-check.txt" 2>&1
check_rc=$?
set -e

cat "$CTX/pip-check.txt"
test "$check_rc" -eq 1
test "$(wc -l < "$CTX/pip-check.txt")" -eq 2
grep -Fq 'nvidia-nccl-cu13==2.29.7' "$CTX/pip-check.txt"
grep -Fq 'nvidia-cusparselt-cu13 0.8.1 is not supported' \
  "$CTX/pip-check.txt"

```

This sequence was executed on a DGX Spark GB10. The full wheelhouse
contains 61 packages (approximately 243 MiB). Following the final
build, `pip check` reported **only two existing base-image issues**:

- PyTorch requires `nvidia-nccl-cu13==2.29.7`, but the base contains 2.30.7.
- `nvidia-cusparselt-cu13==0.8.1` reports an unsupported platform.

The source of those two warnings was verified against the unchanged
upstream vLLM base; there were no additional missing Python dependencies.
The supplied pin list reproduces the tested package versions but does not
include independent download hashes for all wheel artifacts.

## Command

The validated one-device profile is stored at
[`Qwen3-Omni-DGX-Spark-GB10.yaml`](Qwen3-Omni-DGX-Spark-GB10.yaml).
It is derived from the previous three-stage profile by removing the legacy
`duplex_session.server_vad_model_path` block, which the Chat Completions workload
does not need. The running container mounted this exact file and reported
`/health` healthy; the host and container SHA256 checksums both returned:

```text
800a037b26fc37032d1b6ba3b825ad473ee6d7eb6f38c3ea4d356b713c4bb2d5
```

The older text-request benchmark and first audio-input run used the profile
with the VAD field, so do not present those measurements as clean-profile results.

Replace `/path/to/model` and `/path/to/profile` with local paths:

```bash
# Verify the container image is locally available and record its provenance.
docker image inspect vllm-omni:gb10-f1a6e7ce-repro-final >/dev/null

# Validated GB10 profile; use the named file in this recipe.
docker run --rm --name qwen3-omni-gb10 \
  --gpus all --ipc host --network host \
  -v /path/to/model:/models/Qwen3-Omni-30B-A3B-Instruct:ro \
  -v /path/to/profile:/workspace/gb10.yaml:ro \
  vllm-omni:gb10-f1a6e7ce-repro-final \
  vllm serve /models/Qwen3-Omni-30B-A3B-Instruct \
  --omni --port 8093 --deploy-config /workspace/gb10.yaml
```

Check readiness:

```bash
curl -fsS http://127.0.0.1:8093/health
```

## Verification

For a short text request with both text and speech output:

```bash
curl -fsS http://127.0.0.1:8093/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"/models/Qwen3-Omni-30B-A3B-Instruct",
    "messages":[{"role":"user","content":"请用一句简短的中文介绍人工智能。"}],
    "modalities":["text","audio"]
  }' -o /tmp/qwen3-omni-response.json
```

The response contains separate text and speech choices. Do not interpret
`choices[0].message.audio == null` as missing speech; inspect all choices.
Save an audio choice's `message.audio.data` (Base64 WAV bytes) and play it
to check speech intelligibility.

For audio input, encode a mono PCM16 16 kHz WAV as a
`data:audio/wav;base64,...` URL. Here is a self-contained verification command
using only the Python standard library (run on the serving host):

```bash
python - /path/to/input_16k_mono.wav <<'PYCODE'
import base64
import json
import sys
import urllib.request
from pathlib import Path

source = Path(sys.argv[1])
payload = {
    "model": "/models/Qwen3-Omni-30B-A3B-Instruct",
    "messages": [{"role": "user", "content": [
        {"type": "audio_url", "audio_url": {"url":
            "data:audio/wav;base64," + base64.b64encode(source.read_bytes()).decode()}},
        {"type": "text", "text": "请用中文回答这段录音中的问题。"},
    ]}],
    "modalities": ["text", "audio"],
}
req = urllib.request.Request(
    "http://127.0.0.1:8093/v1/chat/completions",
    data=json.dumps(payload, ensure_ascii=False).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=300) as response:
    result = json.load(response)
for choice in result.get("choices", []):
    message = choice["message"]
    if message.get("content"):
        print(message["content"])
    if isinstance(message.get("audio"), dict) and message["audio"].get("data"):
        Path("gb10_audio_reply.wav").write_bytes(
            base64.b64decode(message["audio"]["data"])
        )
        print("Saved gb10_audio_reply.wav")
PYCODE
```

Verify the returned text corresponds to the spoken input and listen to the
output WAV. The verified utterance yielded a relevant Chinese answer and
24 kHz mono output; the input test fixture itself is not redistributed here.
For the repository's more general payload builder, see the
[online client](../../examples/online_serving/openai_chat_completion_client_for_multimodal_generation.py).

## Measured latency and memory

Measurements below are from **one** DGX Spark, the earlier local
qualification image, and a single concurrent HTTP request. They are not Realtime TTFA or
throughput numbers.

| Workload | Samples | Client completion time | Result |
| --- | ---: | ---: | --- |
| Warm-up text input -> text + audio | 1 | 1.806 s | Both modalities returned |
| Short Chinese text input -> text + audio | 5 | Median 1.682 s; range 1.612–1.771 s | 5/5 returned both modalities |
| WAV audio input -> text + audio, prior profile | 1 | 9.556 s | Text + 3.497 s, 24 kHz mono WAV |
| WAV audio input -> text + audio, first request after clean-profile restart | 1 | 75.292 s | Text + 4.217 s, 24 kHz mono WAV |
| Same WAV audio input -> text + audio, repeat without restart | 1 | **2.016 s** | Text + 3.977 s, 24 kHz mono WAV |

The first clean-profile audio request took 75.292 s, but the repeat on the
same running container took 2.016 s. This is consistent with one-time startup
work (e.g., compilation or warm-up), but the exact cause was not proven from
logs. Neither result is TTFA; both measure complete HTTP response time.

The text-input request asked for a one-sentence Chinese introduction to AI.
The audio-input fixture was mono PCM16 16 kHz. The two workloads are different
and should **not** be compared as if input sizes were equal.

After the model had loaded, the benchmark observed approximately **30.1 GiB**
`MemAvailable` throughout five short requests. This is whole-system remaining
memory, **not** peak model allocation, and does not capture the startup peak.
The script sampled `/proc/meminfo` at 0.5 s intervals. Other host workloads
may change this value.

### Independently rebuilt image verification

The following requests ran with the same pinned GB10 YAML and one
concurrent `/v1/chat/completions` HTTP request. Latencies are end-to-end
complete-response times, **not** time-to-first-audio.

| Image | Input | First measured request | Subsequent warm requests |
| --- | --- | ---: | --- |
| `repro-test` | Text | 38.775 s | 1.736 s, 1.610 s |
| `repro-test` | WAV | 41.079 s | 2.078 s, 2.100 s |
| `repro-final` | Text | 36.284 s | Not measured |
| `repro-final` | WAV | 38.960 s | Not measured |

Both rebuilt images returned Chinese text and valid 24 kHz mono speech.
The audio-input fixture contained the spoken codename "Orchid Falcon",
which the model recovered correctly. Long first-request times are
consistent with one-time warm-up, but the precise cause was not
independently established. The warm timings were measured on `repro-test`
before the final `onnx`/`pre-commit` dependency additions, not on
`repro-final`. Neither multi-turn history nor Realtime duplex was tested.

## Notes

- The working three-stage memory-utilization configuration was 0.52 / 0.09 /
  0.06 for Thinker, Talker and Code2Wav. These are configuration settings,
  **not** additive measurements of actual allocation.
- All three stages share GPU device `0`. Stage 0/1 use eager execution; Stage 2
  uses `enforce_eager: false` and `max_num_batched_tokens: 65536`.
- Streaming output is enabled in the deploy configuration, but the HTTP
  benchmarks above measured **fully completed non-streaming responses**.
- Tool calling, client interruption, Server VAD and native duplex are outside
  this hardware qualification.
- The validated profile's SHA256 was checked on the host and in the
  container. The independent build uses unmodified pinned upstream
  source and the documented dependency overlay.

## Supported features

| Feature | This GB10 profile | Reference |
| --- | --- | --- |
| Single-device text + speech serving | Verified on pinned build | [Online serving](../../examples/online_serving/qwen3_omni/README.md) |
| Audio input with spoken reply | Verified on pinned build | [Online client](../../examples/online_serving/openai_chat_completion_client_for_multimodal_generation.py) |
| `/v1/realtime` automatic turns | Not qualified | [Qwen realtime tracking](https://github.com/vllm-project/vllm-omni/issues/7055) |
| Interruption and tool calling | Not qualified | [Qwen realtime tracking](https://github.com/vllm-project/vllm-omni/issues/7055) |
