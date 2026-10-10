# Qwen3-Omni on NVIDIA DGX Spark (GB10)

> **Qualification revision:** locally built image with vLLM-Omni package commit
> `f1a6e7ce`, vLLM 0.29.0, on a single DGX Spark. The exact GB10 YAML below
> has been verified by matching host/container SHA256 hashes and two successful
> audio-input requests. This is not a claim of support on every upstream release.
> The exact local image build has not been reconstructed; a public upstream
> Docker build path is documented below but is not yet re-qualified on GB10.

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

- Runtime: locally built Docker image
  `vllm-omni:0.29.0-aarch64-main-f1a6e7ce-m1` (**not a published upstream image**).
- Source baseline: `vllm-omni` commit `f1a6e7ce`, as identified by package
  metadata and Docker build history. The final installation used an existing
  dependency environment; a clean rebuild was not completed, and the absence
  of local source patches has not been independently established.
- Container Python: **3.12.3**.
- Container vLLM: **0.29.0**.
- Container vLLM-Omni package: **0.0.0.dev0+gf1a6e7ce**.
- Container PyTorch: **2.13.0+cu130**; PyTorch CUDA runtime: **13.0**.
- Container Transformers: **5.14.1**.
- Container image was built locally. Verify build instructions and absence of local source patches before publishing; the package version alone is not proof of an unmodified upstream build.

The host-side `vllm-omni-pr7759` Python environment is **not** the runtime
of this deployment; its package versions must not be substituted here.

### Image provenance and build limitations

The qualified image was built locally and is not an official published image.
Its Docker history shows that it was derived from a previously prepared
vLLM-Omni image and that the final source installation used:

```bash
VLLM_OMNI_VERSION_OVERRIDE=0.0.0.dev0+gf1a6e7ce uv pip install --no-build-isolation --no-deps --reinstall "."
```

The `--no-deps` flag reused dependencies already installed in the parent
image. This final installation step does not constitute a reproducible
installation from a clean environment.

At the pinned revision, the upstream `docker/Dockerfile.cuda` uses
`vllm/vllm-openai:v0.29.0` as its base image. An attempted fresh rebuild
required an explicit version override because the Docker build context did
not provide usable Git metadata. After resolving the version, the build
failed while downloading `nvidia-cusparselt-cu13==0.8.1` due to a network
timeout.

Consequently, the reported inference results qualify the pinned local
image and YAML profile, not a successfully rebuilt clean upstream image.
A fresh installation must resolve the required dependencies separately
and repeat the verification tests.

### Fresh image build (not yet qualified)

The validated local image is not publicly distributed. To prepare a
fresh image from the pinned upstream source, use the following procedure
on an ARM64 CUDA host with access to the required Python packages:

```bash
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
git checkout f1a6e7ce

sed '/^RUN cd .*vllm-omni && uv pip install/i ARG VLLM_OMNI_VERSION_OVERRIDE=0.0.0.dev0+gf1a6e7ce'   docker/Dockerfile.cuda > /tmp/Dockerfile.gb10

docker build --pull=false   -f /tmp/Dockerfile.gb10   --build-arg BASE_IMAGE=vllm/vllm-openai:v0.29.0   -t vllm-omni:gb10-f1a6e7ce .
```

This procedure was attempted but did not complete because downloading
an ARM64 CUDA dependency timed out. It is therefore an **unverified
fresh-build path**, not the image used for the measurements below.
If this build succeeds, substitute `vllm-omni:gb10-f1a6e7ce` in the
serving command and re-run the verification tests.

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
docker image inspect vllm-omni:0.29.0-aarch64-main-f1a6e7ce-m1 >/dev/null

# Validated GB10 profile; use the named file in this recipe.
docker run --rm --name qwen3-omni-gb10 \
  --gpus all --ipc host --network host \
  -v /path/to/model:/models/Qwen3-Omni-30B-A3B-Instruct:ro \
  -v /path/to/profile:/workspace/gb10.yaml:ro \
  vllm-omni:0.29.0-aarch64-main-f1a6e7ce-m1 \
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

Measurements below are from **one** DGX Spark, the locally built image named
above, and a single concurrent HTTP request. They are not Realtime TTFA or
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
- The validated clean profile's SHA256 was checked both on the host and inside
  the container. Rebuild and re-qualify the public upstream Dockerfile before publication;
  disclose any local changes if they are still required.

## Supported features

| Feature | This GB10 profile | Reference |
| --- | --- | --- |
| Single-device text + speech serving | Verified on pinned build | [Online serving](../../examples/online_serving/qwen3_omni/README.md) |
| Audio input with spoken reply | Verified on pinned build | [Online client](../../examples/online_serving/openai_chat_completion_client_for_multimodal_generation.py) |
| `/v1/realtime` automatic turns | Not qualified | [Qwen realtime tracking](https://github.com/vllm-project/vllm-omni/issues/7055) |
| Interruption and tool calling | Not qualified | [Qwen realtime tracking](https://github.com/vllm-project/vllm-omni/issues/7055) |
