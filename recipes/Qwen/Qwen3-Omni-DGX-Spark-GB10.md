# Qwen3-Omni on NVIDIA DGX Spark (GB10)

> **Qualification revision:** vLLM-Omni `f1a6e7ce` with vLLM 0.29.0
> on one NVIDIA DGX Spark GB10 (aarch64). The pinned deployment YAML and a
> clean-source build from the official vLLM base passed text/audio inference
> checks. This does not qualify Realtime VAD, native Duplex, or other releases.
>
> **Current main:** Not validated. This YAML targets the pinned
> `f1a6e7ce` deployment schema; compatibility with current main must
> be checked separately.

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

Stage 0 uses `max_model_len: 8192`, which limits the total prompt
length, including multimodal tokens. Longer audio and image/video
inputs may exceed this limit. Increasing it requires checking KV-cache
capacity and memory headroom under the GB10 Stage 0
`gpu_memory_utilization: 0.52` setting. The safe maximum above 8192 is
unknown; no higher limit was qualified. Do not increase it without a new
KV-cache and memory-headroom test on GB10.

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
- Earlier independently built image: `vllm-omni:gb10-f1a6e7ce-repro-final`
  (historical local qualification tag).
- Newly reproduced and inference-tested candidate: `vllm-omni:gb10-repro-check-20261030132325-3382504`
  (local Docker tag; not an upstream-published image).
- Source: clean Git archive of upstream commit `f1a6e7ce`.
- Python: 3.12.3; vLLM: 0.29.0; vLLM-Omni: `0.0.0.dev0+gf1a6e7ce`.
- PyTorch: `2.13.0+cu130`; CUDA runtime: 13.0.
- Transformers: 5.14.1.
- Dependency overlay: 61 pinned Python packages, including two wheels
  locally built from source distributions.

The overlay retains several packages normally associated with development
or verification tooling because they are unconditional transitive
dependencies in the pinned distribution metadata:

- `s3tokenizer==0.3.0` requires `pre-commit`, which requires `cfgv`,
  `identify`, `nodeenv`, and `virtualenv`.
- `kernels==0.16.1` requires `sigstore>=4,<5` and `tomlkit>=0.13.3`.
  Sigstore introduces dependencies including `tuf`, `rfc3161-client`,
  and `sigstore-rekor-types`.

These dependencies were checked against installed package metadata on
the qualified GB10 image. Their inclusion preserves the declared
dependency closure for the offline `--no-deps` installation; it does
not establish that every package is imported by Qwen3-Omni inference.
Removing them requires separate validation or a change to the upstream
dependency declarations.

An earlier local image used a preinstalled vLLM-Omni environment and
was not independently reproducible from the public base. The build below
starts from the official vLLM image, exports unmodified pinned upstream
source, installs its dependency overlay offline, and has now passed
text and audio inference on the same GB10 host. It does not claim
bit-for-bit reproducibility of Docker layers or externally sourced wheels.

### Reproduced image build

Use a checkout containing this recipe on an ARM64 DGX Spark with the official
`vllm/vllm-openai:v0.29.0` image available locally. Model weights are not
part of the image. The commands below run in a child Bash process and do not
change the calling shell's options. The package index defaults to PyPI; on a
network requiring a mirror, set `PIP_INDEX_URL` (for example, to
`https://pypi.tuna.tsinghua.edu.cn/simple`) before running the block.

```bash
bash <<'GB10_BUILD'
build_gb10_image() {
  local repo ctx base commit image base_rc built_rc
  repo="$(git rev-parse --show-toplevel)" || return 1
  base=vllm/vllm-openai:v0.29.0
  commit=f1a6e7ce2a8fceb1417ec86b0b49a6386601ac65
  image="vllm-omni:gb10-repro-check-$(date +%Y%m%d%H%M%S)-$$"
  ctx="$(mktemp -d "$HOME/gb10-repro-build.XXXXXX")" || return 1
  echo "Build context: $ctx"
  mkdir -p "$ctx/src" "$ctx/wheelhouse" "$ctx/sdists" "$ctx/verified" || return 1

  docker image inspect "$base" >/dev/null 2>&1 || return 1
  git -C "$repo" cat-file -e "$commit^{commit}" 2>/dev/null ||
    git -C "$repo" fetch origin "$commit" || return 1
  git -C "$repo" archive --format=tar -o "$ctx/src.tar" "$commit" || return 1
  tar -xf "$ctx/src.tar" -C "$ctx/src" || return 1
  rm "$ctx/src.tar" || return 1
  cp "$repo/recipes/Qwen/Qwen3-Omni-DGX-Spark-GB10.Dockerfile" \
    "$ctx/Dockerfile" || return 1
  cp "$repo/recipes/Qwen/Qwen3-Omni-DGX-Spark-GB10-wheel-pins.txt" \
    "$ctx/wheel-pins.txt" || return 1
  grep -vE '^(#|$|antlr4-python3-runtime==|openai-whisper==)' \
    "$ctx/wheel-pins.txt" > "$ctx/binary-pins.txt" || return 1

  # Forward proxy variables only when configured.
  local -a proxy_opts=()
  local key
  for key in HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY \
             http_proxy https_proxy all_proxy no_proxy; do
    if [ -n "${!key:-}" ]; then proxy_opts+=(-e "$key=${!key}"); fi
  done

  docker run --rm --pull never --network host \
    -e "PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.org/simple}" \
    "${proxy_opts[@]}" -v "$ctx:/work" --entrypoint bash "$base" -lc '
      python3 -m pip download --only-binary=:all: --no-deps \
        -r /work/binary-pins.txt -d /work/wheelhouse &&
      python3 -m pip download --no-deps --no-binary=:all: \
        --no-build-isolation -d /work/sdists \
        antlr4-python3-runtime==4.9.3 openai-whisper==20250625 &&
      python3 -m pip wheel --no-index --no-deps --no-build-isolation \
        --wheel-dir /work/wheelhouse /work/sdists/*.tar.gz
    ' || return 1

  # Resolve every pinned requirement from the offline wheelhouse.
  docker run --rm --pull never --network none -v "$ctx:/work" \
    --entrypoint python3 "$base" -m pip download --no-index --no-deps \
    --find-links /work/wheelhouse -r /work/wheel-pins.txt \
    -d /work/verified || return 1

  local expected actual
  expected="$(grep -cE '^[^#[:space:]]' "$ctx/wheel-pins.txt")" || return 1
  actual="$(find "$ctx/wheelhouse" -maxdepth 1 -name '*.whl' | wc -l)"
  if [ "$actual" -ne "$expected" ]; then
    echo "ERROR: wheel count mismatch ($actual != $expected)" >&2
    return 1
  fi

  # Keep only files required by Dockerfile COPY instructions.
  rm -rf "$ctx/sdists" "$ctx/verified" || return 1
  rm -f "$ctx/wheel-pins.txt" "$ctx/binary-pins.txt" || return 1

  docker build --pull=false --network=none -f "$ctx/Dockerfile" \
    -t "$image" "$ctx" || return 1

  docker run --rm --pull never --network none --entrypoint python3 \
    "$base" -m pip check > "$ctx/base-check.txt" 2>&1
  base_rc=$?
  docker run --rm --pull never --network none --entrypoint python3 \
    "$image" -m pip check > "$ctx/image-check.txt" 2>&1
  built_rc=$?

  python3 - "$ctx/base-check.txt" "$ctx/image-check.txt" \
    "$base_rc" "$built_rc" <<'CHECK' || return 1
from pathlib import Path
import re
import sys

def read(path, rc):
    lines = {
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip() and line.strip() != "No broken requirements found."
    }

    if rc not in (0, 1) or (rc == 0 and lines) or (rc == 1 and not lines):
        raise SystemExit(f"Unexpected pip check result: rc={rc}, {path}")

    for line in lines:
        requirement = re.fullmatch(
            r"^.+ has requirement .+, but you have .+\.?$", line
        )
        platform_warning = re.fullmatch(
            r"^.+ is not supported on this platform\.?$", line
        )
        if not (requirement or platform_warning):
            raise SystemExit(
                f"Unexpected pip check output; possible Docker error: {line}"
            )

    return lines

base = read(sys.argv[1], int(sys.argv[3]))
image = read(sys.argv[2], int(sys.argv[4]))
new = image - base
if new:
    raise SystemExit(f"New dependency findings: {sorted(new)}")
print("No additional pip check findings compared with the base image.")
CHECK
  echo "Build validated (not inference-qualified): $image"
  echo "Build context: $ctx"
}

if build_gb10_image; then
  echo "Image build and dependency checks passed."
else
  echo "ERROR: build or dependency checks failed; existing images remain unchanged." >&2
  false
fi
GB10_BUILD
```

The updated build procedure was executed end-to-end on an ARM64
DGX Spark GB10. It generated a 61-wheel dependency overlay
(approximately 243 MiB) and successfully built the candidate image.

The rebuilt image's `pip check` output was identical to that of the
official base image. The following two pre-existing findings remain:

- PyTorch requires `nvidia-nccl-cu13==2.29.7`, but the base image
  contains `nvidia-nccl-cu13==2.30.7`.
- `nvidia-cusparselt-cu13==0.8.1` reports that it is unsupported
  on this platform.

No additional dependency findings were introduced by the overlay.
The candidate image subsequently passed text and audio inference
on the same GB10 host. Results are recorded in the independent
image verification section below.

## Command

The validated one-device profile is stored at
[`Qwen3-Omni-DGX-Spark-GB10.yaml`](Qwen3-Omni-DGX-Spark-GB10.yaml).
It is derived from the previous three-stage profile by removing the legacy
`duplex_session.server_vad_model_path` block, which the Chat Completions workload
does not need.

The checked-in profile, including Stage 1 `seed: 42`, was mounted
in the seed-42 verification container. The host and container SHA256
checksums both returned:

```text
1a1bff44d2e00412a501e567fac33d877b0c60c30875462e0055afaf39dec7c5
```

The older text-request benchmark and first audio-input run used the profile
with the VAD field, so do not present those measurements as clean-profile results.

The SHA256 above identifies the checked-in profile, including Stage 1
`seed: 42`. The previous SHA256

`800a037b26fc37032d1b6ba3b825ad473ee6d7eb6f38c3ea4d356b713c4bb2d5`

identifies the pre-seed profile and is retained only as a historical reference.

Docker image tags in this recipe are local names, not upstream version
identifiers. Earlier validation used `gb10-f1a6e7ce-repro-final` and
`gb10-f1a6e7ce-closure-verified`. The independently rebuilt candidate
uses `vllm-omni:gb10-repro-check-20261030132325-3382504`.

The build script generates a timestamped candidate tag. Use the exact
tag printed by the build when launching the candidate image. If a stable
local alias is preferred, create one explicitly with:

Set `CANDIDATE_IMAGE` to the exact timestamped tag printed by the
build command, then create the stable alias:

`CANDIDATE_IMAGE="<candidate-image-tag>"; docker image tag "$CANDIDATE_IMAGE" vllm-omni:gb10-f1a6e7ce-repro-final`

Replace `<candidate-image-tag>` with the actual tag printed by the
build. This command only creates a local alias; it does not rebuild
or modify the candidate image.

The launch command below uses this stable alias. Before running it,
execute the tagging command above. Alternatively, replace the image
reference in the launch command with the exact timestamped candidate tag.
The alias is a local Docker tag and does not change the image contents.

Replace `/path/to/model` and `/path/to/profile` with local paths:

```bash
# Verify the container image is locally available and record its provenance.
docker image inspect vllm-omni:gb10-f1a6e7ce-repro-final >/dev/null

# Validated GB10 profile; use the named file in this recipe.
# Requires the alias above; otherwise change both image references to the printed candidate tag.
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

## Historical latency and memory (earlier local image)

These are historical measurements from **one** DGX Spark and the earlier
local qualification image, with a single concurrent HTTP request. They are
not measurements from the independently rebuilt `repro-final` image,
Realtime TTFA, or throughput.

The row marked `prior profile` used the earlier profile containing
`duplex_session.server_vad_model_path`. The clean-profile measurements
below are also from the earlier local image. Use the separate independently
rebuilt image verification table for reproducible-image measurements.

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

| Image | Input | Recorded complete-response time(s) |
| --- | --- | --- |
| `repro-test` | Text | 38.775 s; 1.736 s, 1.610 s (warm repeats) |
| `repro-test` | WAV | 41.079 s; 2.078 s, 2.100 s (warm repeats) |
| `repro-final` | Text | 36.284 s |
| `repro-final` | WAV | 38.960 s |
| `gb10-repro-check-20261030132325-3382504` | Text | 1.787 s |
| `gb10-repro-check-20261030132325-3382504` | WAV | 40.088 s (first audio-input request); 2.041 s (repeat) |

The candidate text request was measured after earlier text/audio regression
requests had already been served by the same running container. It is not a
cold-start measurement. The WAV measurements represent the first and repeated
audio-input requests in the candidate regression sequence.
The first WAV request (40.088 s) is the first audio-input-path observation,
not a cold server start, and is not directly comparable to the warm text
request (1.787 s).

The newly reproduced candidate was
`vllm-omni:gb10-repro-check-20261030132325-3382504` with image ID `sha256:3bed8329050ebd4dff1daffbdb9a2676d96fe6266ca934cf9ff5c4c731367001`.
It was built from the pinned upstream source commit
`f1a6e7ce2a8fceb1417ec86b0b49a6386601ac65` and the official
`vllm/vllm-openai:v0.29.0` ARM64 base image.

The candidate container mounted the seed-42 GB10 YAML.
The host and container configuration SHA256 both matched
`1a1bff44d2e00412a501e567fac33d877b0c60c30875462e0055afaf39dec7c5`.

The text request returned HTTP 200, Chinese text and a valid
4.137 s speech output. Its complete HTTP response time was 1.787 s.
This request was measured after earlier regression requests on the same
running container, not during a cold start.

The audio-input fixture contained the spoken codename "Orchid Falcon".
Both audio-input requests returned the expected Thinker text.
The first audio-input request in this sequence took 40.088 s and
generated 4.297 s of speech. The repeated audio-input request took
2.041 s and generated 3.977 s of speech.

All three generated WAV files were validated as 24 kHz mono PCM16.

The timings measure complete HTTP responses, not time to first audio.
The first and repeated audio requests are individual observations,
not a throughput benchmark. The earlier `repro-test` warm timings
were measured before the final `onnx`/`pre-commit` dependency additions.
Neither multi-turn history nor Realtime duplex was tested.

### Stage 1 sampling seed validation (2026-10-30)

The GB10 recipe now sets `seed: 42` for Stage 1 (Talker),
matching the upstream sampling configuration fix in #7474.

The updated profile was tested on a single DGX Spark GB10 using
`vllm-omni:gb10-f1a6e7ce-closure-verified`, without rebuilding
the image. The host and container SHA256 matched:

```text
1a1bff44d2e00412a501e567fac33d877b0c60c30875462e0055afaf39dec7c5
```

| Request | Completion time | Result |
| --- | ---: | --- |
| Text to text + audio | 38.430 s | Passed |
| WAV to text + audio, first run | 39.544 s | Passed |
| Same WAV, repeated | 2.239 s | Passed |

All three requests returned the expected text and valid
24 kHz mono PCM16 speech. Both audio-input requests returned the
expected Thinker response mentioning "Orchid Falcon".

Whisper Small transcribed the first generated audio as
"RG 的 FACON" and the second as "Orchid Falcon".
Manual listening found both outputs intelligible, with the
first codename pronunciation less clear than the second.
No obvious speech degeneration was observed.

The two audio outputs differed in duration (3.497 s versus
4.617 s) and SHA256. These observations establish functional
regression coverage, not bitwise deterministic audio or a measured
reduction in Talker degeneration rate.

Times measure complete HTTP responses, not time to first audio.
The original pre-seed qualification measurements above remain
historical observations and must not be relabeled as seeded results.

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
