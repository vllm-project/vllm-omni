# SheetSage2 audio-to-score preprocessing — H200

## Summary

- Vendor: m-a-p
- Model: `m-a-p/SheetSage2` with the `m-a-p/MERT-v2-FullSong` parent
- Task: recording → ABC score, MIDI and timed music annotations
- Mode: standalone official Transformers inference; optional YuE2 request export
- Hardware: one NVIDIA H200 (141 GB)
- Maintainer: [@princepride](https://github.com/princepride)

## When to use this recipe

Use a recording as the musical score for a YuE2 cover, or inspect its melodies,
chords and structure. The tool calls SheetSage2's official `transcribe()` API.
It does not register SheetSage2 as a native vLLM model or start an Omni server.

The transcription tool works independently of YuE2. For score-conditioned
generation, use an Omni checkout containing the merged
[PR #7886](https://github.com/vllm-project/vllm-omni/pull/7886)
(`423f34326ed420e5acf0b1fb862a1b5ffb7e0fa7` or later). See the
[YuE2 recipe](YuE2-3B.md) for the generation model's runtime contract.

## Supported model contract

| Item | Contract |
| --- | --- |
| Input | One local recording decodable by FFmpeg; upstream converts it to mono, 24 kHz |
| Duration | Entire recording by default; `--max-seconds` limits preprocessing to a prefix |
| Full score | Default: melody voices and chords; exported request uses `cot=full` |
| Melody score | `--melody-only`: vocal and instrumental melodies, without score/playback chords; request uses `cot=melody` |
| Outputs | `score.abc`, `transcription.mid`, upstream annotations and `manifest.json` |
| YuE2 handoff | `--lyrics-file` and `--style` together also write `yue2_request.json` |
| Output directory | Must be new or empty; failed exports produce no YuE2 request |
| Deployment profile | One process, one H200, FP32 weight loading and BF16 inference autocast |

`--melody-only` does not isolate or clone a singer. The upstream model still
decodes its task annotations and retains those raw annotation files. YuE2
uses the exported score to condition newly generated audio; a transcription
does not guarantee an exact reproduction of the recording.

## References

- [Official SheetSage2 model card](https://huggingface.co/m-a-p/SheetSage2)
- [Pinned SheetSage2 source](https://huggingface.co/m-a-p/SheetSage2/tree/cafc0df1021e14f49e928c4b345f5959d414ef64)
- [Official YuE2 model card and cover documentation](https://huggingface.co/m-a-p/YuE2-3B)
- Tool: [`tools/sheetsage2_transcribe.py`](../../tools/sheetsage2_transcribe.py)
- Offline generation: [`examples/offline_inference/yue2/end2end.py`](../../examples/offline_inference/yue2/end2end.py)
- [Supported generation models](../../docs/models/supported_models.md)

## Hardware

- Accelerator: one NVIDIA H200, 141 GB; no multi-device interconnect required.
- Qualification: local M4A transcription, melody-only prefix, full-score
  whole-song export and one offline YuE2 cover. Run transcription and generation
  sequentially in separate processes. Other devices, CPU inference and
  performance scaling are not qualified by this recipe.

## Software environment

- Ubuntu 22.04, Python 3.10, NVIDIA driver 590.48.01.
- PyTorch/torchaudio 2.8.0 (CUDA 12.8 wheels), Transformers 4.45.2.
- FFmpeg 4.4.2 was used for the local M4A check; upstream recommends FFmpeg 6.1.
- vLLM: not required in the preprocessing environment.
- vLLM-Omni: tool developed against main `a038b38179e9c788d3af6a8e73f94652afb247e4`.
- YuE2 qualification: vLLM 0.30.0, PyTorch 2.13.0+cu130, Transformers 5.14.1;
  #7886 at `5cc8acc942e76fa6f12f8b2f2558c2663abc2aa5` with the FP32 VAE-loading
  and scalar `truncated` fixes subsequently included in the merged integration.

## Command

### Set up transcription

Run from this repository's root. Create a **separate environment**: the
official transcriber's Transformers version differs from Omni's serving
dependencies. FFmpeg must be on `PATH`.

```bash
uv venv --python 3.10 .venv-sheetsage2
uv pip install --python .venv-sheetsage2/bin/python \
  -r tools/requirements/sheetsage2.txt

.venv-sheetsage2/bin/python tools/sheetsage2_transcribe.py reference.wav \
  --trust-remote-code --device cuda:0 --melody-only \
  --output-dir outputs/reference-score
```

The default Hub checkpoint and its custom Python code are pinned to
`cafc0df1021e14f49e928c4b345f5959d414ef64`. Its configuration pins the MERT-v2
parent to `d8ba1c745e733b3908ce6ad16ebeb17ac7600a42`. Review that code before
passing `--trust-remote-code`. `--revision` can override the default pin;
custom Hub IDs use their default revision unless one is supplied.

Local snapshots must preserve this pinned source layout: `modeling_sheetsage2.py`
and its transitive relative-import dependencies are sibling `.py` files in the
snapshot root. The local module-cache workaround targets that flat layout;
reorganized snapshots with nested Python packages are not supported by it.

For offline use, download **both** model snapshots, including their Python,
JSON and safetensors files, before disconnecting:

```bash
.venv-sheetsage2/bin/hf download m-a-p/SheetSage2 \
  --revision cafc0df1021e14f49e928c4b345f5959d414ef64 \
  --local-dir models/SheetSage2
.venv-sheetsage2/bin/hf download m-a-p/MERT-v2-FullSong \
  --revision d8ba1c745e733b3908ce6ad16ebeb17ac7600a42 \
  --local-dir models/MERT-v2-FullSong
```

### Transcribe a full recording for a cover

Supply target lyrics in `lyrics.txt`, including YuE2 section tags such as
`[Verse]` and `[Chorus]`. Match the lyric phrasing to the reference melody.
This example retains the melody and chords of the whole recording:

```bash
.venv-sheetsage2/bin/python tools/sheetsage2_transcribe.py reference.wav \
  --model models/SheetSage2 --base-model-path models/MERT-v2-FullSong \
  --local-files-only --trust-remote-code --device cuda:0 \
  --lyrics-file lyrics.txt --style 'Chinese folk, guzheng, gentle vocals' \
  --seed 831001 --output-dir outputs/cover-score
```

Add `--melody-only` for a new arrangement without the original chords;
this also changes the exported request from `cot=full` to `cot=melody`.
For a short transcription smoke test, add `--max-seconds 45` and use lyrics
appropriate to that excerpt. Use a fresh output directory for each run.

### Generate offline

Switch to your **Omni environment** for generation. Download `m-a-p/YuE2-3B`
and `m-a-p/YuE2-Vae` to `models/YuE2-3B` and `models/YuE2-Vae`, respectively.
The offline entrypoint needs a local YuE2 directory containing `qwen.tiktoken`.
From the repository root, pass the exported ABC to the existing entrypoint:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/offline_inference/yue2/end2end.py \
  --model models/YuE2-3B --vae models/YuE2-Vae \
  --lyrics "$(cat lyrics.txt)" \
  --style 'Chinese folk, guzheng, gentle vocals' \
  --abc-file outputs/cover-score/score.abc --cot full \
  --seed 831001 --max-frames 9000 --gpu-memory-utilization 0.25 \
  --output cover.wav
```

Use `--cot melody` if the score was transcribed with `--melody-only`.
Changing `--cot` alone does not remove chords from an existing score.
The 9,000-frame budget permits up to 360 seconds at 25 frames/s; it does
not force that duration. The entrypoint's default 200-frame budget caps
audio at 8 seconds. Check that the final report says `truncated=False`.
The memory fraction above was used on one H200; it is not a measured
minimum memory requirement.

### Submit over HTTP (configuration-only)

The exported JSON can also be submitted to YuE2's speech endpoint. These
commands have not been qualified with an HTTP synthesis run in this recipe.
Start YuE2 in its **Omni environment**:

```bash
vllm serve m-a-p/YuE2-3B --omni --port 8091
```

Submit the exported file:

```bash
curl --fail-with-body http://localhost:8091/v1/audio/speech \
  -H 'Content-Type: application/json' \
  --data-binary @outputs/cover-score/yue2_request.json \
  --output cover.wav
```

The JSON contains the score itself, not a server-local file path. It sets
`stream=false`, requests WAV, and preserves an optional generation seed.
Use `--yue2-model` to match a custom served model name. The tool does not
submit the request or set a generation token budget; the server's own
duration and context limits apply.

## Verification

After a successful transcription, set `output` below to its output directory
(`outputs/reference-score` for the transcription-only example). The YuE2
request is checked when exported with `--lyrics-file` and `--style`:

```bash
.venv-sheetsage2/bin/python - <<'PY'
import json
from pathlib import Path
import mido

output = Path("outputs/cover-score")
abc = (output / "score.abc").read_text(encoding="utf-8")
manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
assert abc.strip()
request_path = output / "yue2_request.json"
if request_path.exists():
    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert request["extra_params"]["abc"] == abc
    assert request["extra_params"]["cot"] == ("melody" if manifest["melody_only"] else "full")
    print("YuE2 request verified")
midi = mido.MidiFile(output / "transcription.mid")
assert any(msg.type == "note_on" and msg.velocity for track in midi.tracks for msg in track)
print("ABC, MIDI and manifest verified")
PY
```

Run the CPU CLI contract tests in the repository's normal test
environment (with pytest and pytest-mock installed):

```bash
python -m pytest tests/tools/test_sheetsage2_transcribe.py -m 'core_model and cpu' -q
```

These tests cover request construction, loading options, validation and
failure handling with a mocked backend. They do not measure transcription
accuracy or generated cover similarity.

Local H200 checks with the pinned backend:

| Input / mode | Result |
| --- | --- |
| 45 s M4A prefix, melody-only | 579 ABC characters; parseable MIDI with 118 note-on events |
| 242.219 s M4A, full score | 3,033 ABC characters; parseable MIDI with 1,087 note-on events |
| Default Hub ID, 10 s prefix | Successful ABC and MIDI export |
| Local snapshots, fresh Transformers module cache | Successful offline full-score export |
| Both generated JSON requests, #7886 adapter at `5cc8acc942e76fa6f12f8b2f2558c2663abc2aa5` | Validation and prompt construction passed (919 / 2,821 prompt tokens) |
| Full score + new Mandarin lyrics, offline YuE2, `cot=full`, seed 831001, 9,000-frame budget | 241.319 s, 48 kHz stereo WAV; 6,034 generated tokens; `truncated=False` |

The offline cover used a 108 BPM Chinese traditional pop style and one H200
with `--gpu-memory-utilization 0.25`. The saved waveform contained finite
samples, peak amplitude 0.9501, RMS 0.1232 and no full-scale samples. This is
one functional run, not an accuracy or performance benchmark. It does not
establish perceptual equivalence to the recording. HTTP synthesis and
recordings longer than 300 seconds were not exercised. The local recording,
lyrics, generated scores and cover audio are not included in the repository.

## Notes

- Weights load in FP32 because upstream merges MERT adapters before inference
  autocast. `--dtype fp32` disables BF16 autocast.
- For local snapshots the tool prepares transitive Python dependencies in
  Transformers' module cache. This avoids the pinned Transformers version's
  direct-import-only copy behavior without editing upstream files.
- SheetSage2 manages long recordings with its own overlapping windows. This
  tool leaves windowing and notation generation to the official implementation.
- Failed runs may retain upstream diagnostic annotations. Choose a new output
  directory when retrying; no incomplete request is published for synthesis.
- Model weights have upstream license terms (CC BY-NC 4.0); this repository
  does not bundle weights or recordings.

## Supported features

| Feature | Status |
| --- | --- |
| Audio-to-score preprocessing | Official backend; local file input |
| YuE2 score-conditioned generation | Offline ABC handoff qualified on H200; HTTP request export validated, HTTP synthesis configuration-only |
| Native Omni inference, continuous batching, TP/PP/SP | Not implemented for SheetSage2 |
| Streaming transcription or cover audio | Not supported by this tool |
