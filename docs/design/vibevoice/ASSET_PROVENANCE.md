# VibeVoice bundled voice asset provenance

> **Audit date:** 2026-08-25
>
> **Status:** `RESOLVED — CANONICAL APACHE-2.0 PROJECT ASSETS REUSED`
>
> **Scope:** `vllm_omni/model_executor/models/vibevoice/assets/default_{0..3}.wav`

## Decision

The original four VibeVoice default references had no traceable source or asset
license. They have been replaced with byte-for-byte copies of canonical
reference/default voice assets already distributed by vLLM-Omni and their
source projects under Apache-2.0.

No clip is transcoded, trimmed, normalized, or otherwise transformed. The
packaged SHA-256 therefore matches the canonical source SHA-256. VibeVoice
resamples references at request-processing time without modifying the packaged
files.

The machine-readable manifest shipped in the wheel is:

```text
vllm_omni/model_executor/models/vibevoice/ASSET_MANIFEST.json
```

It records source revisions, paths, hashes, media properties, licenses,
attribution, intended-use evidence, and approved reuse scopes.

## Inventory

| Slot | Packaged file | Canonical project asset | SHA-256 | License |
| ---: | --- | --- | --- | --- |
| 0 | `default_0.wav` | `tests/assets/cosyvoice3/zero_shot_prompt.wav` | `c7b31d6dbe7cc6a716dded00550db5b50940bf209e424e4ad207b12e657c8ff6` | Apache-2.0 |
| 1 | `default_1.wav` | `vllm_omni/model_executor/models/step_audio2/assets/default_female.wav` | `5fc92ddcd9bc9af10437d9630642378777a98fc260f16508a9777db12c830a41` | Apache-2.0 |
| 2 | `default_2.wav` | `tests/assets/indextts2/ref_audio.wav` | `e33e6ee0107a1dd58e1d66dd90c13df3d55a8683047cc3d7ea206dad84ed3fc8` | Apache-2.0 |
| 3 | `default_3.wav` | `tests/assets/qwen3_tts/clone_2.wav` | `480f55f41c71c3d79c2a9acc48f0bfb3c5a46222e6e9ebf3e2888e93501a6b5c` | Apache-2.0 |

| Slot | Duration | Source rate | Channels | Encoding |
| ---: | ---: | ---: | ---: | --- |
| 0 | 3.480000 s | 24 kHz | 1 | WAV IEEE float |
| 1 | 9.042250 s | 48 kHz | 1 | WAV PCM 16-bit |
| 2 | 2.438708 s | 48 kHz | 1 | WAV IEEE float |
| 3 | 8.080000 s | 24 kHz | 1 | WAV IEEE float |

Every clip is mono and shorter than the 60-second per-reference limit.

## Sources and attribution

### Slot 0 — CosyVoice zero-shot prompt

- vLLM-Omni introduction:
  [`b1ff69502920df1f65f2f62c7e661169eb2bca65`](https://github.com/vllm-project/vllm-omni/commit/b1ff69502920df1f65f2f62c7e661169eb2bca65)
- upstream revision:
  [`QwenAudio/CosyVoice@074ca6dc`](https://github.com/QwenAudio/CosyVoice/tree/074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc)
- upstream file:
  [`asset/zero_shot_prompt.wav`](https://github.com/QwenAudio/CosyVoice/blob/074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc/asset/zero_shot_prompt.wav)
- license:
  [Apache-2.0](https://github.com/QwenAudio/CosyVoice/blob/074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc/LICENSE)
- attribution: CosyVoice contributors.

The source project publishes the clip as its zero-shot reference prompt, and
vLLM-Omni already uses it for CosyVoice reference conditioning.

### Slot 1 — Step-Audio2 default female prompt

- vLLM-Omni introduction:
  [`9bc8705c98788ca1429c55e512383b559afcec12`](https://github.com/vllm-project/vllm-omni/commit/9bc8705c98788ca1429c55e512383b559afcec12)
- upstream revision:
  [`stepfun-ai/Step-Audio2@76e272b5`](https://github.com/stepfun-ai/Step-Audio2/tree/76e272b56c3917a8d7188f18bbb5a65dfc8a0845)
- upstream file:
  [`assets/default_female.wav`](https://github.com/stepfun-ai/Step-Audio2/blob/76e272b56c3917a8d7188f18bbb5a65dfc8a0845/assets/default_female.wav)
- license:
  [Apache-2.0](https://github.com/stepfun-ai/Step-Audio2/blob/76e272b56c3917a8d7188f18bbb5a65dfc8a0845/LICENSE)
- attribution: Step-Audio2 contributors.

The source project and vLLM-Omni already ship the exact bytes as the default
female speaker prompt for Step-Audio2.

### Slot 2 — IndexTTS2 voice 01

- vLLM-Omni introduction:
  [`044240d8ebeae9c19a7be01db9d37a1cd1a57c8a`](https://github.com/vllm-project/vllm-omni/commit/044240d8ebeae9c19a7be01db9d37a1cd1a57c8a)
- upstream Space revision:
  [`IndexTeam/IndexTTS-2-Demo@b01840e8`](https://huggingface.co/spaces/IndexTeam/IndexTTS-2-Demo/tree/b01840e8e4fd9753743a6d0466cd73ae1d634a68)
- upstream file:
  [`examples/voice_01.wav`](https://huggingface.co/spaces/IndexTeam/IndexTTS-2-Demo/blob/b01840e8e4fd9753743a6d0466cd73ae1d634a68/examples/voice_01.wav)
- license:
  [Apache-2.0](https://huggingface.co/spaces/IndexTeam/IndexTTS-2-Demo/blob/b01840e8e4fd9753743a6d0466cd73ae1d634a68/LICENSE)
- attribution: IndexTTS contributors.

The Apache-2.0 demo Space publishes the clip as `voice_01.wav` for reference
voice cloning. vLLM-Omni's `ref_audio.wav` contains the same bytes.

### Slot 3 — Qwen3-TTS clone 2

- vLLM-Omni introduction:
  [`b1ff69502920df1f65f2f62c7e661169eb2bca65`](https://github.com/vllm-project/vllm-omni/commit/b1ff69502920df1f65f2f62c7e661169eb2bca65)
- official source:
  [`clone_2.wav`](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav)
- Qwen3-TTS audited revision:
  [`QwenLM/Qwen3-TTS@022e286b`](https://github.com/QwenLM/Qwen3-TTS/tree/022e286b98fbec7e1e916cb940cdf532cd9f488e)
- license:
  [Apache-2.0](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/LICENSE)
- attribution: Qwen3-TTS contributors.

Qwen publishes the clip as an official voice-cloning reference. The immutable
vLLM-Omni commit and SHA-256 pin the bytes because the object-storage URL does
not contain a revision.

## Identity and intended use

vLLM-Omni does not assign a name, demographic description, or claimed identity
to any bundled reference. The slot numbers are implementation indices only.
Each source project publishes its clip specifically as a default speaker prompt
or reference-conditioning/voice-cloning example. Reuse for VibeVoice preserves
that purpose without claiming that the voice belongs to the framework.

Users providing their own references remain responsible for obtaining all
rights and consent required for those recordings. Explicit `ref_audio` and
uploaded `voice` values never mix with the bundled defaults.

## Excluded candidate

`tests/assets/glm_tts/jiayan_zh.wav` was considered but deliberately excluded.
Its upstream prompt-audio license is CC BY-NC 4.0 and states that commercial use
is strictly prohibited. It is therefore not suitable as a general-purpose
runtime default in the vLLM-Omni wheel.

## Verification and update rule

Tests verify that:

1. all four packaged files match their manifest SHA-256;
2. slot numbers and filenames are unique and ordered;
3. every file is decodable, mono, non-empty, and no longer than 60 seconds;
4. VibeVoice resolves each file to finite 24 kHz mono samples;
5. the built wheel contains all four WAVs and this manifest.

Any future replacement must update the manifest and this audit in the same
change. A filename-only replacement or transformed file without a documented
source hash is not acceptable.
