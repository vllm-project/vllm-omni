# Vevo2

> Online + offline zero-shot TTS with voice cloning (Amphion unified AR + flow-matching, 24 kHz)

## Summary

- Vendor: Amphion / CUHK-Shenzhen (checkpoint published as [`RMSnow/Vevo2`](https://huggingface.co/RMSnow/Vevo2))
- Model: `RMSnow/Vevo2`
- Task: Text-to-speech (zero-shot synthesis and reference-audio voice cloning)
- Mode: Online serving via the OpenAI-compatible `/v1/audio/speech` API, plus
  an offline end-to-end example
- Maintainer: Community

> **License**: the `RMSnow/Vevo2` checkpoint is **CC BY-NC-ND 4.0**
> (non-commercial, no-derivatives); the Amphion framework itself is MIT.
> Commercial deployment of the weights requires contacting the upstream
> authors. See the
> [Vevo2 license considerations](../../docs/serving/speech_api.md#vevo2-license-considerations)
> for the allowed/forbidden-use matrix and a production migration path.

## When to use this recipe

Use this recipe as a known-good starting point for serving `RMSnow/Vevo2` on
vLLM-Omni on a single 24 GB GPU. Vevo2 is Amphion's unified controllable
AR + flow-matching model; this integration runs the **MVP zero-shot TTS path**
(Qwen2.5-0.5B AR LM → 350M flow-matching transformer → Vocos vocoder, with a
Whisper-medium encoder used internally for reference-audio feature extraction)
as a single AR worker stage and emits 24 kHz mono audio. Singing-voice
synthesis, voice / singing style conversion, editing, melody control, and
`async_chunk` streaming are deferred to follow-up PRs (see
[#3391](https://github.com/vllm-project/vllm-omni/issues/3391)).

Vevo2 has **no built-in speaker presets** — every request must include a
reference clip (`ref_audio`); voice cloning is driven entirely by that clip.
The full pipeline fits well under 24 GB (peak ~7.55 GiB, see below), so no
tensor parallelism is required.

> **Setup is the most involved of any TTS model here.** Amphion is not on
> PyPI, and the published checkpoint ships no root `config.json`, so this
> recipe requires an Amphion clone on `PYTHONPATH` plus a one-time
> `init_vevo2_checkpoint.py` pass. Follow the prerequisites below exactly.

## References

- Example guide (offline, full prerequisites):
  [`examples/offline_inference/text_to_speech/README.md`](../../examples/offline_inference/text_to_speech/README.md#vevo2)
- Example guide (online):
  [`examples/online_serving/text_to_speech/README.md`](../../examples/online_serving/text_to_speech/README.md#vevo2)
- Server launch script:
  [`examples/online_serving/text_to_speech/vevo2/run_server.sh`](../../examples/online_serving/text_to_speech/vevo2/run_server.sh)
- Offline end-to-end script:
  [`examples/offline_inference/text_to_speech/vevo2/end2end.py`](../../examples/offline_inference/text_to_speech/vevo2/end2end.py)
- One-time checkpoint init:
  [`examples/offline_inference/text_to_speech/vevo2/init_vevo2_checkpoint.py`](../../examples/offline_inference/text_to_speech/vevo2/init_vevo2_checkpoint.py)
- Default deploy config:
  [`vllm_omni/deploy/vevo2.yaml`](../../vllm_omni/deploy/vevo2.yaml)
- Model / pipeline source:
  [`vllm_omni/model_executor/models/vevo2/`](../../vllm_omni/model_executor/models/vevo2/)
- Speech API reference:
  [`docs/serving/speech_api.md`](../../docs/serving/speech_api.md#vevo2)
- Upstream: [open-mmlab/Amphion — `models/svc/vevo2`](https://github.com/open-mmlab/Amphion/tree/main/models/svc/vevo2)

## Hardware Support

This recipe documents one tested single-GPU (24 GB-class) configuration.
Larger-VRAM and other-vendor (ROCm, NPU) sections are welcome as community
validation lands.

## GPU

### 1 x 24 GB-class GPU (Single GPU, Minimum Recommended)

The full Vevo2 pipeline fits comfortably on a single 24 GB GPU. The bundled
default config at
[`vllm_omni/deploy/vevo2.yaml`](../../vllm_omni/deploy/vevo2.yaml)
(`gpu_memory_utilization: 0.4`, `max_num_seqs: 4`, `enforce_eager: true`,
`trust_remote_code: true`) leaves real headroom: the Amphion pipeline is built
outside vLLM's allocator and peaks at **~7.55 GiB allocated / ~8.80 GiB
reserved**, so with `gpu_memory_utilization: 0.4` (~9.6 GiB reserved for
vLLM's own KV cache, which this wrapper barely uses) the total lands near
~18 GiB / 24 GiB.

#### Prerequisites

Amphion is not on PyPI and the published checkpoint has no root `config.json`:

```bash
# 1. Amphion on PYTHONPATH
git clone https://github.com/open-mmlab/Amphion.git
export PYTHONPATH=$PWD/Amphion:$PYTHONPATH
pip install -r Amphion/models/svc/vevo2/requirements.txt
# NOTE: do NOT pin transformers<5. Amphion builds LlamaConfig positionally,
# which transformers>=5 rejects, but vLLM-Omni installs an import-time shim
# (modeling_vevo2._shim_llama_config_positional_args) so the repo's pinned
# transformers>=5.5.3 works as-is.
pip install pyworld json5 praat-parselmouth torchcrepe ruamel.yaml

# 2. Checkpoint (CC BY-NC-ND 4.0 — non-commercial). Exclude training-only
#    artifacts so the download is ~6.5 GB rather than ~11 GB.
hf download RMSnow/Vevo2 --local-dir ./ckpts/Vevo2 \
    --exclude '*optimizer.pt' --exclude '*rng_state_*.pth' --exclude '*scheduler.pt' \
    --exclude '*trainer_state.json' --exclude '*training_args.bin' \
    --exclude '*optimizer.bin' --exclude '*scheduler.bin' --exclude '*random_states_*.pkl'

# 3. One-time: write a root config.json so vLLM-Omni's auto-dispatch resolves
#    model_type. Run from the repo checkout (this script imports
#    vllm_omni.model_executor.models.vevo2 — it is not standalone).
python examples/offline_inference/text_to_speech/vevo2/init_vevo2_checkpoint.py ./ckpts/Vevo2
```

#### Environment

- OS: Linux (Ubuntu 22.04)
- Python: 3.12
- transformers: >= 5.5.3 (positional-`LlamaConfig` shim applied at import)
- vLLM-Omni: current `main`
- Amphion: from source (open-mmlab/Amphion, on `PYTHONPATH`)

#### Command

Start the server from the repository root (with Amphion on `PYTHONPATH`):

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 \
vllm serve ./ckpts/Vevo2 --omni --host 0.0.0.0 --port 8092
```

or use the bundled launcher:

```bash
MODEL=./ckpts/Vevo2 PORT=8092 ./examples/online_serving/text_to_speech/vevo2/run_server.sh
```

The deploy config at
[`vllm_omni/deploy/vevo2.yaml`](../../vllm_omni/deploy/vevo2.yaml) is loaded
automatically (HF `model_type=vevo2`). Pass `--deploy-config <path>` to
override.

#### Verification

The numbers below were measured during maintainer review (end-to-end on an
L20X-class GPU, single card) against this recipe's exact `load_weights` +
`inference_ar_and_fm` call shape. Transcript quality was checked by running
Whisper over the generated audio.

**T1 — zero-shot synthesis (offline `end2end.py`)**:

```bash
python examples/offline_inference/text_to_speech/vevo2/end2end.py \
    --model ./ckpts/Vevo2 \
    --text "The quick brown fox jumps over the lazy dog while the morning sun rises slowly above the quiet river." \
    --ref-audio ./Amphion/models/vc/vevo/wav/arabic_male.wav \
    --ref-text "Philip stood undecided, his ears strained to catch the slightest sound."
```

Observed (real speech, not silence — RMS 0.155, 24 kHz mono):

| input | Whisper WER | audio duration | s/request | RTF |
|-------|------------:|---------------:|----------:|----:|
| long sentence  | **0.00** | 8.14 s | 2.56 | 0.314 |
| short sentence | 0.20¹    | 4.22 s | 1.69 | 0.400 |

¹ Every WER error in the short run is the invented proper noun "Vevo2"; on
real dictionary words there are zero errors.

**T2 — voice cloning (swap the timbre reference)**:

Swapping only the timbre reference moves the median output F0 from **119.6 Hz**
(male reference, source F0 123.4 Hz) to **239.8 Hz** (female reference, source
F0 229.8 Hz), confirming the cloning path conditions on the reference clip.

**T3 — online serving (`/v1/audio/speech`)**:

```bash
curl -X POST http://127.0.0.1:8092/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "model": "vevo2",
        "input": "Hello, this is Vevo2 served by vLLM-Omni.",
        "voice": "default",
        "ref_audio": "data:audio/wav;base64,...",
        "response_format": "wav"
    }' --output output.wav
```

`ref_audio` accepts a local path (auto-base64), an HTTP(S) URL, or a
`data:audio/wav;base64,...` data URI. The OpenAI `voice` field is required by
the schema but ignored unless it names an uploaded speaker. An empty `input`
or a missing `ref_audio` is rejected with a 400 (the model also raises rather
than emitting a silent WAV, so the offline path fails loudly too).

**Peak VRAM**: ~7.55 GiB allocated for the pipeline itself.

#### Notes

- **fp32 cast**: the pipeline currently casts all sub-models to fp32 on CUDA to
  avoid a `expected scalar type Float but found BFloat16` error seen on some
  drivers. Maintainer measurements found this cast costs ~1 GiB of VRAM and no
  measurable speed (fp32 ON 1.69 s/request vs OFF 1.67 s/request); dropping it
  is a tracked follow-up pending a re-measurement across drivers.
- **Output**: 24 kHz mono. When streaming PCM to a player, use `-r 24000`.
- **`--response-format`**: `wav` (default), `mp3`, `flac`, `pcm`.
- **No streaming yet**: the full waveform is returned in one response
  (single-shot batch mode); `async_chunk` streaming is a follow-up.
- **`trust_remote_code`**: set by `vevo2.yaml`; no extra flag needed.
- **Cold start**: ~14 s engine init + ~19 s Amphion pipeline load, then
  ~2 s/request. A missing `/opt/Amphion` (or Amphion not on `PYTHONPATH`)
  surfaces as an actionable ImportError at load time.
