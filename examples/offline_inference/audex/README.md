# Audex (Nemotron-Labs-Audex-2B) offline inference

One checkpoint, four tasks — each script builds its own `Omni` engine with
the right deploy yaml, so they run as plain `python <script>` with no extra
flags. Pass the HF repo ROOT as `--model` (default
`nvidia/Nemotron-Labs-Audex-2B`); per-stage subfolders and download subsets
resolve automatically.

| script | pipeline (`vllm_omni/deploy/<name>.yaml`) | audio in | text out | speech out | general audio out |
|---|---|---|---|---|---|
| `text_to_speech.py` | `audex_tts` | ❌ | ❌ | ✅ | ❌ |
| `text_to_audio.py` | `audex_tta` | ❌ | ❌ | ❌ | ✅ |
| `audio_qa.py` | `audex_thinker_only` | ✅ | ✅ | ❌ | ❌ |
| `speech_to_speech.py` | `audex_s2s` | ✅ | ✅ | ✅ | ❌ |

## text_to_speech.py — text → speech

Thinker generates `<speechcodec_N>` tokens; the streaming causal decoder
turns them into 16 kHz WAVs (one file per prompt, slugified filename).

```bash
# three built-in demo sentences -> results/audex_wavs/
python examples/offline_inference/audex/text_to_speech.py

# your own texts, with classifier-free guidance (official quality setting)
python examples/offline_inference/audex/text_to_speech.py \
    --texts "Hello there." "Nice to meet you." --cfg-scale 1.5 \
    --output-dir results/my_tts
```

Key flags: `--texts` / `--texts-file` (TSV `utt_id<TAB>text`),
`--cfg-scale` (default 1.0 = off; 1.5 recommended — guided requests run
one at a time, each paired with a length-matched null prompt),
`--temperature` (overrides the deploy yaml's stage-0 default).

## text_to_audio.py — caption → general audio (sound effects)

Same thinker over the interleaved 4-codebook `<audiocodec_N>` RVQ block,
decoded by the external XCodec1 checkpoint. CFG is effectively mandatory
(default scale 3.0). Clips are capped at `--codec-cap 4000` codec tokens
(10 s).

```bash
# three built-in demo captions -> results/audex_tta_wavs/
python examples/offline_inference/audex/text_to_audio.py

python examples/offline_inference/audex/text_to_audio.py \
    --captions "Thunder rolling across a valley." --output-dir results/my_tta
```

XCodec1 resolves from `--xcodec1-path`, the `XCODEC1_PATH` env var, or the
default `hf-audio/xcodec-hubert-general-balanced` repo (downloaded on
first use).

## audio_qa.py — audio (+ instruction) → text

Single-stage audio understanding on the full checkpoint (NV-Whisper
encoder + projector + LM). The default question is ASR
("Transcribe the input speech."); without `--audio-files`, vLLM's public
`mary_had_lamb` asset is transcribed.

```bash
# transcribe the built-in demo asset
python examples/offline_inference/audex/audio_qa.py

# your own clips / free-form QA
python examples/offline_inference/audex/audio_qa.py \
    --audio-files a.wav b.wav --question "What language is being spoken?"
```

## speech_to_speech.py — spoken question → spoken answer

The official three-pass cascade over ONE `audex_s2s` deployment: ASR
(audio → transcript, text-final) → chat (transcript → answer, text-final)
→ TTS (answer → speech through the streaming decoder). Only the TTS pass
touches the codec path. Without `--audio-file`, the `mary_had_lamb` asset
is used as the spoken input.

```bash
# full cascade on the built-in demo asset -> results/audex_s2s_answer.wav
python examples/offline_inference/audex/speech_to_speech.py

python examples/offline_inference/audex/speech_to_speech.py \
    --audio-file question.wav --output results/answer.wav --cfg-scale 1.5
```

`--cfg-scale` (default 1.5) applies to the TTS pass only; 1.0 disables it.

Online-serving counterparts (server + HTTP clients) live in
`examples/online_serving/audex/`.
