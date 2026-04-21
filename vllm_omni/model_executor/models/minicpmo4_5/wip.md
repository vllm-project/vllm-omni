1. Official model implementation
Start here first. This is the closest thing to the “real” inference implementation.

modeling_minicpmo.py: main model logic, chat/streaming/TTS/duplex behavior
https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/main/modeling_minicpmo.py
processing_minicpmo.py: multimodal input packing and preprocessing
https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/main/processing_minicpmo.py
configuration_minicpmo.py: how the submodules are wired
https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/main/configuration_minicpmo.py
utils.py: helper functions used by the duplex and streaming demos
https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/main/utils.py
Model card examples: the runnable inference recipes
https://huggingface.co/openbmb/MiniCPM-o-4_5

## Test Status

### Offline Inference

Stage config:
- `vllm_omni/model_executor/stage_configs/minicpmo.yaml`

Verified:
- `text -> audio` works
- `text -> audio with ref_audio` works
- `image -> text` works
- `audio -> text` works
- `video -> text` works
- `image -> audio` works
- `video -> audio` works

Artifacts already saved locally:
- `minicpmo45_offline_pytest_artifacts/.../minicpmo45_text_image_to_audio.wav`
- `minicpmo45_offline_pytest_artifacts/.../minicpmo45_text_video_to_audio.wav`

Not done yet:
- offline thinker / talker debug dump for comparison

Offline `async_chunk` experiment:
- stage config: `vllm_omni/model_executor/stage_configs/minicpmo_async_chunk.yaml`
- prompt: `What is the capital of China? Answer in one short spoken sentence.`
- transcript artifact: `minicpmo45_offline_async_chunk_artifacts/.../minicpmo45_async_chunk_text_to_audio.txt`
- wav artifact: `minicpmo45_offline_async_chunk_artifacts/.../minicpmo45_async_chunk_text_to_audio.wav`
- debug artifacts:
  - `.../debug/minicpmo4_5_async_chunk/.../thinker_tts_chunks.jsonl`
  - `.../debug/minicpmo4_5_async_chunk/.../talker_codec_chunks.jsonl`

Current finding from the first offline async run:
- final thinker text is correct: `The capital of China is Beijing.`
- but `thinker2talker_async_chunk` currently dumped an empty emitted TTS-token stream
- stage summary also shows `stage 1 num_tokens_in = 0` while `stage 1 num_tokens_out = 55`
- so the first visible divergence is already at the thinker -> talker async handoff, before comparing against online

### Online Serving

Stage config:
- `vllm_omni/model_executor/stage_configs/minicpmo_async_chunk.yaml`

Verified at API / pipeline level:
- `text -> audio` request path works
- `image -> text` works
- `image -> audio` request path works
- `video -> text` works
- `video -> audio` request path works

Artifacts already saved locally:
- `minicpmo45_online_audio_artifacts/.../image_to_audio_001.wav`
- `minicpmo45_online_audio_artifacts/.../video_to_audio_001.wav`

Notes:
- online multimodal prompting required a MiniCPM-owned chat template with explicit OpenAI content-part handling
- online audio artifacts are useful for output inspection, but they do not expose thinker / talker intermediate outputs
- current online `async_chunk` audio quality is not yet considered correct
- so online `async_chunk` should be treated as "serving path passes" rather than "audio output validated"

### Current Gap

What is still missing:
- thinker text vs talker codec-token comparison under `async_chunk`
- apples-to-apples offline vs online comparison using the same prompt and, ideally, the same synthetic input payload
- fix offline thinker -> talker async handoff so stage 1 consumes non-empty thinker output
- fix online `async_chunk` audio correctness / quality regression

solid example should look like this: https://github.com/vllm-project/vllm-omni/pull/1516
