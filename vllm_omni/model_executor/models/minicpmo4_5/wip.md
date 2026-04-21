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

## MiniCPM-o 4.5 Async Talker Parity Debug

Reference files:
- `scripts/ref_generate_with_buffer.py`
- `scripts/modal_minicpmo45_hf_smoke.py`
- `vllm_omni/model_executor/stage_input_processors/minicpmo4_5.py`
- `vllm_omni/model_executor/models/minicpmo4_5/minicpmo4_5_talker.py`

Reference artifacts:
- HF talker chunks: `modal_compare_condition2/hf_artifacts/hf_talker_chunks.jsonl`
- HF raw talker condition dumps: `modal_compare_condition2/hf_artifacts/condition_dumps/hf_talker/`
- Latest deterministic vLLM talker decode trace used during debugging:
  - `/tmp/modal_compare_fix3/vllm_async_artifacts/217312e3c07c4285b9d9c804023592cc_artifacts/debug/minicpmo4_5_async_chunk/0_ac5421f3-237f-4f78-b134-4637dced00cd/talker_decode_steps.jsonl`
- Latest exact step-0 replay artifacts:
  - vLLM rerun root:
    - `/tmp/modal_compare_replay_step0/de2bcf85aedf466682838733883ce0c6_artifacts/debug/minicpmo4_5_async_chunk/0_64afb3d7-a676-4e20-b8c8-c759e49fbd73/`
  - vLLM consumed step-0 prompt dump:
    - `/tmp/modal_compare_replay_step0/de2bcf85aedf466682838733883ce0c6_artifacts/debug/minicpmo4_5_async_chunk/0_64afb3d7-a676-4e20-b8c8-c759e49fbd73/talker_decode_step_tensors/step_0000/`
  - HF replay result on the exact captured prompt:
    - `/tmp/minicpmo45_hf_step0_replay.json`

### What Is Already Fixed

- The thinker-side async off-by-one handoff bug was fixed with a carry-token style alignment in `thinker2talker_async_chunk`.
- `yield_chunk_token_ids` now match the HF reference.
- Thinker-side `tts_embeds` / raw condition content now match HF very closely.
- Stage bootstrap was fixed so stage 1 reads the correct outgoing connector config instead of silently using the incoming connector.
- Because of that connector fix, `async_talker_greedy: true` now actually takes effect and deterministic talker comparison is possible.

### What Still Does Not Match

- The talker output still diverges from HF even after the thinker-side condition is aligned.
- The remaining mismatch is now believed to be inside the async talker private decode path, not the stage input processor.
- The most important remaining difference is how the talker assembles and consumes the first decode prompt internally.

HF reference behavior from `generate_with_buffer`:
- It receives a raw `condition` tensor.
- If `text_finished`, it appends `text_eos_embed`.
- It always appends `audio_bos_embeds`.
- Then it runs the first decode step on that internal prompt.

What this means for debugging:
- HF raw condition dump shape for the first chunk is `10`, but the internal first decode prompt is effectively `11` because `audio_bos` is appended inside `generate_with_buffer`.
- The final chunk is similar: raw condition is shorter, and HF appends `text_eos` and `audio_bos` internally before decoding.

### Strongest Current Clue

The older "HF step 0 is `3701` while vLLM step 0 is `3704`" story is no longer the best summary.

After adding an exact consumed-prompt dump on the vLLM side and a one-step HF replay path:
- The vLLM async talker now dumps the actual consumed step-0 tensors:
  - `inputs_embeds.pt`
  - `position_ids.pt`
  - `hidden_states.pt`
  - `raw_logits.pt`
  - `sampling_logits.pt`
  - `probs.pt`
- Those are dumped under:
  - `/tmp/modal_compare_replay_step0/de2bcf85aedf466682838733883ce0c6_artifacts/debug/minicpmo4_5_async_chunk/0_64afb3d7-a676-4e20-b8c8-c759e49fbd73/talker_decode_step_tensors/step_0000/`
- The exact captured vLLM `inputs_embeds.pt` + `position_ids.pt` were then replayed through HF using:
  - `scripts/modal_minicpmo45_hf_smoke.py`

Replay result:
- vLLM step-0 greedy token: `3704`
- HF replay on the exact same captured prompt: `3704`
- Prompt tensor bytes match exactly in the replay:
  - `inputs_embeds` SHA256: `d8f7e3024ba5dfc2ab8c610e5555a67b8b47bec4ccbb9723f8a5e4146fd513ba`

Interpretation:
- The exact step-0 decoder prompt is no longer the main suspect.
- With identical step-0 decoder input, HF and the current vLLM async private decoder pick the same first token.
- So the earlier "HF 3701 vs vLLM 3704" gap was not a proof that the same prompt produced different first-token behavior.
- The remaining mismatch likely lives in one of:
  - end-to-end HF vs vLLM prompt/state not actually being equivalent in the earlier comparisons
  - later autoregressive / cache / chunk-state handling after step 0
  - smaller decoder numerical differences that do not flip step-0 top-1 but may still accumulate later

### Newer Deterministic Comparison Note

- The earlier `modal_compare_condition2/hf_artifacts/hf_talker_chunks.jsonl` reference was not a clean apples-to-apples baseline for the deterministic async talker because it was produced with HF defaults (`do_sample` not forced off, `temperature` recorded as `0.7`).
- A newer deterministic HF rerun with:
  - `streaming=true`
  - `do_sample=false`
  - `temperature=0.9`
  - `seed=42`
  - the same long async-chunk prompt
  produced a cleaner reference at:
  - `/tmp/modal_compare_hf_greedy/hf_talker_chunks.jsonl`
- That deterministic HF reference starts with:
  - `[3701, 4299, 4299, 6486, 4218, 6405, ...]`
- The current vLLM async talker trace still starts with:
  - `[3704, 4218, 4218, 4218, ...]`
- So after removing the compare-setup confusion, the remaining mismatch is still real and still inside the async private talker decode path.

### Current Best Root-Cause Hypothesis

- The async private decoder was constructing a fresh `transformers.LlamaModel` from a hand-written minimal `LlamaConfig`.
- That means the async path could silently drop behavior-critical TTS-backbone config fields present in the real HF checkpoint config, for example:
  - `rms_norm_eps`
  - `rope_theta`
  - `hidden_act`
  - attention / RoPE / implementation-related config overrides
- Since thinker-side condition dumps now match HF and the remaining divergence starts immediately at step 0 / step 1 of talker decode, this config mismatch is currently the cleanest explanation.

### Patch Applied

- `vllm_omni/model_executor/models/minicpmo4_5/minicpmo4_5_talker.py`
  now builds the async HF backbone config by cloning the real TTS config with `LlamaConfig.from_dict(self.config.to_dict())`, then reasserting the core structural dimensions.
- This is intended to make the async private decoder match HF `self.tts.model` much more closely than the old shape-only config reconstruction.
- The async talker also now dumps exact consumed decode-step tensors for the first few steps:
  - `inputs_embeds`
  - `position_ids`
  - `hidden_states`
  - `raw_logits`
  - `sampling_logits`
  - `probs`
- `scripts/modal_minicpmo45_hf_smoke.py` now has a one-step replay path that can feed a captured `inputs_embeds.pt` and `position_ids.pt` into HF TTS and report the resulting first-step logits / top tokens.

### Verification Status

- A fresh Modal async-chunk rerun completed successfully with the new decode-step dumps enabled.
- That rerun produced:
  - vLLM audio duration: `24.36s`
  - debug root:
    - `/tmp/modal_compare_replay_step0/de2bcf85aedf466682838733883ce0c6_artifacts/debug/minicpmo4_5_async_chunk/0_64afb3d7-a676-4e20-b8c8-c759e49fbd73/`
- The exact captured step-0 prompt was replayed through HF and produced the same first token `3704`.
- However, full raw logits are still not bitwise identical:
  - vLLM step-0 `raw_logits` summary SHA256:
    - `dee54bdfcfba4f9254b099e2ef324be705a498f459d4a2bdb8fa1f4d164e8532`
  - HF replay step-0 `raw_logits` summary SHA256:
    - `6a6114dc3a8064e58af7a8d227207cca7a60733841e7f6d8b765a43ed72df731`
- So the current state is:
  - identical step-0 prompt -> same top-1 token
  - identical step-0 prompt -> not bitwise-identical full logits

### Secondary Suspects

- HF `generate_with_buffer` has explicit `attention_type` branches:
  - `sliding_recompute`
  - `reindex`
  - `sliding_window`
- Our current async private decoder path is not yet a full 1:1 port of those behaviors.
- Even if token 0 is fixed, later chunks may still drift if `attention_type` cache handling is different.

### Next Things To Investigate / Debug / Try / Run

1. Move from step-0 prompt replay to step-1 / later-step state replay.
   - Step 0 prompt equivalence is no longer enough.
   - The next clean question is whether HF and vLLM still agree once KV cache / `past_key_values` state is involved.

2. Confirm the final-chunk raw condition semantics still match HF.
   - HF raw final condition dump is shorter than the non-final one.
   - EOS should be added inside the reference decoder, not pre-baked into the raw condition dump.

3. Log the reference control fields that affect decoding.
   - `attention_type`
   - `chunk_size`
   - `temperature`
   - `eos_token`
   - whether step 0 skips logits processors / warpers because it is treated as `audio_bos`

4. Compare later-step state handling, not just emitted tokens.
   - Dump / inspect:
     - `past_key_values` length growth
     - position-id evolution
     - whether HF `attention_type` uses `reindex`, `sliding_recompute`, or `sliding_window`
   - If needed, add a narrower replay harness that captures / replays step 1 with cache state.

5. Compare raw logits directly for step 1 and later.
   - Step 0 top-1 now agrees, but full logits still differ.
   - Need to see whether that smaller numerical gap becomes the real source of later token drift.

6. Port the relevant `attention_type` cache behavior if HF is not using the simple path.
   - Especially `reindex` or `sliding_recompute`, since those can change positions and KV-cache state after the first chunk.

7. Rerun the deterministic Modal comparison after each cache/state fix and compare full chunk streams.
   - HF reference: `modal_compare_condition2/hf_artifacts/hf_talker_chunks.jsonl`
   - vLLM debug: latest `talker_decode_steps.jsonl`, `talker_decode_step_tensors/`, and `talker_codec_chunks.jsonl`

### Practical Short-Term Plan

- First priority: understand whether HF uses a non-trivial `attention_type` in the real end-to-end path.
- Second priority: compare step 1 and later with cache state, not only step 0 prompt replay.
- Third priority: treat the remaining full-logit mismatch as a secondary clue unless it flips later token choices.
