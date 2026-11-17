# TaoMate-H3: realtime streaming MiniMax-H3

[TaoLiveAIGC/TaoMate-H3](https://huggingface.co/TaoLiveAIGC/TaoMate-H3) is a rank-128 LoRA
for the [MiniMax-H3 FL2VA checkpoint](https://huggingface.co/MiniMaxAI/MiniMax-H3).
It generates a continuous presenter video with audio and accepts prompt changes between
five-second generation windows. vLLM-Omni serves it through the AR-Diffusion engine and
`WS /v1/realtime/video`.

## Execution and cache ownership

A window contains four causal phases of 34/34/34/17 frames. The first window produces
124 frames including the decoder's five-frame tail; subsequent windows add 119 frames.
The timeline is fixed at 24 fps with 32 kHz stereo audio.

| Operation | Model behavior |
| --- | --- |
| Prompt encode | Qwen3-VL encodes the prompt at startup and on an interaction. One prompt conditions all four phases of a window. |
| Audio teacher | The base DiT, with LoRA disabled, runs nine audio-only forwards once per window. It retains the previous window's last 40 audio latents per channel. |
| Student denoise | Three LoRA forwards per phase consume the current document and clean history. The teacher's states 3, 6 and 9 replace the corresponding student audio states. |
| Clean commit | A separate sigma-zero forward recomputes clean K/V. Only a successful complete forward publishes the phase into history. |
| Decode | Incremental video VAE windows retain their temporal overlap; the audio VAE decodes the matching sample interval. The final phase flushes the held video tail. |

The AR-Diffusion runtime owns session KV allocation, capacity accounting, transactions
and release. TaoMate supplies audio/video token spans and retention rules: keep the
first phase's video sink and two most recent phases, and clear audio history every
12 windows by default. Each Ulysses rank stores its own attention-head shard. Denoising
scratch never becomes committed history; the last student step is still noisy and
cannot replace the clean-commit forward.

The shared step scheduler handles runnable and prompt-waiting requests, interaction
wakeups, deadlines and cancellation. TaoMate defines the window boundary and optional
fallback prompt. A waiting request does not repeatedly execute a model step.

Teacher/student math, global audio/video positions and incremental VAE state remain
model-specific. By default the LoRA uses BF16 sidecars on the resident base weights;
the worker-owned `DiffusionLoRAManager` temporarily suspends the adapter for the teacher.
The checkpoint uses ordinary low-rank A/B pairs in a native serialization format; its
fused QKV factors already match the loaded projection order. The manager also controls
the optional merged student bank, which uses a second weight set and has different
memory and numerical behavior.

## Requirements and serve commands

- MiniMax-H3 `FL2VA/` weights and the TaoMate adapter directory containing
  `adapter_model.safetensors` and `config.json` (or `adapter_config.json`). Point the
  model path at `FL2VA/`, not the snapshot's modular Diffusers index.
- Hopper or newer GPUs with FlashAttention-3 (`fa3_fwd_interface`), or FA4 on Blackwell.
  Reference deployments were exercised on H200-class 141 GB GPUs; budget for the DiT,
  text encoder, VAEs, LoRA, engine KV, scratch buffers and graph allocations.
- Pure Ulysses sequence parallelism with DiT tensor parallelism fixed at 1.
  Both supplied configurations admit one session (`max_num_seqs: 1`).

Four-GPU BF16 configuration:

```bash
vllm serve /path/to/MiniMax-H3/FL2VA --omni --trust-remote-code \
  --deploy-config vllm_omni/deploy/taomate_h3_usp4_realtime.yaml \
  --lora-path /path/to/TaoMate-H3 --port 8000
```

Two-GPU configuration with per-channel FP8 DiT linears and teacher/text CUDA graphs:

```bash
vllm serve /path/to/MiniMax-H3/FL2VA --omni --trust-remote-code \
  --deploy-config vllm_omni/deploy/taomate_h3_usp2_realtime.yaml \
  --lora-path /path/to/TaoMate-H3 --port 8000 \
  --stage-init-timeout 1800 --init-timeout 2100
```

The two-GPU config leaves the text encoder, token refiner, patch/condition projections,
time embeddings, final layer and AdaLN projections in BF16. It uses the checkpoint's
256 px video decoder tiles with 32 px minimum overlap and batched tile decoding.
Do not increase the decoder tile size: 384/480 px tiles previously produced a visible
16 px lattice. Keep `taomate_h3_lora_merge: false`: merging into FP8 student weights
previously erased much of the small adapter delta and blurred the video.

The engine performs a throwaway rollout before serving, including in eager mode when
`ar_diffusion_kv_config.warmup_cudagraph: true`. Four windows cover the teacher's initial
audio length and the later 198/198/199-latent cycle. Capturing teacher and encoder graphs
for a wide prompt-length range can take several minutes; size the startup timeouts and
capture range for the deployment.

## Client and prompt updates

```bash
python examples/online_serving/streaming_video_generation/streaming_video_client.py \
  --port 8000 --model /path/to/MiniMax-H3/FL2VA --size 480x864 --fps 24 --num-frames 362 \
  --seed 8301 --no-helios-distilled-preset --prompt "..." \
  --prompt-updates '[{"at": 6.0, "prompt": "..."}]'
```

A `session.start` specifies `prompt`, `width`, `height`, `fps: 24`, `seed` and
`num_frames`. A budget of `124 + 119 * (n - 1)` frames requests `n` windows.
Binary messages carry fragmented MP4; `video.chunk_metadata` describes each chunk.
Audio transport requires the separate fMP4 audio support in PR #8314. Without it,
the WebSocket MP4 contains video only even though the model produces aligned audio.

Send prompt updates using the interaction envelope:

```json
{
  "type": "session.interaction",
  "interaction": {
    "event_id": "prompt-2",
    "event": {"prompt": "The presenter introduces the next topic."},
    "transition_chunks": 0
  }
}
```

The update is processed at a chunk boundary and conditions the next generation window;
it never changes the teacher or remaining phases of a window already in progress.
The `started_event_ids`, `active_event_ids`, and `completed_event_ids` markers
acknowledge installation of the prompt embeddings in session state. With TaoMate's
immediate transition, these markers can appear before the window that uses the new
prompt: a window snapshots conditioning when its first phase starts. They do not
identify the first visibly changed frame; VAE temporal context and overlap blending
can delay or spread that change across frame boundaries.
Different prompt lengths are supported. Multiple pending updates use the latest prompt.
Send `session.stop` to cancel, and `session.ping` to keep a deliberately held connection
alive. A bounded hold may use the previous prompt or the configured fallback prompt;
an update arriving after that window starts applies to the following window.

## Deployment controls

These keys live under the stage's `model_config`. Defaults below are pipeline defaults;
the supplied YAML files enable additional optimizations.

| Key | Default | Meaning |
| --- | --- | --- |
| `taomate_h3_width` / `taomate_h3_height` | 480 / 864 | Fixed canvas; both dimensions must be 32-aligned, with short edge 480, 768 or 1088. |
| `taomate_h3_seed` | 8301 | Default seed. Window `k` uses `seed + k` for audio and `seed + k * 1000003` for video. |
| `taomate_h3_audio_kv_reset_requests` | 12 | Clear retained audio K/V every N windows while keeping the video sink and recent video. |
| `taomate_h3_allow_no_lora` | false | Run the base checkpoint without the adapter for debugging. |
| `taomate_h3_pad_text_tokens` | unset | Bound packed text shapes using 64-token buckets, capped by this budget. Longer prompts run unpinned with a warning. |
| `taomate_h3_hold_for_prompt` | false | Starting with window 1, wait for a prompt update received since the previous window started. |
| `taomate_h3_hold_max_seconds` | 0 | Maximum hold duration; 0 waits indefinitely. On timeout start a window with the previous or fallback prompt. |
| `taomate_h3_hold_fallback_prompt` | unset | Prompt used on hold timeout; encoded once on first use. Unset retains the previous prompt. |
| `taomate_h3_hold_poll_seconds` | 0.02 | Compatibility setting from worker polling; prompt waits now use shared scheduler wakeups. |
| `taomate_h3_teacher_cuda_graph` | false | Capture/replay teacher forwards by document shape. Capture failure disables graphs and falls back to eager execution consistently across Ulysses ranks. |
| `taomate_h3_cuda_graph_max_entries` | 16 | Teacher graph capacity; unpinned entries are evicted in LRU order. |
| `taomate_h3_teacher_graph_text_lengths` | unset | Inclusive token-count range such as `"360-470"` to capture at startup. Requires a text budget and at least three teacher graph entries per length; these entries stay pinned. |
| `taomate_h3_text_encoder_cuda_graph` | false | Capture/replay text-only encoding by token count, including the configured startup range. Offloaded encoders keep eager execution. |
| `taomate_h3_warmup_requests` | 4 | Number of windows in the startup rollout; four cover the initial and continuation teacher geometries. |
| `taomate_h3_adaln_cache` | true | Cache exact AdaLN projections. Disable to recompute small projections and avoid the host digest synchronization. |
| `taomate_h3_lora_merge` | false | Experimental shadow student weights with merged LoRA; restores base weights for the teacher. Leave off for the validated FP8 configuration. |
| `taomate_h3_decode_overlap` | false | Queue video decode on a second CUDA stream alongside clean commit. |
| `taomate_h3_cudnn_benchmark` | false | Enable cuDNN autotuning for fixed-shape VAE convolutions. |
| `taomate_h3_vae_decoder_tile_size` | checkpoint: 256 | Decoder tile edge, a positive multiple of 16. Preserve 256 for the released checkpoint; too few tiles for the VAE ranks trigger a slower fallback. |
| `taomate_h3_vae_decoder_tile_overlap_min` | checkpoint: 64 | Minimum tile overlap, a nonnegative multiple of 16. The USP2 config uses 32. |
| `taomate_h3_vae_stack_tiling` | false | Decode each rank's tiles as a batch to reduce peak memory. |
| `taomate_h3_fp8_quant_cuda_op` | true | Use vLLM's CUDA activation-quantization kernel for FP8 linears. |
| `taomate_h3_log_timings` | false | Log phase/stage durations; adds device synchronizations. |
| `taomate_h3_profile_chunk` / `taomate_h3_profile_dir` | unset / `/tmp/taomate_h3_profile` | Profile the selected chunk and write per-rank kernel tables and Chrome traces. |

`ar_diffusion_kv_config` controls the engine pool and warmup. Its memory fraction now
budgets real persistent K/V; it is not a bookkeeping-only allocation. Include graph
and model-owned decoder/teacher state when checking available capacity.

Both configurations default to `enforce_eager: true`. Regional compilation remains an
option with `enforce_eager: false`, a suitable `taomate_h3_pad_text_tokens` budget and
`VLLM_OMNI_TORCH_DYNAMO_RECOMPILE_LIMIT=64`. Validate it with the chosen graph and LoRA
settings; combining compilation and merged weights previously failed graph capture.

## Validation and limits

- One session per server; this integration does not add concurrent-session batching.
- Text-only T2VA with a fixed deployment canvas. Reference media, another canvas, or
  explicit `fps`/`frame_rate` values other than 24 are rejected before generation.
- Video is not bit-reproducible between runs, including in the release runtime.
- Teacher graphs distinguish valid prompt lengths. A previously unseen length outside
  the startup range can pause the stream while capturing; size the range using actual
  token counts, not characters.

Validate the deployment with prompt changes, at least 13 windows to cross audio-history
reset, bounded/unbounded prompt holds, cancellation followed by another session, and
complete MP4 decoding. Compare both per-window latency and peak GPU memory with the
same checkpoint, prompts, precision, graph settings and host load.
