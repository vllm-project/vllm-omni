# TaoMate-H3: realtime streaming MiniMax-H3 with the TaoMate LoRA

TaoMate-H3 ([TaoLiveAIGC/TaoMate-H3](https://huggingface.co/TaoLiveAIGC/TaoMate-H3)) is a
rank-128 LoRA over [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) that turns the
bidirectional five-second text-to-video-and-audio model into a causal live streamer: a
presenter that keeps talking and acting while the prompt of each next five-second request
is chosen just in time. vLLM-Omni hosts it on the AR-Diffusion realtime runtime with pure
Ulysses sequence parallelism.

## How it works

Each five-second request (124 native frames for the first request, 119 afterwards, at
24 fps with 32 kHz stereo audio) is generated as four causal phases of 34/34/34/17 frames:

| Step | What runs | Where |
| --- | --- | --- |
| Prompt | Qwen3-VL text encoder (tensor parallel across the Ulysses ranks) | at session start and whenever the client updates the prompt |
| Audio teacher | base MiniMax-H3 (LoRA disabled), nine audio-only forwards over `[text \| 1 s clean reference tail \| new audio]` | once per request, before its first phase |
| Phase denoise | three LoRA forwards of the phase document `[text \| audio \| video]`, attending to text, the persistent clean audio/video KV of earlier phases and the current chunk | per phase |
| Clean commit | one sigma-zero forward that recomputes the phase's K/V from its clean latents and appends them to the KV history (first-phase video sink + two most recent phases; audio history dropped every twelve requests) | per phase |
| Decode | tile-parallel temporal-window video VAE decode plus sliding-window audio VAE decode | per phase, streamed as one fragmented-MP4 chunk with an AAC track |

The LoRA is never merged: the same resident BF16 weights serve the LoRA student and the
LoRA-free audio teacher, so one 62 GB DiT copy per rank is enough. TP is fixed at 1; the
attention heads are split across the Ulysses ranks and each rank keeps the clean KV
history of its own head shard.

## Requirements

- 4 GPUs with at least 96 GB each (measured 85 GB per rank at 480x864 after load: DiT
  62 GB, text encoder TP4 16 GB, VAEs, LoRA 1.2 GB, model-owned session state 12.6 GB).
- Hopper or newer (`fa3_fwd_interface` / FlashAttention-3 dense kernels; FA4 on Blackwell).
- The MiniMax-H3 snapshot (`FL2VA/` partition) and the TaoMate-H3 adapter directory
  (`adapter_model.safetensors` + `config.json`).

## Serve

```bash
vllm serve /path/to/MiniMax-H3/FL2VA --omni --trust-remote-code \
  --deploy-config vllm_omni/deploy/taomate_h3_usp4_realtime.yaml \
  --lora-path /path/to/TaoMate-H3 --port 8000
```

`--model` must point at the `FL2VA/` partition directory (the snapshot root carries the
Diffusers modular index and is resolved as the FastH3 modular checkpoint). The deploy
config pins `tensor_parallel_size: 1`, `ulysses_degree: 4`, `text_encoder_tp_size: 4`,
`vae_patch_parallel_size: 4`, the AR-Diffusion engine, step execution and streaming output.
Per-deployment knobs live under `model_config`:

| Key | Default | Meaning |
| --- | --- | --- |
| `taomate_h3_width` / `taomate_h3_height` | 480 / 864 | The fixed session canvas (32-aligned, short edge 480, 768 or 1088) |
| `taomate_h3_seed` | 8301 | Default seed; request `k` draws audio noise from `seed + k` and video noise from `seed + k * 1000003` |
| `taomate_h3_audio_kv_reset_requests` | 12 | Drop the audio KV history every N requests (video sink and recents are kept) |
| `taomate_h3_allow_no_lora` | false | Stream the base H3 without the adapter (debugging only) |
| `taomate_h3_pad_text_tokens` | unset | Prompt token budget that pins every phase and teacher document to one packed length per kind (for compiled or CUDA-graph runs). The text rows reserved are the prompt's length rounded up to 64 tokens, capped by this budget, so a 335-token prompt under a 512 budget carries at most 49 pad rows. Prompts above the budget run unpinned (warning once). Real persona prompts measure 330-410 tokens (cited from the live-agent team), so deployments should set 512 |
| `taomate_h3_log_timings` | false | Log per-phase stage timings (adds device synchronizations) |
| `taomate_h3_hold_for_prompt` | false | Just-in-time lock: request `k >= 1` starts only after a `session.interaction` prompt update arrived since request `k-1` started; until then the stream idles at the request boundary (send `session.ping` to keep the stall timer fresh). For clients that choose every request's prompt at the last moment |
| `taomate_h3_hold_max_seconds` | 0 | Bound on the just-in-time hold: after this many seconds at a request boundary without a new prompt, the request starts with the previous prompt and a late update applies one request later (a warning is logged). 0 keeps the hold unbounded, which stalls the stream for as long as the client's prompt decision takes (measured locally in the live-agent demo: an 8 s stall while a hosted LLM took 4 s to decide). Real-time deployments that hold should set about 0.5-1.0 |
| `taomate_h3_hold_fallback_prompt` | unset | Prompt a request runs when its bounded hold ran out, instead of replaying the previous prompt (which may be a spoken line): a neutral "listening" description, encoded once on first use. The expiry is agreed across the DiT ranks before any of them starts the request |
| `taomate_h3_hold_poll_seconds` | 0.02 | Idle-step period while a request boundary is held |
| `taomate_h3_teacher_cuda_graph` | false | Replay the audio teacher's nine forwards from CUDA graphs, one graph per document shape (prompt length, first or later request, reference tail or not). The first capture is checked with `torch.cuda.set_sync_debug_mode("error")`; any capture failure falls back to eager for the rest of the process, agreed across the Ulysses group |
| `taomate_h3_cuda_graph_max_entries` | 16 | Resident teacher graphs (least recently used shape evicted); they share one memory pool |
| `taomate_h3_warmup_requests` | 4 | Requests of the load-time warmup session. Later requests cycle through three audio latent counts (198, 198, 199 per channel), so four requests visit every teacher document shape: the graphs are captured and every phase size has been allocated before the first client connects |
| `taomate_h3_teacher_graph_text_lengths` | unset | Prompt token counts (`"lo-hi"`) whose teacher graphs are captured during the load-time warmup, three document shapes per count (about 0.5 s and 5 MB each, estimate); these graphs are pinned against LRU eviction by later shapes. A teacher graph is keyed by the prompt's token count, so without this each new prompt length captures inside the stream (about 0.7 s per shape). Needs `taomate_h3_pad_text_tokens` and enough `taomate_h3_cuda_graph_max_entries` |
| `step_async_output` | false | Separate PR (branch `feat/step-async-output`). Generic step-execution knob (read by the worker and the executor, not by the pipeline): pack each streamed chunk's media into shared memory on the worker's background thread and let the engine await it, instead of copying the 42 MB of frames per phase on the step thread. Measured locally at USP2: 4.98 -> 4.78 s per request (ten requests, constant short prompt); prompt updates unaffected |
| `taomate_h3_text_encoder_cuda_graph` | false | Replay the text-only prompt encode of a prompt update from a CUDA graph (one graph per token count, captured for `taomate_h3_teacher_graph_text_lengths` at load, exact: the encoder's own modules run with graph-safe indexing). Prompts with images or videos, offloaded encoders and non-encoder ranks keep the eager path |
| `taomate_h3_adaln_cache` | true | Exact AdaLN projection cache. Its key is a host digest of the timestep embedding (one device-to-host copy per forward); `false` recomputes the few projected rows per layer and removes that synchronization from every student forward |
| `taomate_h3_cudnn_benchmark` | false | cuDNN autotuning for the fixed-shape VAE convolutions (measured: no change) |
| `taomate_h3_lora_merge` | false | **Leave off.** Merges the LoRA delta into a second weight set for the student (per target a shadow linear `W + scale*B@A`, quantized like the base). Measured locally (2026-09-29, corrected): with per-channel FP8 the merged student renders about 4.7x less sharp video (median Laplacian variance over 24-25 recorded frames: 105-110 with the merge, 497 without it, 500-512 for the four-GPU BF16 demo, so the exact path matches BF16): the delta is 0.2-0.7% of |W| and 83-98% of it becomes FP8 rounding error, so the fine-tune is largely lost even though the merge is unbiased in expectation. It saved 0.12 s per request. Kept as an experiment knob; the BF16 delta on the hooks is the exact path |
| `taomate_h3_decode_overlap` | false | Queue the phase's video VAE decode on a second CUDA stream before the clean-commit forward (the decode only needs the phase's clean latents; frames are fetched after the host has prepared the next phase). Measured locally at persona length: no gain (phase totals within 0.015 s of the serial order), because the commit forward is device time too; kept as an experiment knob |
| `taomate_h3_vae_decoder_tile_size` | unset (checkpoint: 256) | Decoder tile edge of the video VAE in pixels (multiple of 16). **Do not raise it: 384 and 480 px tiles render a 16 px lattice over the whole frame (isolated 2026-09-28/29; the decoder's positional ids are normalized to the tile extent).** Fewer tiles than `vae_patch_parallel_size` falls back to the slower whole-frame decode (a warning is logged) |
| `taomate_h3_vae_decoder_tile_overlap_min` | unset (checkpoint: 64) | Minimum overlap between decoder tiles in pixels (multiple of 16). At 480x864 the checkpoint's 64 px gives 15 tiles of 256 px covering 2.37x the canvas; 32 px gives 8 tiles covering 1.26x and halves the decode (0.37 -> 0.19 s per 34-frame phase, measured locally) with no seams at the tile borders (2x crops compared against the 64 px decode); 16 px gives the same 8 tiles |
| `taomate_h3_vae_stack_tiling` | false | Decode a rank's video VAE tiles as one batched forward of the 3D ViT decoder (the checkpoint's `stack_tiling`). Measured locally on one GPU (8 tiles, 7-latent window): output bit-identical to sequential tiles, 170 vs 174 ms, peak memory 5.9 vs 9.1 GB, so the two-GPU config turns it on for the memory |
| `taomate_h3_fp8_quant_cuda_op` | true | Quantize the FP8 linears' activations with vLLM's CUDA op instead of the inductor-compiled native path (profiled on the two-GPU server: the inductor reduction kernel took 54 ms per 34-frame phase at about 600 GB/s; the CUDA op reads the rows at about 1.3 TB/s). Same per-token scale and e4m3 payload |

The deploy config keeps `ar_diffusion_kv_config.warmup_cudagraph: true`: the AR runner runs
one throwaway five-second request at load time (the pipeline opts into this warmup in eager
mode as well), so the first chunk of a session arrives in about 3 s instead of 17-20 s on
a cold server.

## Stream

Open `WS /v1/realtime/video`, send `session.start` with the prompt, `width`, `height`,
`fps: 24`, `seed` and `num_frames` (the frame budget decides how many five-second
requests the session runs; `124 + 119 * (n - 1)` frames is `n` requests), and receive one
binary fragmented-MP4 chunk per phase. The bundled client works unchanged:

```bash
python examples/online_serving/streaming_video_generation/streaming_video_client.py \
  --port 8000 --model /path/to/MiniMax-H3/FL2VA --size 480x864 --fps 24 --num-frames 362 \
  --seed 8301 --no-helios-distilled-preset --prompt "..." \
  --prompt-updates '[{"at": 6.0, "prompt": "..."}]'
```

A `session.interaction` prompt update is applied at the next chunk boundary and takes
effect for the *next five-second request* (the audio teacher and all four phases of a
request share one prompt), which is TaoMate's just-in-time prompt lock. Every chunk's
`video.chunk_metadata` carries `num_frames`, `num_audio_samples` and `audio_sample_rate`;
the stream has an H.264 video track and an AAC audio track (the audio track comes with the separate fMP4 audio PR #8314; without it the stream is video only). Chunk 1 of a session is
39 frames of latents but 34 frames of pixels: the decoder holds the last five frames of
each temporal window until the next phase, and flushes them with the final chunk.

## Measured (4x H200-class 141 GB, eager BF16, 480x864)

Steady-state chunk period after the cold start, one session, measured locally:

| Chunk | Frames | Period |
| --- | --- | --- |
| phase 0 of a request (includes the teacher's nine forwards and, after a prompt update, the text encode) | 34 | 1.7-2.3 s |
| phases 1-2 | 34 | 0.85-0.95 s |
| phase 3 | 17 | 0.7-0.85 s |

A 13-request session (1552 frames, 64.6 s of video and audio) took 3.9-4.1 s of wall time
per 4.958 s request in steady state (real-time factor 0.78-0.82); the first chunk arrived
after 2.3-2.9 s with the load-time warmup. Stage timings per 34-frame phase
(`model_config.taomate_h3_log_timings: true`): teacher 0.7-0.8 s per request, three student
forwards 0.42 s, clean commit 0.15 s, video decode 0.2 s, audio decode 0.03 s. Regional
`torch.compile` (`enforce_eager: false`, `VLLM_OMNI_TORCH_DYNAMO_RECOMPILE_LIMIT=64`) and
FP8 online linears (`quantization: fp8`) gave no steady-state gain over eager here (4.3 s per
request); the launch-bound teacher forwards and the per-phase VAE decode are the next targets.

## Two GPUs (USP2): `vllm_omni/deploy/taomate_h3_usp2_realtime.yaml`

> **Quality finding (2026-09-28, frames compared locally and reviewed by a second model):**
> `taomate_h3_vae_decoder_tile_size: 480`, the setting behind every two-GPU number below,
> renders a uniform 16 px lattice over the whole frame, deformed faces and smeared texture from
> the first chunk on. Isolated with four runs on the same persona prompt: eager BF16 without the
> tile is clean; BF16 with the 480 px tile shows the lattice; per-channel FP8 (blocks only,
> boundary layers in BF16) without the tile is clean and sharp; per-channel FP8 with the tile
> shows the lattice. The lattice has the VAE's 16 px stride, so the decoder cannot be tiled at
> 480 px, and a 384 px tile shows a weaker lattice too (measured). The likely mechanism: the
> decoder's attention uses length-normalized positional ids (`create_token_ids`, coordinates
> scaled to the tile's own extent), so a tile size other than the 256 px it was trained with
> changes every position's encoding; a whole-frame decode would therefore not be correct either.
> cuDNN autotuning (`taomate_h3_cudnn_benchmark`) does not change the decode time. The per-tensor `quantization: fp8` was never isolated from this bug;
> the config now uses per-channel FP8, which is validated. Consequence: the VAE decode costs
> 0.36 s per 34-frame phase again and a persona-length request takes about 5.4 s (real-time
> factor 1.09) with correct output, 5.28 s with the merged student weights below; the 4.87-5.05 s
> figures were measured with the broken decode. Regional `torch.compile` (`enforce_eager: false`)
> fails to build with the merged weights (inductor backend error) and breaks the teacher graph
> capture; it stays off.

The two-GPU config keeps TP=1 and splits the heads across two Ulysses ranks
(`sequence_parallel_size: 2`, `text_encoder_tp_size: 2`, `vae_patch_parallel_size: 2`).
Memory per rank in eager BF16 is 100.7 GB after load (measured locally). Eager BF16 is
**not** real time on two H200-class GPUs at 480x864 (measured locally, 2026-09-28, five
requests, constant prompt, `taomate_h3_log_timings: true`, rank 0):

| Stage | 34-frame phase | 17-frame phase |
| --- | --- | --- |
| audio teacher (once per request, 9 forwards) | 0.78-0.84 s | - |
| 3 student forwards | 0.77-0.81 s | 0.43 s |
| clean-commit forward + cache append | 0.29-0.32 s | 0.16 s |
| tile-parallel video VAE decode | 0.35-0.37 s | 0.18 s |
| audio VAE decode | 0.03 s | 0.03 s |

Per request: 2.29 + 1.55 + 1.59 + 0.86 = 6.3 s of wall time for 4.958 s of content
(real-time factor 1.27; the client saw 6.4 s between request starts). Against the USP4
breakdown the DiT forwards take 1.9x longer per rank (2200 instead of 1100 rows through the
linears, twice the heads in attention), the VAE decode 1.8x, and the launch-bound teacher
is unchanged.

The two-GPU deploy config therefore enables the levers that leave the generated content
unchanged, plus a decoder tiling that only changes how the VAE is split across the ranks:

- `quantization: fp8` for the DiT linears (the GEMMs are about 60% of a student forward at
  USP2, estimate from parameter and row counts);
- `taomate_h3_teacher_cuda_graph` (the teacher's forwards run a few hundred rows and are
  launch-bound at about 85-90 ms each; a replay takes 38 ms);
- `taomate_h3_vae_decoder_tile_size: 480` (two 480x480 tiles, one per rank, instead of the
  checkpoint's 3x5 grid of 256 px tiles that decodes 2.4x the canvas);
- `taomate_h3_pad_text_tokens: 256` so every document kind has one packed length, and
  `taomate_h3_teacher_graph_text_lengths: "1-96"` so the graphs of every plausible prompt
  length exist before the first client connects.

Measured locally (2026-09-28, 480x864, two H200-class GPUs, five requests, constant prompt
of 32 tokens, `taomate_h3_log_timings: true`, rank 0; the warmup already holds the graphs):

| Stage | 34-frame phase | 17-frame phase |
| --- | --- | --- |
| audio teacher (once per request, 9 graph replays) | 0.35 s | - |
| 3 student forwards (FP8) | 0.67-0.70 s | 0.42-0.45 s |
| clean-commit forward + cache append | 0.24-0.26 s | 0.15-0.17 s |
| tile-parallel video VAE decode (2 x 480 px tiles) | 0.21-0.22 s | 0.10 s |
| audio VAE decode | 0.03 s | 0.03 s |
| phase preparation (packing, RoPE table) | 0.01-0.03 s | 0.00 s |

Per request the phases take 1.51 + 1.19 + 1.24 + 0.76 = 4.70 s of stage wall time (stage
ends synchronized with the device by the timing log), and the wall
time between the ends of consecutive requests (phases plus the runner's output handling) is
4.91-4.97 s for 4.958 s of content: real-time factor 0.99-1.00, against 1.27 for eager BF16.
Without the timing instrumentation a ten-request session gave 4.98 s per request on average
between the first chunks of consecutive requests (range 4.70-5.10 s; measured locally), so
the device synchronizations of the log cost nothing. This is real time with no headroom: a
playback buffer of about one second is needed, and the two events below each stall the
stream once. The 0.05 s per phase between the end of a phase and the next preparation is
the runner's output transport (34 uint8 frames, 42 MB, leave the worker process per phase).

- **A prompt length without a resident graph** captures the teacher graphs of that length
  inside the stream: about 0.7 s per shape (three shapes for a session's first prompt, two
  per later prompt update), plus one or two slow commits (0.5-0.9 s) right after a capture
  while the allocator regrows. Measured without the length range: a session start cost
  2.1 s before the first chunk, and each prompt update with a new token count stretched
  the next request by 2-3 s. With `taomate_h3_teacher_graph_text_lengths: "8-96"` the
  warmup captured 267 graphs in 149-165 s (measured locally; about 1.3 GB of static inputs)
  and the live sessions captured nothing.
- **A prompt update** re-encodes the text on the workers (Qwen3-VL at TP2): 0.13-0.27 s
  for 21-32 tokens, inside the phase in which the update arrives, so a request that also
  applies an update took 5.1-5.3 s of wall time (measured locally). A client that changes
  the prompt every request therefore runs at real-time factor 1.03-1.07 and drains its
  buffer by 0.2-0.3 s per request; one that changes it every few requests stays level.
  Measured over a 100-request session with a different prompt (3-60 words) every request:
  5.09 s per request once warm (requests 20-100; real-time factor 1.03), 20 s of drift over
  the 101 requests, four graph captures for a 3-token prompt because the range then started
  at 8 tokens (the config now starts at 1).

**Prompt length caveat.** Every number above is for prompts of 3-60 words (up to 96 tokens
of the FL2VA tokenizer). Persona prompts of the live agent measure 345-360 tokens (measured
locally by a peer session), above the config's `taomate_h3_pad_text_tokens: 256`: such
prompts run unpinned (one warning), so every teacher document shape is captured lazily and
the pinned-length assumptions do not hold. For that workload set `taomate_h3_pad_text_tokens`
at or above the longest prompt (e.g. 448) and a graph range around the real lengths (e.g.
`"300-430"`, about 390 graphs), and expect the larger documents (about 320 more text rows per
phase) to run somewhat slower than measured here. Measured locally by a peer session on two
GPUs (FP8, teacher graphs, pad 512, graph range 320-420, 480x864, 2026-09-28): 330-405-token
prompts with 17 updates over 44 requests gave a median of 5.26 s (min 5.15, max 5.35) between
consecutive requests' first chunks, **real-time factor 1.06, not real time** (about 0.3 s of
drift per request); a 100-request run with 332-340-token prompts and an update every request
(same server, AdaLN cache off, encoder graphs) gave 5.31 s per request mean, 5.33 s warm and 37 s
of drift over 101 requests, real-time factor 1.07. The two-GPU deployment is therefore real time for short prompts only; the
persona workload needs roughly another 6-7%. A pad just above the longest prompt (416 rather
than 512 for these prompts) removes about 100 wasted rows per phase document (estimate 1-2%).
On four GPUs (the USP4 recipe, eager BF16, `taomate_h3_hold_for_prompt`
on) persona prompts of 330-405 tokens with a prompt update on almost every request measured a
median of 4.29 s and a maximum of 4.53 s between consecutive requests' first chunks over 54
requests, real-time factor about 0.86 (measured locally by a peer session, 2026-09-28).

**LoRA path cost (measured locally, 2026-09-28).** The same two-GPU config served the base H3
without the adapter (`taomate_h3_allow_no_lora: true`, content not meaningful, timing valid):
3 student forwards 0.62 s per 34-frame phase (0.67-0.70 s with the pre-fusion hooks), commit
0.23 s (0.25 s), 4.70 s of wall time per request against 4.95 s: the LoRA hooks cost about
0.25 s per request with two GEMMs and an add per target, about half of that after the fused
`addmm_` (208 targets, two launches each; estimate). Recovering the rest needs the delta merged
into a second FP8 weight set for the student (8 GB per rank). A CPU check of the merge on 22
targets (`|D|/|W|` = 0.2-1.2%): per-tensor e4m3 requantization keeps the delta in expectation
(projection of Q(W+D)-Q(W) onto D is 1.01-1.05 |D|) with requantization noise of about 2.6 |D|
orthogonal to it, and the total error against the BF16 student is 0.0265 either way, so the
merge is unbiased but re-rolls the FP8 noise; it is an opt-in trade, not an exact optimization.

**Latest persona-length state (measured locally by the sibling session, 2026-09-28 12:30, two
GPUs, 332-340-token prompts with an update every request, 64-token text buckets, fused LoRA
hooks, text-encoder graphs, 31 requests):** 5.05 s per request (real-time factor 1.02); per
34-frame phase teacher 0.35, 3 student forwards 0.68-0.71, commit 0.25-0.27, VAE decode 0.21,
preparation 0.02-0.04, and 0.045 s between the end of a phase and the next preparation, i.e.
0.18 s per request of output transport. That transport is the worker packing the phase's 42 MB
of uint8 frames into a fresh POSIX shared-memory segment on the step thread (`ipc.py`
`_array_to_shm`: segment creation, first-touch page faults and a memcpy); the worker already
has a background packing thread with a side-stream D2H path, but `_return_result` and the
thread's creation are limited to request mode (`not step_execution`). Moving step-mode chunk
outputs onto that thread (worker, executor id propagation, and the engine's chunk streaming
awaiting `OUTPUT_READY` per chunk) is the remaining lever that closes the persona gap without
touching the model; it is not implemented yet.

**Where the time goes at persona length (measured locally, 2026-09-28 12:35, two GPUs, 12
requests, host issue time of each forward logged next to its wall time):** the three student
forwards of a 34-frame phase take 0.68-0.71 s of wall time with 0.41-0.43 s of host issue time,
so they are device-bound (about 0.23 s of device work each). The 17-frame phase (0.44 s wall,
0.41 s issue) and the clean-commit forward (0.25-0.27 s wall, 0.15 s issue) looked launch-bound
by that test, but a commit forward runs the same rows as a student forward (about 0.23 s of
device work), so its wall time is device time as well and the host merely finishes issuing
early; overlapping the video decode with it gave nothing (see `taomate_h3_decode_overlap`).
A steady request is teacher 0.36 + student forwards 2.52 + commits 0.95 + VAE decode 0.73 +
preparation 0.07 + inter-phase gaps 0.24 = 5.07 s. Consequences: the device is saturated end to end, so CUDA graphs over the student
forwards would recover little (at most the 17-frame phase's margin, well under 0.1 s per
request) and are not pursued; the streaming attention already runs FlashAttention-3 (`fa3_fwd_interface`); the
prompt re-encode is solved (0.048 s per update from the graph). Two overlaps address the rest
without touching the model: the next phase's packed layout and the next request's noise draws
are built on the host while the device decodes (`prepare` 0.01-0.04 s -> 0.001-0.02 s per
phase, measured; period unchanged within noise). What is left is device work and the host
transport: the LoRA delta's device time (0.2-0.3 s per request, a merged FP8 student weight
set) and the inter-phase transport gap (0.24 s per request, `step_async_output`); together they
cover the 0.1 s per request that persona-length prompts with a change every request still lack.

**`step_async_output` validated (measured locally, 2026-09-28 14:00, two GPUs, ten 32-token
requests, timings on):** 4.78 s between consecutive requests' first chunks (4.61-4.84 s;
4.98 s without the knob), real-time factor 0.96; the prompt-update session ran 4.75-4.90 s
per request with the content changing per update; no warnings. The phase totals fell by
0.05-0.1 s each while the inter-phase gap stayed at 0.05 s, so the removed shared-memory copy
had also been slowing the following phase's host work. The two-GPU deploy config turns the
knob on. Persona-length prompts are not re-measured with it yet (factor 1.02 before).

**Persona-length prompts with a change every request are real time on two GPUs (measured
locally, 2026-09-28 14:15, 30 requests, 332-340-token prompts, a new prompt every request, all
of: FP8, teacher and text-encoder graphs for 320-420 tokens, 64-token text buckets, fused LoRA
hooks, AdaLN cache off, `step_async_output`):** 4.87 s per 4.958 s request on average (warm
4.87, maximum 4.99), real-time factor 0.98, and the stream gained 1.0 s over the 30 requests
instead of drifting. Per 34-frame phase: teacher 0.35 (request start only), 3 student forwards
0.68-0.71, commit 0.25-0.27, VAE decode 0.22-0.23, 0.06 s between phases. The margin is 2% and
it depends on the host: a 100-request session right after it on the same server averaged 4.99 s
(gained 0.7 s over the first 50 requests, then lost 5.7 s over the next 50 while the shared
machine's load average was 35); the device-bound stages did not move (student forwards 0.68,
commit 0.25) but the host-side parts did (frame fetch and packing +0.04 s per phase, launch
times +0.02 s, phase preparation +0.01 s), i.e. CPU contention from other users' jobs. On a host
without competing load the 4.87 s figure is the expected steady state; on a shared host the
config sits at the line, and a client should buffer 1-2 s. A further 5% of device time (the
LoRA delta, see above) would make the margin independent of host load.

**Two GPUs at persona length after the quality fix (measured locally, 2026-09-29, with
per-channel FP8 with BF16 boundary layers, merged student weights [since found to blur the
video, see `taomate_h3_lora_merge`; the deploy config now runs without them, estimate +0.12 s
per request, about 4.81 s], teacher and
text-encoder graphs, 64-token buckets, AdaLN cache off, `step_async_output`, 256 px decoder
tiles with 32 px minimum overlap):** a 100-request session with a different 332-340-token
prompt every request averaged **4.69 s per 4.958 s request (real-time factor 0.945)**, one
request above budget (5.61 s), and the stream gained 25.8 s over the 100 requests on a host
with load average 21; a 7-request run with timings on gave 4.60 s. Output matches eager BF16
(frames at 12, 120, 300 and 480 s checked). Per 34-frame phase: teacher 0.39 (request start
only), 3 student forwards 0.65-0.68, commit 0.23-0.25, VAE decode 0.19, 0.06 s between phases.
The two levers that closed the gap after the tile bug: the merged student weights (0.12 s per
request) and the 32 px tile overlap (0.6 s per request). Memory: the two ranks reported 139 and
143 GB in use (caching allocator included) on 144 GB cards; if a deployment runs short, quantize
the text encoder too (`text_encoder: {method: fp8_per_channel}`, about 4 GB per rank).

**Kernel choice, measured locally (2026-09-29, one GPU, the DiT's shapes at 3840 rows per rank):**
the served Cutlass per-token/per-channel FP8 GEMM runs at 1.27-1.46 PFLOP/s (qkv 0.61 ms, out
0.23 ms, fc1 0.86 ms, fc2 0.47 ms per block), within 5% of cuBLASLt's rowwise `torch._scaled_mm`
and about 1.9x faster than BF16 cuBLAS; the per-token quantization prologue costs 0.03-0.07 ms
per linear. FlashAttention-3 in BF16 runs the 3840x17000 attention of one block in 1.34 ms
(700 TFLOP/s); its FP8 path is 25% faster but changes the numerics and is not enabled. A
student forward is therefore about 108 ms of GEMM and 67 ms of attention out of 210 ms, both
near the cards' FP8/BF16 tensor-core rates, so no exact kernel substitution remains for the
DiT; the remaining exact lever is the VAE decode path.

**Video VAE decode profile (measured locally, 2026-09-29, one GPU, one 7-latent window, 8 tiles):**
174 ms, of which 159 ms of device time: 100 ms in the 3D ViT decoder's fp16 cuBLAS GEMMs, 33 ms
in its attention (PyTorch's flash SDPA), 7 ms `silu_and_mul`, 12 ms residual and norm kernels,
4 ms casts; the convolutions are negligible. Batching the tiles (`taomate_h3_vae_stack_tiling`)
gives identical output and no speed change, so the GEMMs are not starved by small batches: the
decoder is a 2.4 B-parameter 3D ViT (36 blocks, width 2048, FFN 16384, 32 heads of 64) and one
window is about 69 TFLOP of GEMM, i.e. the 100 ms already run at the cards' fp16 tensor-core rate
(about 690 TFLOP/s). Its attention (288 calls of 1797x1797, head dim 64) runs at about 220 TFLOP/s
and a FlashAttention-3 swap would save roughly 15 ms per window (0.03 s per request); nothing
larger remains in the decode without lower precision.

**On-server phase profile and two exact fixes (2026-09-29, two GPUs, one 34-frame phase, rank 0).**
Of about 1.2 s of device time per phase, the DiT's FlashAttention-3 took 300 ms and its FP8 GEMMs
259 ms; the rest was overhead: assembling keys and values for the streaming attention (a
`torch.cat` of the 120 MB history per layer per forward, stacks, gathers and index copies: about
135 ms), copies (63 ms), an inductor-compiled FP8 activation-quantization kernel (54 ms), the
Ulysses all-to-all (58 ms), a tile-decode barrier wait (56 ms) and small elementwise kernels
(47 ms). Two of these are removed at identical numerics: the streaming attention now keeps one
persistent K and V buffer per layer (history in front, the phase's rows copied behind it, a commit
only advances the history end, retention compacts a few contiguous runs), and the activation
quantization runs vLLM's CUDA op (`taomate_h3_fp8_quant_cuda_op`). Measured together on the same
server (exact output, merge off, 21 persona-length requests with a new prompt each): **4.58 s per
4.958 s request on average, 4.73 s at most, real-time factor 0.92**, against 4.97 s before the two
fixes; per 34-frame phase the three student forwards fell from 0.69 to 0.66 s, the commit from 0.26
to 0.23 s and the 17-frame phase from 0.83 to 0.71 s of period. A 100-request session on the same
server (new prompt every request, host load average about 30) averaged **4.74 s per request (warm
4.76 s), minimum 4.38 s, maximum 4.88 s, no request above 4.958 s**, and the stream gained 22 s over
the 100 requests; the test driver's last prompt update, scheduled on the wall clock, then arrived
after the stream had finished and was rejected with an error message, which is the expected
behaviour for an update to a completed session.

**What is left at exact numerics (2026-09-29).** With the DiT and the VAE both at tensor-core rate
and the K/V assembly and quantization overheads removed, the two-GPU config runs persona-length
prompts with a change every request at about 4.6 s per 4.958 s request (real-time factor 0.92,
an 8% margin). The remaining
speed-ups all change numerics and would need a quality gate against eager BF16 (sharpness and frame
difference on the persona prompt) before use: FP8 attention in the DiT (25% faster attention,
about 0.2 s per request), FP8 GEMMs in the VAE decoder (about 0.15 s per request), FP8 for the
DiT's boundary layers (small). They are not enabled.


**Live-agent demo, paced by the app (measured locally by the demo session, 2026-09-29, two GPUs, exact
path, 26 requests of a lesson, prompts of 373-444 tokens with the just-in-time hold):** median 4.905 s,
mean 4.954 s between requests' first chunks, 7 requests above 4.958 s. With the hold the app keeps
the server at most two requests ahead of playback, so this measures pacing rather than the server's
speed (4.58 s per request unpaced, above). One 7.3 s gap was a live graph capture for a 427-token
prompt outside the configured range: the demo's prompts span 373-444 tokens (p99 424 over 1233
prompts), so its range is now `"360-470"`; size the range to the deployment's prompt lengths, since
every length outside it costs about 2 s the first time it appears.

Teacher graphs are keyed by the prompt's token count because the H3 attention treats the
document's valid rows as a prefix whose length is a Python int of the forward (a fixed
text length would need extra rows inside that prefix, which changes the attention), so
the graphs of a length range are captured up front instead. Reusing the last denoise
step's K/V instead of the clean-commit forward (the LingBot-World `reuse_last_step_kv`
pattern) is not offered: TaoMate's last student forward runs at sigma 0.853 of the shift-12
schedule (the ladder is 1.0, 0.961, 0.853, 0), far from the clean K/V the model was trained
to attend to.

**Start-up time.** With the teacher and text-encoder graph captures for a 120-token range the two-GPU
server takes about 7 minutes to become ready (3 min of loading, 3-4 min of captures), which exceeds
vLLM-Omni's default stage initialization timeout of 600 s; raise it (for example 1800 s) or narrow
`taomate_h3_teacher_graph_text_lengths` to the deployment's prompt lengths.

## Limits

- One session per server (`max_num_seqs: 1`, `session_capacity: 1`).
- The canvas is fixed per deployment; a request with another size is rejected.
- Only text prompts (T2VA); the TaoMate release has no image or audio conditioning.
- Video is not bit-reproducible run to run (the release runtime has the same property).
