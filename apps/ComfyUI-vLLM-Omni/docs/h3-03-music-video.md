# H3-03: MiniMax-H3 music video from a supplied track

Turn a music track you supply into a multi-shot music video. Each shot is a
MiniMax-H3 Ref2VA request that uses a shared character or style image and its
own slice of the track as **driving audio**, so the visuals follow that part of
the music. ComfyUI then joins the shots and puts the original recording on the
final MP4. All inference runs on a remote vLLM-Omni service; ComfyUI does not
load H3 weights.

This is the supplied-audio branch of H3-03. The YuE2-generated-music branch
depends on the YuE2 serving API and ComfyUI music controls (CUI-02).

## Requirements

- ComfyUI **v0.36.0 or later**, for the core **Concatenate Video** node and its
  `complete_audio` input. The graph also uses the core Load Audio, Trim Audio
  Duration, Primitive, and Save Video nodes.
- ComfyUI-vLLM-Omni with the `audio_mode` option on **MiniMax-H3 Video Params**.
- A Ref2VA-capable H3 service: the combined service (repository ID, no
  `--task-type`) or one started with `--task-type ref2va`. Follow the
  [H3 recipe](../../../recipes/MiniMaxAI/MiniMax-H3.md) for your hardware.

## How it works

| Stage | Nodes | What happens |
| --- | --- | --- |
| Inputs | Load Audio, Load Image | The supplied track (WAV or MP3) and one character/style reference |
| Shot slices | Trim Audio Duration (one per shot) | Cuts consecutive 124-frame (5.1667 s) slices of the track |
| Shots | Video References → Generate Video (one per shot) | Ref2VA with `image_1` = the shared reference and `audio_1` = the shot's slice |
| Sync | MiniMax-H3 Video Params, `audio_mode=lock_source` | H3 uses the slice as the shot's soundtrack and keeps it fixed while sampling |
| Assembly | Trim Audio Duration, Concatenate Video, Save Video | Joins the shots in order and replaces their audio with the original track |

With `audio_mode=lock_source`, the server places the single driving audio in
the target audio latent, crops or zero-pads it to the shot duration, and keeps
it clean throughout sampling. The driving audio is not a `<Audio 1>` reference
tag; it is the shot's soundtrack. The node sends `audio_mode` only for
`lock_source`, so requests from other workflows are unchanged. The server
rejects `lock_source` unless exactly one audio is connected to Video
References.

## Use

Open **vLLM-Omni MiniMax-H3 Music Video.json** from **Templates →
ComfyUI-vLLM-Omni**, or drag the JSON onto the canvas.

1. Set **Server URL** and **Served model name** for your H3 service. Defaults:
   `http://localhost:8000/v1` and `MiniMaxAI/MiniMax-H3`.
2. Select your track in **Supplied music track** and your subject in
   **Character / style reference**. The template's filenames are placeholders
   relative to ComfyUI's input directory; no sample assets are bundled.
3. Write one prompt per shot. Name the subject `<Picture 1>` and describe the
   action, camera, and how the shot should follow the music, such as
   lip-syncing for vocals or movement on the beat.
4. Run the workflow. Save Video writes `output/video/MiniMax-H3-MV_*.mp4`.

### Shot timing

Each shot is 124 frames at 24 FPS (H3 requires `17k+5` frames), which is
124/24 = 5.1667 seconds. The shared **Shot duration** primitive feeds every
Generate Video and every slice length. Slice starts are set on each shot's
Trim Audio Duration node:

| Shot | Slice start (s) | Video time in the final MP4 (s) |
| --- | --- | --- |
| 1 | 0.00 | 0.00 – 5.17 |
| 2 | 5.17 | 5.17 – 10.33 |
| 3 | 10.33 | 10.33 – 15.50 |

The final soundtrack is trimmed to 3 × 5.1667 = 15.5 seconds from the same
start. Trim widgets step by 0.01 s, so a slice start is at most 0.005 s from
its exact boundary, well under one frame (0.042 s).

To start later in the song, add the same offset to every slice start and to
the full-soundtrack start. To add shots, duplicate a shot group, set its slice
start to the next boundary (`start + i × 5.1667`), connect its Generate Video to
the next free Concatenate Video input, and extend the full-soundtrack duration
by 5.1667 s per shot. To change the shot length, keep it on the `17k+5` frame
grid (for example 107, 124, 209 frames) and recompute every start and the
total duration.

### Replace or keep the shot audio

- **Replace (default):** Concatenate Video's `complete_audio` puts the
  original recording on the final video, so the output carries the exact
  source audio and has no seams between shots.
- **Keep shot audio:** disconnect `complete_audio`. Each shot then keeps the
  audio H3 returned: the audio VAE's reconstruction of its slice, not a
  byte-for-byte copy, which can differ slightly at shot boundaries.

`lock_source` keeps the driving audio fixed, so H3 does not add sound effects
or vocals on top of the track. To mix another recording with the music, combine
it with the full soundtrack using the core **Audio Merge** node before
`complete_audio`.

## Limits

- Each shot is a separate request. Identity comes from the shared reference
  image and the prompt; there is no latent continuity between shots, so
  expect cuts rather than continuous camera motion.
- A shot is 4–15 seconds (Ref2VA). Longer shots can use the server's
  `long_video` mode, which this workflow does not expose.
- The workflow has no prompt-assistance stage; H3 does not supply a prompt
  LLM.
- The graph's reference topology is not yet audited against the linked
  community example, so this template does not claim parity with it.

## Validation

Check the template's wiring and the client changes locally:

```bash
python -m pytest tests/e2e/features/comfyui/test_h3_music_video_workflow.py \
  tests/e2e/features/comfyui/test_video_references.py -q
```

For a real-model check, run the workflow against a served H3 model and inspect
the saved file:

```bash
ffprobe -v error -show_entries stream=codec_type,width,height,r_frame_rate,nb_frames,sample_rate,channels \
  -show_entries format=duration -of json "$OUTPUT_MP4"
```

Expect 372 frames (3 × 124) of 24 FPS H.264 video and an audio stream whose
duration matches the video (15.5 s). Watch and listen for lip-sync or beat
alignment within each shot, subject consistency across shots, and no audio
gaps at shot boundaries. Record the server commit, model checkpoint, ComfyUI
version, graph revision, seed, inputs, runtime, and output.
