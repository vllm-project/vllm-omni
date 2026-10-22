# Media handoff

Drag
[`vLLM-Omni Media Handoff.json`](../example_workflows/vLLM-Omni%20Media%20Handoff.json)
onto the canvas. It connects a saved H3 or restored video to an external
soundtrack. Choose your files in **Load Video** and
**Load Audio**; the example filenames are placeholders in ComfyUI's input folder.

Queue once to save four outputs under `output/CUI03/`:

| Output | Audio policy |
| --- | --- |
| `preserve` | Save the original VIDEO with its original audio. |
| `replace` | Use only the supplied soundtrack with the source VIDEO. |
| `mix` | Use AudioMerge's `mean`: average the source and supplied waveforms. |
| `loop` | Repeat the supplied soundtrack to fill the source video, then replace the source audio. |

**Save Video** clips audio longer than the video. A shorter replacement
soundtrack ends early; the remaining video has no soundtrack.
The loop branch uses **Audio to Video Duration**, which computes the target
sample count from the video's frame count, rational frame rate and the audio's
sample rate. It rounds up by at most one sample to cover the last video frame.
It repeats only the sample axis, preserving batch, channels, sample rate and
sample values. The join is hard, without a crossfade.

Change its `short_audio` widget to `pad` to append silence, or `trim` to leave
short audio unchanged. All three modes cut longer audio to the target length.
The node does not change the input VIDEO or resample the supplied AUDIO.
It targets constant-frame-rate media such as H3 output; a variable-frame-rate
source requires conversion before using frame-count/frame-rate timing.
Use **Audio Adjust Volume**
before **AudioMerge** when you need a different balance.

The three soundtrack branches use **Concatenate Video** with a single
`video0` input and `complete_audio` to override the soundtrack. Keep its codec
and **Save Video**'s codec set to `auto`. For an encoded file VIDEO, this copies
the video packets into the output container instead of rebuilding RGB frames.
It retains encoded video quality and frame spacing; ComfyUI normalizes the
clip start to zero. The container and encoded audio change.
**Get Video Components** extracts the original audio for mixing;
its IMAGE/FLOAT outputs are not used to reconstruct the video.

A tensor-backed VIDEO from a generation/restoration node must first be encoded
by ComfyUI. This graph cannot recover encoding or timestamps already discarded
by a producer. Stream-copy behavior applies to encoded inputs supported by
ComfyUI's output container; choose MP4 as in the template.
Replacement preserves the supplied audio's sample rate and channel count.
Mixing uses the standard AudioMerge sample-rate conversion and normalization;
a mono supplied track broadcasts across an H3 stereo source. If the source has
no audio, AudioMerge passes the supplied soundtrack through. The source-video
branch remains separate, so changing the soundtrack does not discard the
original output.

## Connect remote services

To process a fresh generation, connect **Generate Video → VIDEO** to
all three **Concatenate Video → video0** inputs,
**Get Video Components → video**, **Save original → video**, and
**Audio to Video Duration → video**, replacing **Load Video**.
The same sockets accept a restoration node's VIDEO output.
An AUDIO returned by **Generate Music** may replace the Load Audio output.
Each generation/restoration node must be configured for its own service URL
and served model. The media nodes operate on returned media; they do not load
or switch remote model weights.

H3 generation setup is covered by the [MiniMax-H3 recipe](../../../recipes/MiniMaxAI/MiniMax-H3.md).
This saved-media template can be validated on CPU after generation.

## Compatibility and validation

Use a ComfyUI version with **Concatenate Video**'s `complete_audio` input.
The validation baseline is ComfyUI
`06c67806bcb36789a1e8e1278c6a764b389ff858`. Install this extension's
**Audio to Video Duration** node for the loop branch. The other media nodes
are built into ComfyUI.

After running the graph, inspect each MP4:

```bash
ffprobe -v error -count_frames \
  -show_entries stream=codec_type,nb_read_frames,r_frame_rate,time_base,duration,sample_rate,channels \
  -of json output/CUI03/replace_00001_.mp4
```

Compare video frame count, FPS, time base and frame timestamps with the source;
for encoded MP4 inputs, also compare video packet contents to confirm that
no video re-encoding occurred. Check the intended audio
sample rate/channel count, listen to the selected audio policy, and watch for
audio/video timing differences. Test both a longer soundtrack and a shorter
one so clipping and early audio termination are visible. Record the source H3
server/model revision, source file, ComfyUI revision and graph revision with
the results. Workflow-schema checks alone do not establish media correctness.
