# Streaming Video to Low-Latency HLS

Packages the `WS /v1/realtime/video` fragment stream into a live LL-HLS playlist, so a
generation can be watched while it is still being generated, by ordinary players and
through a CDN.

This is transport and packaging only. It is model-agnostic: it never looks at the
prompt, the pipeline, or the model, only at the fragmented MP4 the server already
emits.

## Why add this alongside the existing example

`streaming_video_generation/` covers the protocol well, and this example does not
replace it. The two consumers there stop at different points:

| Existing | Ends at |
|---|---|
| `streaming_video_client.py` | Saves chunks, remuxes to a progressive MP4 after `session.done` |
| `gradio_demo.py` | Appends fragments through MSE in one attached browser |

Neither produces something a second viewer can open, and neither reports when a frame
first becomes playable. vLLM-Omni already emits fragmented MP4, which is one packaging
step from LL-HLS. This takes that step.

## Files

| File | Purpose |
|---|---|
| `fmp4.py` | Incremental fMP4 splitter. Emits the init segment once, then one buffer per `moof`+`mdat` |
| `llhls.py` | Publishes each fragment as an LL-HLS part, maintains the playlist, records latency |
| `hls_client.py` | Connects to the server and drives the two above |

WebSocket framing and MP4 box boundaries are independent, so the splitter accepts bytes
in any slicing rather than assuming one binary frame equals one fragment.

## Run

Start a server as in `streaming_video_generation/README.md`:

```bash
vllm serve BestWishYsh/Helios-Distilled --omni --diffusion-streaming-output --port 8000
```

Then:

```bash
pip install websockets

python hls_client.py \
  --host 127.0.0.1 --port 8000 \
  --model BestWishYsh/Helios-Distilled \
  --prompt "A serene lakeside sunrise with mist over the water." \
  --width 640 --height 384 --fps 16 --num-frames 99 \
  --out out/live

python -m http.server 8080 --directory out/live
# then open out/live/stream.m3u8 in any HLS player
```

## What it reports

```json
{
  "parts": 11,
  "media_seconds": 6.188,
  "realtime_factor": 10.281,
  "publish_latency_ms": { "mean": 0.771, "max": 2.903 },
  "live": {
    "prompt_to_first_byte_s": 0.0,
    "prompt_to_first_playable_part_s": 0.0
  }
}
```

`prompt_to_first_playable_part_s` is the term the existing clients cannot report: how
long after the prompt before anything is watchable. Render time is only the first part
of that.

## Verification

`test_hls_packaging.py` runs without a GPU, a server, or ffmpeg. It builds fMP4 box
structures in memory and asserts the splitter reassembles them identically across
framings from whole-buffer down to one byte at a time, and that the playlist references
only files that exist.

```bash
python -m pytest examples/online_serving/streaming_video_hls/test_hls_packaging.py
```

The packaging path has separately been checked end to end against an ffmpeg-produced
fMP4 source of the same shape the server emits (640x384, 16fps, 9-frame chunks): ffmpeg
decodes the resulting playlist at exit 0, ffprobe reads back all 96 frames and 6.1875s,
and the round trip is lossless.

## Known limits

- **Not yet exercised against a live server.** Development used a synthetic source of
  the same shape, so the protocol handling in `hls_client.py` is the part still wanting
  a real run. The splitter and packager are covered by the test above.
- Single rendition; no bitrate ladder.
- No transcode anywhere. Fragments are written verbatim, which keeps latency low and
  leaves output quality identical to what the server produced.
