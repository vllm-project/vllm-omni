# RTSP local streaming

Simulate an RTSP camera from a local video file so the WebUI (or any RTSP client) can
test the RTSP input path without a physical IP camera. RTSP is a **webui-side input** —
the browser pulls the stream and feeds frames to the orchestrator over the normal API;
no serving-layer code is involved.

Needs [`ffmpeg`](https://ffmpeg.org/) and [MediaMTX](https://github.com/bluenviron/mediamtx/releases)
(download the binary + `mediamtx.yml` and keep them next to these scripts, or point
`MEDIAMTX_BIN` / `MEDIAMTX_CONFIG` at them).

## 1. Start the RTSP server

```bash
bash ./mediamtx.sh        # MediaMTX, listens on :8554
```

## 2. Push a local video as an RTSP stream

```bash
bash ./rtsp.sh ./videos/example.mp4 rtsp://127.0.0.1:8554/fire1
```

`rtsp.sh` takes `[video-path] [rtsp-url]` (or set `VIDEO_PATH` / `RTSP_URL`). To stream a
whole directory of clips, one stream per file:

```bash
bash ./rtsp_all.sh ./videos rtsp://127.0.0.1:8554
```

## 3. Use it in the WebUI

Enter the URL in the WebUI RTSP box:

```
rtsp://127.0.0.1:8554/fire1
```

If the WebUI runs on another machine, replace `127.0.0.1` with the MediaMTX host's IP.

## Notes

- `rtsp.sh` transcodes audio to AAC; replace `-c:a aac -b:a 128k -ar 44100` with `-an`
  if the source clip has no audio track and your `ffmpeg` errors on it.
- If a client cannot connect, check that port `8554` is reachable and not blocked by a
  firewall, container network, or security group.
