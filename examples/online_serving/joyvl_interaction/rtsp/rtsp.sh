#!/usr/bin/env bash
set -euo pipefail

# Loop a local video file and push it to an RTSP URL, simulating an RTSP camera.
#   bash ./rtsp.sh [video-path] [rtsp-output-url]
# Defaults stream ./videos/example.mp4 to rtsp://127.0.0.1:8554/fire1

VIDEO_PATH="${1:-${VIDEO_PATH:-./videos/example.mp4}}"
RTSP_URL="${2:-${RTSP_URL:-rtsp://127.0.0.1:8554/fire1}}"

if [[ ! -f "$VIDEO_PATH" ]]; then
  echo "Video file not found: $VIDEO_PATH" >&2
  exit 1
fi

echo "Input video: $VIDEO_PATH"
echo "RTSP output: $RTSP_URL"

# Drop "-c:a aac -b:a 128k -ar 44100" for "-an" if the source has no audio track.
exec ffmpeg \
  -hide_banner \
  -loglevel info \
  -re \
  -stream_loop -1 \
  -i "$VIDEO_PATH" \
  -vf "scale='min(1280,iw)':-2" \
  -c:v libx264 \
  -preset veryfast \
  -tune zerolatency \
  -b:v 2500k \
  -maxrate 2500k \
  -bufsize 5000k \
  -c:a aac \
  -b:a 128k \
  -ar 44100 \
  -f rtsp \
  -rtsp_transport udp \
  "$RTSP_URL"
