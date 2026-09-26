#!/usr/bin/env bash
set -euo pipefail

# Push every .mp4 in a directory as its own RTSP stream (one ffmpeg per file),
# named rtsp://<base>/<filename-without-extension>.
#   bash ./rtsp_all.sh [video-dir] [rtsp-base-url]

VIDEO_DIR="${1:-./videos}"
RTSP_BASE="${2:-rtsp://127.0.0.1:8554}"

if [[ ! -d "$VIDEO_DIR" ]]; then
  echo "Video directory not found: $VIDEO_DIR" >&2
  exit 1
fi

shopt -s nullglob
VIDEO_FILES=("$VIDEO_DIR"/*.mp4)
shopt -u nullglob

if [[ ${#VIDEO_FILES[@]} -eq 0 ]]; then
  echo "No video files found in: $VIDEO_DIR" >&2
  exit 1
fi

echo "Found ${#VIDEO_FILES[@]} video(s) in $VIDEO_DIR"

PIDS=()
cleanup() {
  echo "Stopping all streams..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null
}
trap cleanup EXIT INT TERM

for video in "${VIDEO_FILES[@]}"; do
  stream_name="$(basename "${video%.*}")"
  rtsp_url="${RTSP_BASE}/${stream_name}"
  echo "Starting stream: $rtsp_url  <-  $(basename "$video")"
  ffmpeg \
    -hide_banner \
    -loglevel error \
    -re \
    -stream_loop -1 \
    -i "$video" \
    -vf "format=yuv420p,scale='min(1280,iw)':'min(720,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2" \
    -c:v libx264 \
    -preset veryfast \
    -tune zerolatency \
    -b:v 4000k \
    -maxrate 4500k \
    -bufsize 9000k \
    -c:a aac \
    -b:a 128k \
    -ar 44100 \
    -f rtsp \
    -rtsp_transport tcp \
    "$rtsp_url" &
  PIDS+=($!)
done

echo "All ${#VIDEO_FILES[@]} streams started. Press Ctrl+C to stop."
wait
