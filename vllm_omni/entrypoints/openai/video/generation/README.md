# Video Generation

This package owns the `/v1/videos` resource family.

## Helpers vs root video utils

See also `../README.md` and `../../README.md`.

| | PUT HERE | DO NOT PUT HERE | Longevity |
| --- | --- | --- | --- |
| `helpers.py` | Form parse, upload validation, job run/cleanup, response factories, app-state runtime context for `/v1/videos*` | Shared encode/decode backends | **Longer home** through P0.2/P0.3; later P1.3 PR may split (form/jobs) |
| `openai/video_api_utils.py` | Shared encode/decode/media backends only | Form/job/store orchestration | **Temporary.** TODO(#5227, P1.3): move into e.g. `media.py` |

- Helpers call utils; utils never grow endpoint/job logic.
- Prefer growing `helpers.py` over root utils.

## Put Here

- Multipart form parsing for video generation requests.
- Async/sync video generation job helpers.
- Video resource status, delete, retrieve, and download helpers.
- The `api_router.py` for `/v1/videos*` once route bodies move out of
  `api_server.py`.

## Do Not Put Here

- Live video-chat streaming input.
- Generated video streaming-output sessions.
- Generic video encoding/decoding helpers used outside this resource family
  (keep those in `video_api_utils.py` until the P1.3 video family PR moves them).

The P0.2 helper migration places non-route video generation code here first. Full
route extraction (P0.3) and `video_api_utils.py` cleanup (P1.3) are tracked by #5227.
