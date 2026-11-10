# Video Endpoints

This package owns OpenAI-compatible video endpoint families.

## `video_api_utils.py` vs `video/generation/helpers.py` (no overlap)

| | `openai/video_api_utils.py` | `openai/video/generation/helpers.py` |
| --- | --- | --- |
| Layer | Shared video media utils | `/v1/videos*` endpoint helpers |
| Longevity | **Temporary.** TODO(#5227, P1.3): move/tidy into family `media.py` | **Longer home** for generation resource logic through P0.2/P0.3; P1.3 may split further |
| PUT HERE | Decode/encode URLs and bytes, frame/audio coercion, streaming encoders | Multipart form parsing, MiniMax upload limits, job runner/cleanup, `VideoResponse` factory, runtime context from app state |
| DO NOT PUT HERE | Form/job/store orchestration | Reimplemented encode/decode |
| Typical importers | `serving_video.py`, streaming output, generation helpers | `api_server.py` video routes (later `generation/api_router.py`) |
| May import the other? | **No** — must not import `video.generation.helpers` | **Yes** — may call `video_api_utils` |

**Prefer growing `generation/helpers.py`.** Do not grow root `video_api_utils.py` unless the symbol is clearly shared media.

## Put Here

- Video generation/resource endpoint code under `generation/`.
- Generated video streaming-output code under a dedicated subpackage when that
  route moves.
- Video-specific request parsing, response formatting, and storage/job helpers
  for the owning subpackage.

## Do Not Put Here

- Live streaming-input chat sessions such as `/v1/video/chat/stream`; those
  should get their own route-owner package in the router split rather than
  living under video generation.
- Generic server utilities.
- Shared media encode/decode that already lives in root `video_api_utils.py`.
- Image or audio endpoint behavior.
