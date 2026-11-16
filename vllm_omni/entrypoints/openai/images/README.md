# Image Endpoints

This package owns OpenAI-compatible image generation and image edit endpoint
logic.

## `image_api_utils.py` vs `images/helpers.py` (no overlap)

| | `openai/image_api_utils.py` | `openai/images/helpers.py` |
| --- | --- | --- |
| Layer | Shared media utils | Image endpoint helpers |
| Longevity | **Temporary.** TODO(#5227, P1.2): absorb/tidy into this package (e.g. `media.py`) | **Longer home** for `/v1/images*` logic through P0.2/P0.3; P1.2 may split further |
| PUT HERE | `parse_size`, encode/base64, layered-layer validation — plain Python, no FastAPI `Request` | Engine/app-state limits, input image loading from the edit request, Hunyuan edit extras, result extraction/normalization |
| DO NOT PUT HERE | Request/engine/job helpers | Shared encode/`parse_size`/layered validation |
| Typical importers | `protocol/images.py`, `serving_chat.py`, image routes | `api_server.py` image routes (later `images/api_router.py`) |
| May import the other? | **No** — must not import `images.helpers` | **Yes** — may call `image_api_utils` |

**Prefer growing `helpers.py`.** Do not grow root `image_api_utils.py` unless the symbol is clearly shared media with no endpoint semantics.

## Put Here

- Request parsing, validation, and response helpers used only by image
  endpoints (`helpers.py`).
- Image endpoint route bodies after route extraction.
- Image-specific bridge/adapters introduced during the image refactor.

## Do Not Put Here

- Generic server utilities.
- App-state accessors shared by multiple endpoint families.
- Shared encode/size/layered validation already in `image_api_utils.py`.
- Video or audio endpoint behavior.
