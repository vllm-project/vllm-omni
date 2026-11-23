# OpenAI Entrypoint Helpers

This package contains OpenAI-compatible serving code for vLLM-Omni.

## Hard Rule: `*_utils*` vs package `helpers.py`

These are **different layers**. During the #5227 refactor they coexist on purpose.
**Do not mix them. No ownership overlap.**

| Layer | Files | Longevity | PUT HERE | DO NOT PUT HERE |
| --- | --- | --- | --- | --- |
| Shared media / serving utils | root `image_api_utils.py`, `video_api_utils.py`, `audio_utils_mixin.py`, `utils.py` | **Temporary.** TODO(#5227, Phase 1): tidy up / absorb into endpoint-family packages in Phase 1 modality PRs (P1.1-P1.3) | Codec / bytes / format / pure media conversion; cross-cutting stage/pipeline helpers reused by protocol + multiple serving paths | HTTP form parsing, FastAPI `Request` / app-state orchestration, job runners, endpoint response factories from `api_server.py` |
| Endpoint-family helpers | `images/helpers.py`, `video/generation/helpers.py`, … | **Longer home** through helper (P0.2) + router (P0.3) stages; Phase 1 PRs may split further, not delete ownership | Request/job/runtime helpers peeled out of `api_server.py` for one endpoint family | Encode/decode/base64/media backends that already belong in root `*_utils*` |

### Allowed dependency direction

```text
api_server / serving / protocol
        │
        ├──► package helpers.py     (endpoint orchestration; longer home)
        │         │
        │         └──► root *_utils*   (shared media / format; temporary)
        │
        └──► root *_utils*            (direct use OK)
```

- `helpers` **may import** `*_utils*`
- `*_utils*` **must not import** package `helpers`
- Never copy a symbol into both layers; pick one owner

### Where to put new code (until family PRs finish)

1. Pure image/video/audio bytes or format conversion with **no** FastAPI `Request` / job store / route semantics → root `*_utils*` only if it must stay shared; prefer not growing these files.
2. Endpoint request parsing, validation against engine/app state, job lifecycle, or response shaping that used to sit beside routes in `api_server.py` → owning package `helpers.py` (**prefer this**).
3. Cross-family OpenAI app-state / bootstrap → `app_state.py` / similar top-level OpenAI helpers — **not** into modality `*_utils*`.
4. If unsure: **do not grow root `*_utils*`**. Put it in the owning package `helpers.py` and leave a TODO(#5227) for the modality PR.

**TODO(#5227, Phase 1):** root `image_api_utils.py`, `video_api_utils.py`, `audio_utils_mixin.py`, and likely pieces of `utils.py` should be tidied, moved, or absorbed during Phase 1 audio / images / video family PRs (P1.1-P1.3). Package `helpers.py` files stay as the endpoint ownership surface until those PRs further split them.

## What Belongs Here (top-level OpenAI helpers)

- OpenAI protocol or response-shape helpers shared by multiple OpenAI endpoint
  families.
- App-state accessors for OpenAI serving objects, for example helpers that read
  `request.app.state.openai_serving_*`.
- Bootstrap helpers that are specific to OpenAI-compatible serving and are not
  route bodies.

## What Does Not Belong Here

- FastAPI app/server mechanics that are not OpenAI-specific. Put those under
  `vllm_omni.entrypoints.serve.utils`.
- Modality-specific request parsing, validation, or response formatting. Put
  those under the owning endpoint package, such as `images/` or
  `video/generation/`.
- Broad cleanup/rename of existing `image_api_utils.py`, `video_api_utils.py`,
  `audio_utils_mixin.py`, and `utils.py` in this helper-only stage (P0.2) is out
  of scope; tracked by #5227 Phase 1 modality PRs instead.
