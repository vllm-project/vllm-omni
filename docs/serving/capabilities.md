# Capability discovery

`GET /v1/omni/capabilities` describes the model and endpoints of a running
vLLM-Omni deployment. Clients can select a compatible request before submitting
generation. The endpoint uses the server's usual API-key authentication.

```bash
curl -H "Authorization: Bearer $VLLM_API_KEY" \
  http://localhost:8091/v1/omni/capabilities | jq .
```

The response has `object: "omni.capabilities"`, `schema_version: 1`, an
`observed_at` UTC timestamp, and three collections:

| Collection | Contents |
| --- | --- |
| `models` | Public model IDs from the same base-model list used by `/v1/models`, their stage IDs, and all six operations. Aliases describe the same deployment. |
| `stages` | Original stage IDs, stage type, and `depends_on` stage IDs. IDs are strings and need not be consecutive. |
| `routes` | Transport (`http` or `websocket`), method, path, and exposure. WebSocket entries use method `WEBSOCKET`. |

Model paths, download locations, credentials, and GPU configuration are not
part of this public response.

## Choosing an operation

Each operation has independent `support`, `exposure`, `profiles`, `reason_code`,
and `execution` fields.

| Operation | Profiles |
| --- | --- |
| `chat` | `text_response`, `image_response`, `audio_response` |
| `image_generation` | `text_to_image` |
| `image_edit` | `image_to_image` |
| `video_generation` | `text_to_video`, `image_to_video`, `video_to_video` |
| `speech` | `text_to_speech`, `lyrics_to_music`, `caption_to_audio` |
| `audio_generation` | `text_to_audio` |

`support` is `supported`, `unsupported`, or `unknown`. It describes an
implemented request contract for the selected model topology. Output modality
alone does not determine support: producing images does not establish editing,
and producing audio does not establish ordinary TTS. Internal decoder stages,
robot actions, segmentation, and depth estimation do not acquire one of these
operations merely by being present in a registry.

`exposure` is `enabled`, `disabled`, or `unknown`. It describes the initialized
endpoint and deployment policy. A supported model operation can have disabled
routes. An installed route can coexist with an unsupported model operation.
Operation exposure is enabled when at least one of its routes is enabled;
clients should also check the particular route they intend to call. For
example, multiple API workers disable the process-local asynchronous video
store while the synchronous video endpoint can remain enabled.

Each profile has its own support state, `required_inputs`, `outputs`, and route
references. It can include additional `constraints`. Required inputs describe
media and semantic inputs in addition to normal protocol fields; they do not
replace the endpoint's request schema. Check the profile as well as the
operation: a music model serving `/v1/audio/speech` can support
`lyrics_to_music` while rejecting `text_to_speech`. Audex TTA serves captions
through `caption_to_audio` on the speech endpoint. `/v1/audio/generate` has a
separate diffusion request contract.

`outputs` describes generated media modalities, including an audio track in a
video container. MiniMax-H3, LTX-2, Kandinsky6, and Wan S2V produce both
`video` and `audio`; Kandinsky6 also permits disabling audio per request.
`constraints.optional_outputs` identifies a modality that requires additional
request and checkpoint support; Cosmos3 sound generation is optional.

Missing declarations remain unknown. The built-in declarations distinguish
checkpoint-dependent paths and undeclared chat prompt contracts explicitly.
An external Diffusers pipeline or an older plugin without a declaration is
unknown. Clients should retain their existing manual configuration when
discovery is unknown or the endpoint is absent on an older server.

## Request constraints

The optional, typed `request` object adds input roles and implemented parameter
limits without changing `schema_version: 1`. Existing flat `constraints` and
`required_inputs` remain available. Missing facts mean unknown, including a
missing maximum; they do not mean zero or unlimited verified support.

`profile.request` describes the model's implemented contract.
`profile.routes[].request` is the **complete effective contract for that route**,
including deployment and adapter limits. Clients use the selected route's
object directly. They must not inherit a model contract onto a route without
`request`, or combine facts from different routes. A route reference also has
its own `exposure`: a video model requiring reference audio can have an enabled
HTTP route and a disabled WebSocket profile.

| Field | Meaning |
| --- | --- |
| `inputs.<role>.support` | Whether this path consumes the role: `supported`, `unsupported`, or `unknown`. |
| `inputs.<role>.min_count`, `max_count` | Optional inclusive counts. `min_count: 0` means optional; an omitted maximum is unknown. |
| `required_any_of` | Every inner list requires at least one of its roles. All lists must be satisfied. |
| `parameters.<name>` | Optional `support`, scalar `type`, `enum`, inclusive `minimum`/`maximum`, and `step`. These describe accepted inputs, not quality recommendations. |
| `mask` | Native mask semantics and its relationship to source images. See below. |
| `max_output_pixels` | Optional deployment limit from `--max-generated-image-size`, not a universal model resolution limit. |

For example, CosyVoice3's `required_any_of` is
`[["voice", "reference_audio"], ["voice", "reference_text"]]`: a registered voice
satisfies both groups; an inline reference needs both its audio and transcript.
Qwen3-TTS Base also accepts `speaker_embedding`. These are semantic role names;
the speech endpoint's wire fields remain `ref_audio`, `ref_text`, and
`speaker_embedding`.

### Images and native masks

Image counts distinguish single-source editing from multiple reference images
within `image_to_image`; they do not create additional operation IDs. Z-Image
has a real single-image editing path. Klein permits up to four ordinary image
inputs. The effective maximum also honors the deployed image handler's limit.

Klein's native single-stage edit route additionally accepts `mask_image` and
`reference_image` as separate multipart uploads or data-URI form fields.
The latter is an independent conditioning reference, outside the ordinary
`image` list. When a mask is supplied, there must be exactly one ordinary source
image and its dimensions must match the mask. White edits, black preserves, and
the worker binarizes grayscale at 0.5. The discovery shape is:

```json
{
  "support": "supported",
  "source_role": "image",
  "max_source_count": 1,
  "white": "edit",
  "black": "preserve",
  "binarize_threshold": 0.5,
  "requires_same_size": true,
  "geometry": { "state": "unknown" },
  "source_resize": {
    "max_pixels": 1048576,
    "policy": "downscale_preserve_aspect_ratio"
  }
}
```

`geometry.multiple_of` is included only with `state: "known"`. It comes from
the existing initialized Klein postprocessor's VAE configuration. The value
is carried with the completed READY handshake, including remote replicas;
inline execution uses the same initialized hook. Discovery reads the current
local replica snapshots and returns unknown if any active replica lacks the
fact or disagrees. It does not load a VAE or read its configuration on query.

The source resize threshold is 1,048,576 pixels. It is not an output-size cap.
A client requiring pixel-aligned selection editing should use a source/output
canvas divisible by the reported multiple, with matching mask dimensions and
no source downscaling. It can preserve its existing composition workflow when
the geometry is unknown or those conditions are unmet.

Multi-stage image editing currently forwards ordinary images through chat and
cannot carry the two native fields. Supplying either now returns 400 before
media loading. Other built-in image owners explicitly exclude unimplemented
native mask/reference roles. For an external Diffusers backend the frontend
leaves these facts unknown; the worker rejects a supplied field when its loaded
pipeline signature cannot accept it, rather than silently discarding it.

### Other operation families

| Family | Current source-backed facts and boundaries |
| --- | --- |
| `chat` | Declared text/media requirements and response modalities are preserved. Unspecified optional inputs and reference counts stay unknown; an image output does not imply an image-edit input contract. |
| `image_generation` | The image route accepts positive dimensions and output format; it has no image/mask/separate-reference mapping. Declared model limits cannot be widened by the route. |
| `video_generation` | HTTP reference roles remain separate from the WebSocket's single-image input. Neither route maps native Wan `last_image`, so the optional offline last-frame role is excluded on these routes. MAGI-2 declares 125 frames, 12.5 fps, 10 seconds and its two native resolution modes. Joint audio output remains in `outputs`; optional audio remains in `constraints.optional_outputs`. |
| `speech` | Each TTS adapter owns its accepted reference, voice and instruction facts beside its request builder. Loaded voice enums include validated aliases such as Kokoro's `default` only when its target exists. Qwen language enums reflect actual input validation; the generic default language list is not promoted for other adapters. |
| `speech` music/caption profiles | MiniMax Music3/YuE2 require lyrics and style instructions, permit only the placeholder `default` voice, and exclude ordinary TTS reference controls. Audex TTA takes captions. MOSS variants separately expose reference cloning, sound-effect descriptions or voice-design instructions. |
| `audio_generation` | The diffusion audio handler maps a text caption and optional negative prompt. It has no lyrics/instruction pair, voice or reference-media mapping. |

FLUX.3 declares image responses through chat, text-to-image generation, and
image editing. Chat permits zero to ten reference images; edits require at
least one and apply the deployment's image limit. The image-generation route
does not accept reference images. Its current fixed image VAE requires output
dimensions divisible by 16, and each request produces one image. Native masks
and separate reference fields are unsupported. The loaded package selects CFG
or flow-map sampling: flow-map rejects explicit `guidance_scale` and nonempty
`negative_prompt`. Those parameter capabilities remain unknown to the frontend,
which does not read worker package configuration during discovery.

Voxtral reference-encoder availability is unknown when it has not already been
observed by the normal request path. Discovery does not issue the RPC used to
obtain it. Role facts do not replace the adapter's full request validator:
conditional transcript pairing and model-specific extra parameters remain
validated by the existing endpoint. Adding these facts does not create a
general-purpose form schema or promise inference quality.

## Execution observations

`execution.state` is `unpaused`, `paused`, or `unknown`. This observes the
frontend admission condition used by `AsyncOmni.generate`. A partial wake with
remaining sleeping memory tags is still paused. Clients can avoid submitting
new work until the applicable execution gate becomes unpaused.

`execution.scope` is `deployment` only when one API process serves the
deployment. Otherwise it is `worker`, so a paused observation must not disable
other API workers. Duplex session admission is a separate lifecycle and reports
unknown execution here. `observed_at` identifies when the observation was made.
An unpaused gate does not promise sufficient GPU memory, a healthy worker, or
successful generation; normal request errors remain authoritative.

The query reads initialized declarations, route policy, local replica metadata,
and the local execution condition. It does not reload configuration, resolve a checkpoint, download
files, perform health RPCs, wake workers, or run generation.

## Maintaining model declarations

The immutable types and standard profiles live in
`vllm_omni/serving_capabilities.py`. A native model owns its declaration in its
`PipelineConfig.serving_capabilities`; a generic diffusion model owns it in
`DiffusionModelMetadata.serving_capabilities`. A resolver must select the
declaration along with its topology or checkpoint variant. The engine retains
the final resolver result for API initialization.

An optional `serving_capabilities_resolver` names a pure startup function with
signature `(declaration, resolved_stages) -> ServingCapabilities`. It consumes
the final stage fields already in memory, before API initialization. It must
not load a model, resolve paths, read checkpoint files, or perform RPCs. The
same field on diffusion metadata is copied to the generic diffusion topology.
Older plugins omit it and retain their existing behavior.

### Configuration-dependent contracts

| Model/configuration | Declaration and source |
| --- | --- |
| Qwen3-Omni audio-output branch | The existing HF-config resolver selects a full audio topology or the text-only topology. Both own separate declarations. See `model_executor/models/qwen3_omni/pipeline.py`. |
| Audex S2S / thinker-only / TTS / TTA | Decoder presence and the selected entry stage distinguish chat, ordinary speech, and caption-to-audio. See `tts_adapters/audex.py`, `tts_adapters/audex_tta.py`, and the Audex recipe. |
| MOSS-TTS / SoundEffect / VoiceGenerator | Startup uses the same variant dispatch as the speech adapter. SoundEffect requires `ambient_sound`; VoiceGenerator requires text and `instructions`; ordinary TTS requires reference audio or a registered voice. See `tts_adapters/moss_tts.py`. |
| MiniMax-H3 FL2VA / Ref2VA | Explicit `task_type` excludes incompatible profiles. Remaining profiles are unknown with `worker_checkpoint_tasks_unavailable`: the release task list is held only by the worker. See `diffusion/models/minimax_h3/pipeline_minimax_h3.py`. |
| Cosmos3 normal / distilled / Edge | T2I and I2V are common modes. T2V and V2V are unknown with `worker_scheduler_mode_unavailable`, because the frontend does not own the loaded scheduler/backbone mode. Policy deployments have a separate action-only declaration. See `diffusion/models/cosmos3/pipeline_cosmos3.py`. |
| Wan generic / explicit I2V / TI2V | Generic I2V requires checkpoint `expand_timesteps`; explicit task-specific pipelines have separate declarations. See the Wan pipeline definitions. |
| Lance image / video checkpoint | The existing checkpoint resolver selects the matching declaration. I2V is unsupported for the video checkpoint: the current HTTP adapter supplies `image`, while the pipeline requires `first_frame`. Its image-generation and editing contracts remain unknown. See `config/config_factory.py` and the Lance pipeline. |
| Thinker followed by a separate TTS model | A downstream talker does not establish the standalone speech input contract. Aura, JoyAI, and complete Ming Omni topologies retain chat audio; their standalone TTS deployments declare speech separately. |

These are source-derived contracts, not per-checkpoint inference benchmarks.

`declare_capabilities` declares a complete set: omitted standard profiles are
explicitly unsupported. Use a reasoned unknown profile when an actual branch
cannot be established from resolved configuration. A plugin with a partial
declaration can construct `ServingCapabilities` directly; omitted operations
then stay unknown. Every built-in declaration includes repository-relative
source references for review. Those references are not returned to clients.

Typed request facts live in `vllm_omni/request_capabilities.py`. They remain
serializable dataclasses so `PipelineConfig` JSON serialization is unchanged.
Diffusion owners attach facts to their profiles; speech adapters implement
`get_request_constraints()` using only their initialized state. Adapter methods
must not call `load_capabilities()`, tokenizers, filesystem readers or RPCs.
The serving projection applies the actual route's restrictions once at app
initialization. Optional runtime geometry uses `ServingPostProcess` and
`RuntimeServingMetadata`; a pipeline without these retains unknown geometry.

Adding a supported model requires updating its declaration and the relevant
positive and negative regression cases. Registry coverage tests ensure aliases
reach declarations without importing concrete diffusion model implementations.

```bash
pytest tests/config/test_serving_capabilities.py \
  tests/entrypoints/openai_api/test_serving_capabilities.py \
  tests/entrypoints/openai_api/test_request_capabilities.py \
  tests/diffusion/test_startup_serving_metadata.py \
  -m cpu --run-level=core_model
```

These tests need the normal vLLM-Omni Python dependencies and a compatible vLLM
installation. They run on CPU with local engine/config fixtures and no model
weights. Generation quality and device memory measurements belong to model
inference tests.
