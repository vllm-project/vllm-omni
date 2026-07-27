# Realtime Video Prompt Interaction Protocol

This document specifies the prompt-update subset of the realtime video
interaction protocol exposed by:

```text
WebSocket /v1/realtime/video
```

A WebSocket connection contains exactly one video-generation request. Clients
therefore do not send a request or session ID with an interaction. The server
assigns a `request_id` in `video.start` and includes it in subsequent
server-to-client messages.

## Interaction Request

After `video.start`, a client may submit a prompt update:

```json
{
  "type": "session.interaction",
  "interaction": {
    "event_id": "ui-update-42",
    "event": {
      "prompt": "A snowy street illuminated by warm lanterns"
    },
    "transition_chunks": 3
  }
}
```

`interaction` has the following fields:

- `event_id` (optional string): session-unique idempotency key. The
  server preserves a client-provided value or generates a UUID when omitted.
- `event.prompt` (required string): non-empty target prompt.
  over which the prompt reaches its target.
- `transition_chunks` (optional non-negative integer): model-generation chunks
  over which the prompt reaches its target. If omitted, the model's default transition is used.

Note: Although `["interaction"]["event"]` only contains `prompt`, still consider using `OmniTextPrompt` for this part (with total=false) so that other fields will be supported in the future. Currently, `vllm_omni/diffusion/worker/diffusion_model_runner.py` already takes care of enforcing subset format, routing to meaningful input handing routines, and raising `NotImplementedError`. Only need to adapt this design to the input data structure above.

### Transition Semantics

Prompt transitions are target updates and begin at the next eligible model
generation boundary after the interaction is processed.

- `transition_chunks = 0` is a hard update: the first affected generation
  chunk uses the target prompt.
- `transition_chunks = N`, where `N > 0`, reaches the target in the `N`th
  affected generation chunk. For example, a three-chunk linear transition uses
  progress values `1/3`, `2/3`, and `1`.

### Idempotency and Replacement

Within a session:

- repeating an `event_id` with the same payload is a retry and must not
  apply the update twice;
- repeating an `event_id` with a different payload is rejected with
  `event_id_conflict`;
- a newer prompt target supersedes any older pending prompt target;
- when a new target interrupts an active transition, its transition starts
  from the prompt state resolved at its effective boundary.

## Queue Acknowledgement

The server quickly acknowledges admission to the interaction queue:

```json
{
  "type": "session.interaction.queued",
  "request_id": "video_stream-...",
  "event_id": "ui-update-42"
}
```

`queued` confirms syntax validation, idempotency handling, and queue admission.
It does not claim that the model has scheduled or applied the update. A later
failure is reported as a correlated error:

```json
{
  "type": "error",
  "request_id": "video_stream-...",
  "event_id": "ui-update-42",
  "code": "interaction_failed",
  "message": "The active model does not support prompt updates"
}
```

## Video Chunk Metadata

Each emitted binary payload is preceded by a JSON metadata message. The
metadata message and its binary payload form one ordered pair and must not be
interleaved with another outbound message.

```json
{
  "type": "video.chunk_metadata",
  "request_id": "video_stream-...",
  "kind": "media",
  "transport_chunk_index": 7,
  "generation_chunk_index": 6,
  "num_frames": 9,
  "byte_length": 184320,
  "started_event_ids": ["ui-update-42"],
  "active_event_ids": ["ui-update-42"],
  "completed_event_ids": []
}
```

- `transport_chunk_index` identifies the binary WebSocket payload.
- `generation_chunk_index` identifies the model-generation chunk represented
  by the payload.
- `started_event_ids` lists interactions that first affect this
  generation chunk.
- `active_event_ids` lists interactions contributing to this generation
  chunk, including an ongoing transition.
- `completed_event_ids` lists interactions that reach their target in
  this generation chunk.

Encoder trailer bytes are emitted as:

```json
{
  "type": "video.chunk_metadata",
  "request_id": "video_stream-...",
  "kind": "trailer",
  "transport_chunk_index": 8,
  "generation_chunk_index": null,
  "num_frames": 0,
  "byte_length": 824,
  "started_event_ids": [],
  "active_event_ids": [],
  "completed_event_ids": []
}
```
