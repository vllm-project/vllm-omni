# Experimental Full-Duplex Runtime

This package contains three experimental integrations:

- the existing JoyVL framework and example integration;
- the MiniCPM-o 4.5 native audio path used by `/v1/duplex` and
  `/v1/realtime?duplex=1`;
- the PersonaPlex lockstep speech-to-speech path (browser demo + batched
  serving, see `recipes/NVIDIA/PersonaPlex.md`).

For the MiniCPM active runtime path, lifecycle invariants, capability boundary,
Realtime response contract, and validation scope, see [`DESIGN.md`](DESIGN.md).

To run JoyVL, see
[`recipes/JD/JoyAI-VL-Interaction.md`](../../../recipes/JD/JoyAI-VL-Interaction.md).

## Package boundaries

```text
core/        model-agnostic duplex contracts (adapter, session, turn runtime)
engine/      AsyncOmni/orchestrator scheduler data-plane adapter
openai/      WebSocket transport, Realtime projection, and audio codecs
minicpmo45/  MiniCPM input framing, policy, compatibility, and Stage0 state
joyvl/       JoyVL model-specific integration
personaplex/ PersonaPlex lockstep engine, model-owned runtime, and serving
```

MiniCPM does not run through the experimental `core.DuplexRuntime`
facade. Its active path uses the `openai` session controller, the experimental
engine contracts, the standard scheduler/model runners, and an injected
MiniCPM-specific runtime extension from `minicpmo45/runtime.py`.

`personaplex/` is a Moshi-class, pure-lockstep speech-to-speech model on the
`core/` contracts. It keeps `core/` untouched: the lockstep lifecycle (ONE
eternal, frame-clocked response that drains on close, instead of the turn-style
start/cancel-per-trigger one) is model policy and lives in the model package as
`PersonaPlexDuplexRuntime`, mirroring the model-owned runtime shape of the
MiniCPM-o duplex work. Its runnable serving path is `personaplex/serving/`
(single-session lease or `--batch-size` elastic slots) over
`personaplex/session.py` (lockstep driver).

## Adding a full-duplex model on the core contracts

The seam is `core.DuplexAdapter`. `core/` owns the session lifecycle,
epoch-based barge-in, playback cursor, and the event protocol; you implement
only model policy.

1. Create a sibling package `vllm_omni/experimental/fullduplex/<model>/`; keep
   model-specific code there and do not touch `core/`.
2. Implement one `DuplexAdapter` (`capabilities` / `on_input` / `respond`; the
   rest have defaults). Turn-based models run through `core.DuplexRuntime`
   unchanged; a model needing a different lifecycle carries its own runtime in
   its package (see `personaplex/adapter.py::PersonaPlexDuplexRuntime`).
3. Promote a helper from a model package up into `core/` only once a second
   model actually needs it.
