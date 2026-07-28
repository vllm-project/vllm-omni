# Experimental Full-Duplex Runtime

This package contains two experimental integrations:

- the existing JoyVL framework and example integration;
- the MiniCPM-o 4.5 native audio path used by `/v1/duplex` and
  `/v1/realtime?duplex=1`.

For the MiniCPM active runtime path, lifecycle invariants, capability boundary,
Realtime response contract, and validation scope, see [`DESIGN.md`](DESIGN.md).

To run JoyVL, see
[`recipes/JD/JoyAI-VL-Interaction.md`](../../../recipes/JD/JoyAI-VL-Interaction.md).

## Package boundaries

```text
core/       existing JoyVL framework and experimental compatibility exports
engine/     AsyncOmni/orchestrator scheduler data-plane adapter
openai/     WebSocket transport, Realtime projection, and audio codecs
minicpmo45/ MiniCPM input framing, policy, compatibility, and Stage0 state
joyvl/      JoyVL model-specific integration
```

MiniCPM does not run through the removed experimental `core.DuplexRuntime`
facade. Its active path uses the `openai` session controller, the experimental
engine contracts, the standard scheduler/model runners, and an injected
MiniCPM-specific runtime extension from `minicpmo45/runtime.py`.
