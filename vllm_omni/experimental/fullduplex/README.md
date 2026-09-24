# Experimental Full-Duplex (JoyVL)

This package now contains the JoyVL framework and its example integration,
plus two client-side helpers kept for the benchmark and e2e drivers:

```text
core/              generic duplex scaffold used by the JoyVL adapter
joyvl/             JoyVL model-specific integration
client.py          legacy Realtime probe client (RealtimeDuplexClient) used by
                   the omniinteract / omni-duplex-eval benchmarks and the
                   server-VAD and Nemotron e2e drivers; applications should use
                   vllm_omni.clients.duplex.DuplexClient instead
video_stacking.py  camera-frame tiling for omni duplex video input
```

To run JoyVL, see
[`recipes/JD/JoyAI-VL-Interaction.md`](../../../recipes/JD/JoyAI-VL-Interaction.md).

The native full-duplex runtimes graduated out of this package. They now live
in the stable tree, where sessions are engine-resident (RFC
[vllm-omni#7181](https://github.com/vllm-project/vllm-omni/issues/7181)):

```text
vllm_omni/engine/duplex/                       sessions, manager, runner, leases, typed contract
vllm_omni/engine/duplex_omni_engine.py         DuplexOmniEngine (session message surface)
vllm_omni/engine/duplex_orchestrator.py        DuplexOrchestrator (hosts the session manager)
vllm_omni/entrypoints/duplex_omni.py           DuplexOmni + DuplexSessionHandle (Python API)
vllm_omni/entrypoints/duplex/                  WebSocket transport (attachments, resume, replay)
vllm_omni/clients/duplex.py                    DuplexClient / DuplexClientBase
vllm_omni/model_executor/models/minicpmo_4_5/duplex/  MiniCPM-o 4.5 DuplexModelPlugin
vllm_omni/model_executor/models/personaplex/duplex/   PersonaPlex (pre-framework, not ported yet)
vllm_omni/model_executor/models/nemotron_voicechat/duplex/  Nemotron VoiceChat (pre-framework, not ported yet)
vllm_omni/model_executor/duplex_sampling.py    AR-runner sampling hook helper
vllm_omni/outputs/duplex.py                    typed output decision envelope
```

For their architecture and validation scope, see
[`docs/design/fullduplex.md`](../../../docs/design/fullduplex.md) and
[`docs/design/fullduplex-personaplex.md`](../../../docs/design/fullduplex-personaplex.md).
PersonaPlex's single-process demo tier (browser client, standalone Moshi-web
server, `core/`-scaffold adapter) was demo-only and was removed rather than
graduated; the production path serves through the generic `/v1/duplex` stack.

## Adding a full-duplex model on the core contracts

The seam is `core.DuplexAdapter`. `core/` owns the session lifecycle,
epoch-based barge-in, playback cursor, and the event protocol; you implement
only model policy.

1. Create a sibling package `vllm_omni/experimental/fullduplex/<model>/`; keep
   model-specific code there and do not touch `core/`.
2. Implement one `DuplexAdapter` (`capabilities` / `on_input` / `respond`; the
   rest have defaults). Turn-based models run through `core.DuplexRuntime`
   unchanged.
3. Promote a helper from a model package up into `core/` only once a second
   model actually needs it.

For production serving, prefer the stable plugin seam instead: implement one
`vllm_omni.engine.duplex.plugin.DuplexModelPlugin` and name it in the model's
`pipeline.py` as `duplex_plugin`, as MiniCPM-o 4.5 does. PersonaPlex and
Nemotron VoiceChat still carry their pre-framework duplex code and are ported
to the plugin contract in follow-up PRs. The contract is documented in
[`docs/design/fullduplex.md`](../../../docs/design/fullduplex.md).
