# Gander Unit8/50 on one H200

This profile serves native full-duplex speech, video input, function-call
events and editable context through the shared MiniCPM-o 4.5 pipeline.
Use vLLM 0.31 and the vLLM-Omni checkout containing Gander support. The
deployment admits four sessions; that setting is not a measured capacity or
latency guarantee.

## Prepare the release

Install the repository's CUDA dependencies and the Token2Wav dependencies:

```bash
uv pip install 'step-audio2==1.0.0' 's3tokenizer==0.3.0' 'hyperpyyaml==1.2.3'
hf download Gander-Omni/Gander --revision 24fc4cc8543f95daf99be53b6199403a7732688f
python -m vllm_omni.model_executor.models.minicpmo_4_5.gander \
  /path/to/downloaded/snapshot /path/to/new/gander-model
```

Composition links the complete Thinker and Talker weights without changing
the snapshot. Keep the source directory available. The reference voice is
`/path/to/new/gander-model/assets/ref_audio.wav`.

## Serve and validate

Select an available physical GPU before device remapping. Run from the
repository root, replacing the model path and GPU UUID:

```bash
export CUDA_VISIBLE_DEVICES=GPU-YOUR-AVAILABLE-H200-UUID
export VLLM_USE_V2_MODEL_RUNNER=0
export GANDER_MODEL=/path/to/new/gander-model
vllm serve "$GANDER_MODEL" --omni --trust-remote-code \
  --deploy-config vllm_omni/deploy/gander.yaml --port 8091
```

Connect to `ws://localhost:8091/v1/realtime?duplex=1` using the
[shared Realtime client](../../docs/serving/realtime_api.md). Input is mono
PCM16 at 16 kHz; keep streaming microphone input, including silence, while
waiting for a response. Honor audio cancellation and playback acknowledgements.
Stage0 and Stage1 use eager execution and synchronous scheduling in this
profile. Code2Wav inherits the shared MiniCPM deployment settings.

Stop the manual server before running the suite below: pytest starts and stops
its own server. The advanced-model level selects all registered scenarios,
including the paired visual-answer and pending-tool interruption regressions.

```bash
python -m pytest tests/e2e/online_serving/test_gander.py -sv \
  -m 'advanced_model and cuda' --run-level advanced_model
```

## Scope and limits

See the [Gander guide](../../docs/serving/gander.md) for tools, context edits,
reconnect and rollover. Initial `tool_choice` accepts `auto` and `none`; it
cannot change within a session. The application executes tools and returns
results. Reconstruction re-encodes history without replaying tool calls or
old speech to the client.

The limits are 64 registered calls, 128 context receipts and a 256 MiB prompt
journal per session. Reopen a session when these limits are reached. Timeout
and cancellation recovery are tracked by
[RFC #8542](https://github.com/vllm-project/vllm-omni/issues/8542).
The functional tests do not establish speech-quality parity, visual-answer
accuracy or optimized throughput.
