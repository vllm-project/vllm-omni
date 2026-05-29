# Kimi-Audio

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/kimi_audio>.

For a hardware-tagged step-by-step guide, see the
[Kimi-Audio recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/moonshotai/Kimi-Audio-7B-Instruct.md).

## Launch the server

Two-stage audio-out pipeline (default single-GPU sync):

```bash
vllm serve moonshotai/Kimi-Audio-7B-Instruct \
    --omni \
    --port 8091 \
    --stage-configs-path vllm_omni/deploy/kimi_audio.yaml
```

To enable multi-GPU async-chunk streaming for sub-second TTFB, edit the
YAML per the comments at its top. For `audio2text`-only deployments
where the MIMO audio branch is unused, override
`hf_overrides.kimia_generate_audio: false` on stage 0 to save ~4 GB.

## Curl examples

The example dir ships a unified curl script covering all three task
modes:

```bash
TASK=audio2text  bash examples/online_serving/kimi_audio/run_curl.sh
TASK=audio2audio bash examples/online_serving/kimi_audio/run_curl.sh
TASK=text2audio  bash examples/online_serving/kimi_audio/run_curl.sh
```
