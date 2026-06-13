# Ming-omni-tts 0.5B

> Offline and online dense Ming text-to-speech on ROCm

## Summary

- Vendor: inclusionAI
- Model: `inclusionAI/Ming-omni-tts-0.5B`
- Task: Text-to-speech with style, dialect, cloning, and multi-speaker controls
- Mode: Online serving via the OpenAI-compatible `/v1/audio/speech` API; offline inference
- Maintainer: Community

## References

- [Model card](https://huggingface.co/inclusionAI/Ming-omni-tts-0.5B)
- [Upstream repository](https://github.com/inclusionAI/Ming-omni-tts)
- [Offline example](../../examples/offline_inference/text_to_speech/ming_tts/)
- [Online example](../../examples/online_serving/text_to_speech/ming_tts/)

## Hardware Support

## ROCm

### 1x AMD `gfx942`

#### Environment

- OS: Ubuntu 22.04.5 LTS, x86_64
- Python: 3.12.13
- ROCm / HIP: 7.2.53211
- vLLM version: 0.22.0
- vLLM-Omni version or commit: 0.1.dev1873 / `99c35c410`
- Docker image: `vllm/vllm-omni-rocm:v0.22.0`

#### Command

From the vLLM-Omni repository root:

```bash
docker run --rm \
    --group-add=video \
    --ipc=host \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device /dev/kfd \
    --device /dev/dri \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v "$PWD":/app/vllm-omni \
    -w /app/vllm-omni \
    -e VLLM_ROCM_USE_AITER=0 \
    -p 8091:8091 \
    vllm/vllm-omni-rocm:v0.22.0 \
    --model inclusionAI/Ming-omni-tts-0.5B \
    --deploy-config vllm_omni/deploy/ming_tts.yaml \
    --omni \
    --port 8091 \
    --enforce-eager
```

#### Verification

```bash
python examples/online_serving/text_to_speech/ming_tts/openai_speech_client.py \
    --text "我觉得社会企业同个人都有责任" \
    --instruction-json '{"方言":"广粤话"}' \
    --ref-audio /path/to/yue_prompt.wav \
    --max-new-tokens 200 \
    --output dialect.wav
```

`--ref-audio` matches upstream `use_spk_emb=True`; do not add `--ref-text`
for the dialect case.

## Notes

- The official ROCm image includes the platform dependencies.
- See the [ROCm installation guide](../../docs/getting_started/installation/gpu.md)
  for interactive and source-build workflows.
- The [offline example](../../examples/offline_inference/text_to_speech/ming_tts/)
  can be run from an interactive container.
- The tested environment uses `--enforce-eager`.
- Output is mono 44.1 kHz audio.
