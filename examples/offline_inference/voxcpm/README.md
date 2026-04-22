# VoxCPM Offline Example

This directory contains the minimal offline VoxCPM example for vLLM Omni.

`end2end.py` is intentionally small and only covers:

- single text-to-speech
- single voice cloning with `ref_audio` + `ref_text`
- non-streaming with `vllm_omni/model_executor/stage_configs/voxcpm.yaml`
- streaming with `vllm_omni/model_executor/stage_configs/voxcpm_async_chunk.yaml`

Advanced workflows were moved out of the getting-started example:

- `benchmarks/voxcpm/vllm_omni/bench_tts_offline.py`: warmup, batch prompts, profiler, offline TTFP / RTF
- `benchmarks/voxcpm/vllm_omni/run_offline_matrix.py`: fixed offline smoke matrix
- `benchmarks/voxcpm/`: benchmark scripts and benchmark docs

## Prerequisites

Install VoxCPM in one of these ways:

```bash
pip install voxcpm
```

or point vLLM Omni to the local VoxCPM source tree:

```bash
export VLLM_OMNI_VOXCPM_CODE_PATH=/path/to/VoxCPM/src
```

The example writes WAV files with `soundfile`:

```bash
pip install soundfile
```

## Model Path

Pass the native VoxCPM model directory directly:

```bash
export VOXCPM_MODEL=/path/to/voxcpm-model
```

If the native VoxCPM `config.json` does not contain HuggingFace metadata such as
`model_type`, prepare a persistent HF-compatible config directory and point the
stage configs to it with `VLLM_OMNI_VOXCPM_HF_CONFIG_PATH`:

```bash
export VLLM_OMNI_VOXCPM_HF_CONFIG_PATH=/tmp/voxcpm_hf_config
mkdir -p "$VLLM_OMNI_VOXCPM_HF_CONFIG_PATH"
cp "$VOXCPM_MODEL/config.json" "$VLLM_OMNI_VOXCPM_HF_CONFIG_PATH/config.json"
cp "$VOXCPM_MODEL/generation_config.json" "$VLLM_OMNI_VOXCPM_HF_CONFIG_PATH/generation_config.json" 2>/dev/null || true
python3 -c 'import json, os; p=os.path.join(os.environ["VLLM_OMNI_VOXCPM_HF_CONFIG_PATH"], "config.json"); cfg=json.load(open(p, "r", encoding="utf-8")); cfg["model_type"]="voxcpm"; cfg.setdefault("architectures", ["VoxCPMForConditionalGeneration"]); json.dump(cfg, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)'
```

If the model directory itself already has `model_type`, this extra directory is
not required.

## Quick Start

Single text-to-speech, non-streaming:

```bash
python examples/offline_inference/voxcpm/end2end.py \
  --model "$VOXCPM_MODEL" \
  --text "This is a split-stage VoxCPM synthesis example running on vLLM Omni."
```

Single voice cloning, non-streaming:

```bash
python examples/offline_inference/voxcpm/end2end.py \
  --model "$VOXCPM_MODEL" \
  --text "This sentence is synthesized with a cloned voice." \
  --ref-audio /path/to/reference.wav \
  --ref-text "The exact transcript spoken in reference.wav."
```

Streaming:

```bash
python examples/offline_inference/voxcpm/end2end.py \
  --model "$VOXCPM_MODEL" \
  --stage-configs-path vllm_omni/model_executor/stage_configs/voxcpm_async_chunk.yaml \
  --text "This is a split-stage VoxCPM streaming example running on vLLM Omni."
```

By default, `end2end.py` writes to `output_audio/` for non-streaming and
`output_audio_streaming/` for streaming.

## Advanced Workflows

Use `benchmarks/voxcpm/vllm_omni/bench_tts_offline.py` when you need:

- warmup runs
- prompt files
- batch JSONL inputs
- profiler injection
- offline TTFP / RTF emission

Use `benchmarks/voxcpm/vllm_omni/run_offline_matrix.py` when you need the fixed offline smoke matrix that previously lived in `test.py`.

Full matrix benchmark example:

```bash
python benchmarks/voxcpm/vllm_omni/run_offline_matrix.py \
  --model "$VOXCPM_MODEL" \
  --ref-audio /path/to/reference.wav \
  --ref-text "The exact transcript spoken in reference.wav."
```

For online serving examples, see [examples/online_serving/voxcpm](../../online_serving/voxcpm/README.md).

For benchmark reporting, see [benchmarks/voxcpm](../../../benchmarks/voxcpm/README.md).

## Notes

- `voxcpm.yaml` is the default non-streaming stage config.
- `voxcpm_async_chunk.yaml` is the streaming stage config.
- Streaming is currently single-request oriented; the fixed smoke matrix now lives in `benchmarks/voxcpm/vllm_omni/run_offline_matrix.py`.
- `ref_text` must be the real transcript of the reference audio. Mismatched text usually causes obvious quality degradation.
