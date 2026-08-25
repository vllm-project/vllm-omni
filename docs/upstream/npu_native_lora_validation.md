# NPU end-to-end validation: MiniMax-H3 FlashGen native LoRA

Run on Ascend NPU with vLLM-Omni built from this branch, the official
`MiniMaxAI/MiniMax-H3` base checkpoint, and the repacked FlashGen artifact:

```text
Beidouqixing/minimax-h3-4step-lora-flashgen/minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors
```

## Preconditions

1. Base weights downloaded and approved on Hugging Face.
2. FlashGen LoRA downloaded to a local path (`FLASHGEN_LORA`).
3. Server started **without** model-level, layerwise, or DLO offload.
4. `--task-type fl2va --lora-backend peft --lora-path "${FLASHGEN_LORA}"`.

## Binding check

On adapter activation, logs should report **259/259** native targets bound,
including `adaln_proj.linear` and `final_layer.adaln_proj.linear`.

## Decode stream SHA256 matrix

Use `scripts/npu_validate_native_lora.sh` or equivalent curl requests:

| Run | LoRA scale | Expected |
| --- | ---: | --- |
| base | 0.0 | SHA256 A |
| lora | 1.0 | SHA256 B (B ≠ A) |
| base2 | 0.0 | SHA256 A (same as base) |
| lora2 | 1.0 | SHA256 B (same as lora) |

Pass criteria:

- Each state is deterministic across repeats.
- Base and LoRA hashes differ (guards against #6544-style silent no-op).
- `num_inference_steps=4` succeeds; `num_inference_steps=5` returns client error.

## Request contract

```bash
-F 'num_inference_steps=4' \
-F 'extra_params={"task":"t2va","duration":5.2}' \
-F "lora={\"name\":\"h3-flashgen\",\"path\":\"${FLASHGEN_LORA}\",\"scale\":1.0}"
```

Rejections to spot-check:

- `task=fl2va` or `ref2va` with active native LoRA
- `--enable-cpu-offload`, layerwise offload, or DLO
- merged checkpoint with pinned `base_schedule` in `model_index.json`
