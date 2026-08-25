# MiniMax-H3 native LoRA runtime loading

Draft upstream PR text for vllm-project/vllm-omni.

## Motivation and scope

MiniMax-H3 FlashGen ships a native-layout distilled LoRA trained on Ascend NPU.
Unlike LightX2V Turbo, it keeps fused `qkv_proj`, includes `adaln_proj.linear`,
uses rank 64, and declares a non-uniform rectified-flow schedule. PR #6476 added
Turbo support on the legacy `DiffusionLoRAManager`; this PR extends the same
model-owned loader hook with a second contract keyed by
`key_format=minimax-h3-native`.

This complements PR #5991: checkpoint-pinned `base_schedule` covers merged
releases, while adapter metadata covers runtime LoRA activation.

## Changes

- Add a native H3 LoRA loader beside the existing Turbo loader in
  `vllm_omni/diffusion/models/minimax_h3/lora.py`.
- Reorder fused `qkv_proj` LoRA rows with the existing
  `_reorder_grouped_qkv_to_qkv` helper, then emit model-owned
  `PackedLoRALayerWeights` for Q/K/V and gate/up slices.
- Parse adapter-declared `base_schedule` from safetensors metadata through
  `DMD2SigmaSchedule.from_safetensors_metadata`.
- Extend `MiniMaxH3Pipeline` with native adapter classification, schedule
  precedence, T2VA-only task validation, and interval-count request validation.
- Leave `DiffusionLoRAManager` unchanged.

## Validation

- CPU tests in `tests/diffusion/models/minimax_h3/test_minimax_h3_native_lora.py`
  and `tests/diffusion/sched/test_dmd2_sigma_schedule.py`.
- Turbo regression suite remains unchanged.
- NPU validation script: `scripts/npu_validate_native_lora.sh`.

## Known limitations

- Dynamic execution only; no prefusion or DLO.
- Only one LoRA active at a time.
- Formal support is limited to the published FlashGen v1.0 native artifact on T2VA.
- Rejects model-level, layerwise, and distributed layerwise offload.
- Rejects activation on checkpoints that already pin `base_schedule`.

## Related work

- Issue #5700: MiniMax-H3 follow-up roadmap
- PR #6476: Turbo legacy LoRA loader (merged)
- PR #5991: checkpoint-pinned DMD2 schedules (merged)
- PR #6544, #6550, #6565: watch for generic LoRA manager changes
