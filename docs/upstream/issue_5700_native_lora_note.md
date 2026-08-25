# Issue #5700 design note (draft)

Post this comment on https://github.com/vllm-project/vllm-omni/issues/5700 before opening the PR.

---

## MiniMax-H3 FlashGen native LoRA (runtime loading)

Follow-up to merged #6476 (Turbo) and #5991 (checkpoint-pinned schedules).

**Scope**

- Add `key_format=minimax-h3-native` beside existing `minimax-h3-diffusers` Turbo loading.
- Keep `DiffusionLoRAManager` unchanged; loader emits model-owned `PackedLoRALayerWeights`.
- Reuse `_reorder_grouped_qkv_to_qkv` for fused `qkv_proj` LoRA `lora_B` rows, then pack Q/K/V slices.
- Adapter metadata declares `base_schedule`; pipeline precedence is adapter > checkpoint > uniform.
- v1.0 artifact is T2VA-only, rank 64, grouped qkv, no gate/up swap on fc1.

**Conflict watch**

- #6544: silent LoRA no-op — NPU validation uses Base/LoRA SHA256 toggling.
- #6550: Turbo pipeline refactor — native state kept incremental (`_native_lora_adapter_ids`, `_lora_sigma_schedules`).
- #6565: mixed-rank packed slices — native qkv splits preserve rank 64 per slice.

**Validation**

- CPU: `tests/diffusion/models/minimax_h3/test_minimax_h3_native_lora.py`
- NPU: `scripts/npu_validate_native_lora.sh`

PR body template: `docs/upstream/PR_h3_native_lora.md`
