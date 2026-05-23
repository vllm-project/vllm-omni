# SenseNova-U1 Online Dynamic Batching — Test Results

**Date:** 2026-05-23
**Branch:** `feature/sensenova-u1-online-dynamic-batching`
**Environment:** Python 3.11.15, PyTorch 2.11.0+cu126, flash_attn 2.8.4 (sm90), pytest 9.0.3, pytest-asyncio 1.3.0

---

## Summary

| # | Test Script | Type | Total | Passed | Failed | Notes |
|---|------------|------|-------|--------|--------|-------|
| 1 | `test_sensenova_u1_pipeline_batching.py` | CPU | 23 | 23 | 0 | |
| 2 | `test_sensenova_u1_pipeline_cfg.py` | CPU | 5 | 5 | 0 | |
| 3 | `test_sensenova_u1_step_execution_unit.py` | CPU | 12 | 12 | 0 | |
| 4 | `test_sensenova_u1_transformer_varlen.py` | CPU | 10 | 10 | 0 | |
| 5 | `test_diffusion_scheduler.py` | CPU | 55 | 55 | 0 | |
| 6 | `test_sensenova_u1_step_execution.py` | GPU | 17 | 17 | 0 | |

**Total: 122 tests, 122 passed, 0 failed**

---

## Script 1: test_sensenova_u1_pipeline_batching.py

**23 passed** in 33.38s

```
TestDenoiseStepDispatch:
  ✅ test_single_request_calls_fast_path
  ✅ test_multiple_requests_calls_batched
  ✅ test_missing_state_raises_valueerror

TestPrepareSingleEmbeds:
  ✅ test_adds_timestep_embedding
  ✅ test_adds_noise_scale_embedding_when_configured
  ✅ test_returns_correct_indexes

TestBatchedPredictV:
  ✅ test_packs_embeds_correctly
  ✅ test_builds_cu_seqlens_correctly
  ✅ test_returns_per_request_results

TestBatchedDenoiseStep:
  ✅ test_cond_only_when_no_cfg
  ✅ test_uncond_only_for_cfg_requests
  ✅ test_cfg_combination_standard
  ✅ test_cfg_interval_respected
  ✅ test_concatenates_v_preds_in_order
  ✅ test_raises_on_missing_state
  ✅ test_mixed_cfg_produces_different_results
  ✅ test_it2i_dispatches_all_three_forwards
  ✅ test_it2i_dual_cfg_combination_formula
  ✅ test_it2i_img_cfg_scale_one_skips_uncond
  ✅ test_it2i_no_img_cond_cache_falls_to_t2i
  ✅ test_mixed_t2i_and_it2i_in_batch
  ✅ test_cfg_zero_star_step_nonzero
  ✅ test_cfg_zero_star_step_zero_returns_zeros
```

---

## Script 2: test_sensenova_u1_pipeline_cfg.py

**5 passed** in 4.05s

```
TestPrepareCFGBypass:
  ✅ test_prepare_encode_sets_do_true_cfg_false
  ✅ test_prepare_encode_stores_cfg_scale_in_extra
  ✅ test_cfg_scale_one_still_sets_do_true_cfg_false
  ✅ test_prepare_encode_it2i_sets_do_true_cfg_false

TestCFGHandledInternally:
  ✅ test_batched_denoise_uses_extra_cfg_scale_not_do_true_cfg
```

---

## Script 3: test_sensenova_u1_step_execution_unit.py

**12 passed** in 23.12s

```
Structural:
  ✅ test_class_declares_step_execution
  ✅ test_class_has_step_methods
  ✅ test_class_has_step_states_dict
  ✅ test_protocol_isinstance_check
  ✅ test_helper_methods_exist

_parse_request_from_state:
  ✅ test_parse_request_default_values
  ✅ test_parse_request_grid_factor_rounding

step_scheduler:
  ✅ test_step_scheduler_advances_step_index
  ✅ test_step_scheduler_updates_image_prediction

post_decode:
  ✅ test_post_decode_cleans_up_step_states
  ✅ test_post_decode_returns_diffusion_output
  ✅ test_post_decode_passes_think_text
```

---

## Script 4: test_sensenova_u1_transformer_varlen.py

**10 passed** in 3.88s

```
TestAttentionForwardGenVarlen:
  ✅ test_output_shape_single_request
  ✅ test_packs_kv_correctly_multi_request
  ✅ test_flash_attn_called_with_causal_false

TestDecoderLayerForwardGenVarlen:
  ✅ test_residual_connections
  ✅ test_calls_attn_with_correct_args

TestModelForwardVarlen:
  ✅ test_iterates_all_layers
  ✅ test_extracts_prefix_from_cache
  ✅ test_applies_final_norm
  ✅ test_multi_request_prefix_extraction

TestForCausalLMForwardVarlen:
  ✅ test_delegates_to_model
```

---

## Script 5: test_diffusion_scheduler.py

**55 passed** in 22.21s

```
TestGetSamplingParamsKey:           4/4 ✅
TestRequestScheduler:               9/9 ✅
TestDiffusionEngine:
  ✅ test_add_req_and_wait_for_response_single_path
  ✅ test_supports_scheduler_interface_injection
  ✅ test_initializes_injected_scheduler
  ✅ test_scheduler_alias_keeps_default_request_scheduler
  ✅ test_step_raises_aborted_error
  ✅ test_abort_queue_marks_request_finished_aborted
  ✅ test_finalize_finished_request_returns_aborted_output
  ✅ test_initializes_step_scheduler_when_step_execution_enabled
  ✅ test_dummy_run_raises_on_output_error
TestStepScheduler:                  17/17 ✅
TestKeysMatchIgnoring:              6/6 ✅
TestHeterogeneousBatchScheduling:   5/5 ✅
```

---

## Script 6: test_sensenova_u1_step_execution.py (GPU)

**17 passed** in ~40min (includes model loading + think mode AR decoding)

```
  ✅ [1/17]  Protocol conformance
  ✅ [2/17]  Step mode equivalence (MSE=0.0000, PSNR=inf dB)
  ✅ [3/17]  Homogeneous batch (MSE=0.0000)
  ✅ [4/17]  Heterogeneous prompt lengths (MSE=0.0000)
  ✅ [5/17]  Heterogeneous resolutions (MSE=0.0000)
  ✅ [6/17]  Heterogeneous CFG scales (MSE=0.0000)
  ✅ [7/17]  Dynamic step counts (MSE=0.0000)
  ✅ [8/17]  Throughput comparison (varlen speedup: 1.45x)
  ✅ [9/17]  Varlen batched correctness (MSE=0.0000)
  ✅ [10/17] Varlen heterogeneous resolution (MSE=0.0000)
  ✅ [11/17] Varlen mixed CFG (MSE=0.0000)
  ✅ [12/17] Varlen throughput (speedup: 1.45x)
  ✅ [13/17] Dynamic join/leave (MSE=0.0000)
  ✅ [14/17] Varlen throughput stress (16 reqs, think=True, speedup: 1.43x)
  ✅ [15/17] E2E scheduler heterogeneous batch (8 reqs, speedup: 1.19x)
  ✅ [16/17] Think mode step equivalence (MSE=0.0000, PSNR=inf dB)
  ✅ [17/17] IT2I step equivalence (MSE=0.0000, PSNR=inf dB)
```

---

## Coverage Summary

### Modified source files → test mapping

| Source File | CPU Unit Tests | GPU Integration |
|---|---|---|
| `pipeline_sensenova_u1.py` | batching(23) + cfg(5) + unit(12) = **40** | step_execution(17) |
| `sensenova_u1_transformer.py` | varlen(10) | step_execution(17) |
| `data.py` | scheduler(55) | — |
| `base_scheduler.py` | scheduler(55) | — |

### Key features verified

- ✅ `denoise_step` dispatch (single vs batch)
- ✅ `_prepare_single_embeds` (timestep + noise_scale embedding)
- ✅ `_batched_predict_v` (packing, cu_seqlens, unpacking)
- ✅ `_batched_denoise_step` T2I CFG (standard + cfg_zero_star)
- ✅ `_batched_denoise_step` IT2I dual CFG (all 3 branches)
- ✅ Mixed T2I + IT2I in same batch
- ✅ `prepare_encode` CFG bypass (`do_true_cfg=False`)
- ✅ `_parse_request_from_state` defaults + grid rounding
- ✅ `step_scheduler` step advancement + latent update
- ✅ `post_decode` cleanup + think_text passthrough
- ✅ `StepScheduler` lifecycle + heterogeneous batching
- ✅ `forward_varlen` layer iteration + norm + KV extraction
- ✅ `flash_attn_varlen_func` integration (attention forward)
- ✅ Step mode vs `forward()` bit-exact equivalence (GPU, MSE=0)
- ✅ Think mode step equivalence (GPU, MSE=0)
- ✅ IT2I step equivalence (GPU, MSE=0)
- ✅ Dynamic join/leave correctness (GPU, MSE=0)
- ✅ Varlen batching throughput improvement (1.43-1.45x speedup)
- ✅ E2E scheduler → pipeline heterogeneous batch (8 reqs, MSE=0)
