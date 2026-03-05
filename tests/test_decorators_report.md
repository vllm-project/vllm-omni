# tests 用例与修饰器统计（已排除 e2e、examples、perf）

| 文件 | 函数名 | 修饰器 |
| --- | --- | --- |
| benchmarks/patch/test_patch.py | test_mix_request_func_output_has_text_latency | @pytest.mark.core_model \| @pytest.mark.benchmark \| @pytest.mark.cpu |
|  | test_output_tokens_assigned_multiple_metrics | @pytest.mark.core_model \| @pytest.mark.benchmark \| @pytest.mark.cpu |
|  | test_output_tokens_assigned_with_metrics | @pytest.mark.core_model \| @pytest.mark.benchmark \| @pytest.mark.cpu |
|  | test_output_tokens_initialization | @pytest.mark.core_model \| @pytest.mark.benchmark \| @pytest.mark.cpu |
|  | test_output_tokens_not_assigned_without_metrics | @pytest.mark.core_model \| @pytest.mark.benchmark \| @pytest.mark.cpu |
|  | test_output_tokens_with_audio_and_text | @pytest.mark.core_model \| @pytest.mark.benchmark \| @pytest.mark.cpu |
|  | test_output_tokens_with_missing_num_tokens_out | @pytest.mark.core_model \| @pytest.mark.benchmark \| @pytest.mark.cpu |
|  | test_text_latency_assigned_with_text_response | @pytest.mark.core_model \| @pytest.mark.benchmark \| @pytest.mark.cpu |
|  | test_text_latency_initial_value | @pytest.mark.core_model \| @pytest.mark.benchmark \| @pytest.mark.cpu |
|  | test_text_latency_mixed_modalities | @pytest.mark.core_model \| @pytest.mark.benchmark \| @pytest.mark.cpu |
|  | test_text_latency_not_affected_by_metrics | @pytest.mark.core_model \| @pytest.mark.benchmark \| @pytest.mark.cpu |
|  | test_text_latency_updated_with_multiple_text_chunks | @pytest.mark.core_model \| @pytest.mark.benchmark \| @pytest.mark.cpu |
|  | test_text_latency_value_consistency | @pytest.mark.core_model \| @pytest.mark.benchmark \| @pytest.mark.cpu |
|  | test_text_latency_with_only_audio_response | @pytest.mark.core_model \| @pytest.mark.benchmark \| @pytest.mark.cpu |
| benchmarks/test_serve_cli.py | test_bench_serve_chat | @pytest.mark.core_model \| @pytest.mark.benchmark \| @hardware_test(res={"cuda": "L4"}, num_cards=3) |
| comfyui/test_comfyui_integration.py | test_image_generation_node | (无) |
|  | test_tts_nodes | (无) |
|  | test_understanding_node | (无) |
| diffusion/attention/test_attention_sp.py | test_sequence_parallel | (无) |
| diffusion/attention/test_flash_attn.py | test_fa_vs_sdpa | @pytest.mark.skipif(not is_gpu, reason="FlashAttention requires CUDA or XPU") |
|  | test_padding_equivalence | @pytest.mark.skipif(not is_gpu, reason="FlashAttention requires CUDA or XPU") |
| diffusion/cache/test_cache_backends.py | test_enable | @pytest.mark.core_model \| @pytest.mark.cpu \| @patch("vllm_omni.diffusion.cache.teacache.backend.apply_teacache_hook") |
|  | test_enable_single_transformer | @pytest.mark.core_model \| @pytest.mark.cpu \| @patch("vllm_omni.diffusion.cache.cache_dit_backend.cache_dit") |
|  | test_enable_with_coefficients | @pytest.mark.core_model \| @pytest.mark.cpu \| @patch("vllm_omni.diffusion.cache.teacache.backend.apply_teacache_hook") |
|  | test_get_cache_backend_cache_dit | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_get_cache_backend_invalid | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_get_cache_backend_none | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_get_cache_backend_tea_cache | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_init | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_init_with_config_object | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_init_with_dict | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_refresh | @pytest.mark.core_model \| @pytest.mark.cpu \| @patch("vllm_omni.diffusion.cache.cache_dit_backend.cache_dit") |
|  | test_refresh | @pytest.mark.core_model \| @pytest.mark.cpu \| @patch("vllm_omni.diffusion.cache.teacache.backend.apply_teacache_hook") |
| diffusion/distributed/test_cfg_parallel.py | test_predict_noise_maybe_with_cfg | (无) |
|  | test_predict_noise_without_cfg | (无) |
| diffusion/distributed/test_comm.py | test_4d_identity | (无) |
|  | test_5d_identity | (无) |
|  | test_ring_p2p | (无) |
| diffusion/distributed/test_distributed_vae_executor.py | test_balance_tasks | (无) |
|  | test_compute_global_padding_shape | (无) |
|  | test_is_distributed_enabled | (无) |
|  | test_pack_and_unpack | (无) |
| diffusion/distributed/test_hsdp.py | test_condition_matches_blocks | @pytest.mark.diffusion \| @pytest.mark.parallel \| @pytest.mark.cpu |
|  | test_custom_values | @pytest.mark.diffusion \| @pytest.mark.parallel \| @pytest.mark.cpu |
|  | test_default_values | @pytest.mark.diffusion \| @pytest.mark.parallel \| @pytest.mark.cpu |
|  | test_from_dict_with_hsdp | @pytest.mark.diffusion \| @pytest.mark.parallel \| @pytest.mark.cpu |
|  | test_hsdp_auto_shard_size | @pytest.mark.diffusion \| @pytest.mark.parallel \| @pytest.mark.cpu |
|  | test_hsdp_auto_shard_size_fails_standalone | @pytest.mark.diffusion \| @pytest.mark.parallel \| @pytest.mark.cpu |
|  | test_hsdp_cannot_use_with_tp | @pytest.mark.diffusion \| @pytest.mark.parallel \| @pytest.mark.cpu |
|  | test_hsdp_combined_world_size | @pytest.mark.diffusion \| @pytest.mark.parallel \| @pytest.mark.cpu |
|  | test_hsdp_disabled_by_default | @pytest.mark.diffusion \| @pytest.mark.parallel \| @pytest.mark.cpu |
|  | test_hsdp_explicit_shard_size_invalid | @pytest.mark.diffusion \| @pytest.mark.parallel \| @pytest.mark.cpu |
|  | test_hsdp_explicit_shard_size_valid | @pytest.mark.diffusion \| @pytest.mark.parallel \| @pytest.mark.cpu |
|  | test_hsdp_replicate_size_exceeds_world_size | @pytest.mark.diffusion \| @pytest.mark.parallel \| @pytest.mark.cpu |
|  | test_hsdp_standalone_mode | @pytest.mark.diffusion \| @pytest.mark.parallel \| @pytest.mark.cpu |
|  | test_hsdp_standalone_with_replicate | @pytest.mark.diffusion \| @pytest.mark.parallel \| @pytest.mark.cpu |
|  | test_hsdp_standalone_world_size | @pytest.mark.diffusion \| @pytest.mark.parallel \| @pytest.mark.cpu |
|  | test_hsdp_with_replicate | @pytest.mark.diffusion \| @pytest.mark.parallel \| @pytest.mark.cpu |
|  | test_model_with_shard_conditions | @pytest.mark.diffusion \| @pytest.mark.parallel \| @pytest.mark.cpu |
| diffusion/distributed/test_parallel_state_sp_groups.py | test_set_seq_parallel_pg_uses_explicit_sp_groups | @pytest.mark.diffusion \| @pytest.mark.parallel \| @pytest.mark.cpu |
|  | test_set_seq_parallel_pg_validates_sp_group_ranks | @pytest.mark.diffusion \| @pytest.mark.parallel \| @pytest.mark.cpu |
| diffusion/distributed/test_sp_plan_hooks.py | test_apply_sp_registers_hooks | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_apply_sp_with_wildcard | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_config_defaults_invalid | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_config_hybrid | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_config_ring_only | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_config_ulysses_only | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_gather_tensor_simulation | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_get_parameter_from_args | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_get_parameter_from_kwargs | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_hook_init | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_hook_init_multiple_outputs | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_hook_init_single_output | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_hook_initialize | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_input_hook_name | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_invalid_module_key_type | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_invalid_output_index_without_split_output | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_invalid_plan_type | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_invalid_submodule_raises | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_model_with_sp_plan | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_model_without_sp_plan | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_module_dict | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_module_list_by_index | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_multiple_wildcards_raises | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_nested_submodule | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_output_hook_name | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_padding_simulation | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_parameter_caching | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_partial_shard_simulation | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_plan_validation_before_apply | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_qwen_image_transformer_sp_plan | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_remove_sp_removes_hooks | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_resolve_int_source | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_resolve_string_source_from_tensor | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_resolve_text_len_caching | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_root_module | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_sequence_parallel_input_repr | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_sequence_parallel_output_repr | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_sequence_parallel_partial_input_repr | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_sequence_parallel_partial_input_with_int_source | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_shard_tensor_simulation | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_simple_submodule | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_skip_shard_on_wrong_dims | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_unwrap_nested_wrapper | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_unwrap_sequential_single | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_unwrap_simple_module | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_valid_partial_input_plan | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_valid_plan_structure_for_model | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_valid_simple_plan | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_wildcard_modulelist | @pytest.mark.diffusion \| @pytest.mark.parallel |
|  | test_zimage_transformer_sp_plan | @pytest.mark.diffusion \| @pytest.mark.parallel |
| diffusion/distributed/test_vae_patch_parallel.py | test_distributed_tiled_decode_stitches_tiles | (无) |
|  | test_factor_pp_grid | (无) |
|  | test_get_vae_out_channels_defaults_to_3 | (无) |
|  | test_get_vae_out_channels_reads_config | (无) |
|  | test_get_vae_spatial_scale_factor_defaults_to_8_on_exception | (无) |
|  | test_get_vae_spatial_scale_factor_defaults_to_8_on_missing_or_empty | (无) |
|  | test_get_vae_spatial_scale_factor_uses_block_out_channels_len_minus_1 | (无) |
|  | test_get_vae_tile_params_parses_types | (无) |
|  | test_get_vae_tile_params_returns_none_if_missing | (无) |
|  | test_get_vae_tiling_params_parses_types | (无) |
|  | test_get_vae_tiling_params_returns_none_if_missing | (无) |
|  | test_get_world_rank_pp_size | (无) |
| diffusion/lora/test_base_linear.py | test_diffusion_base_linear_apply_multi_slice | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_diffusion_base_linear_apply_respects_inactive_slices | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_diffusion_base_linear_reset_lora_disables_fast_path | @pytest.mark.core_model \| @pytest.mark.cpu |
| diffusion/lora/test_lora_manager.py | test_lora_manager_activates_fused_lora_on_packed_layer | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_lora_manager_activates_packed_lora_from_sublayers | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_lora_manager_applies_multiple_scales_correctly | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_lora_manager_does_not_evict_pinned_adapter | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_lora_manager_evicts_lru_adapter_when_cache_full | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_lora_manager_replace_layers_does_not_rewrap_base_layer | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_lora_manager_replaces_packed_layer_when_targeting_sublayers | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_lora_manager_scales_correctly_with_rank_changes | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_lora_manager_supported_modules_are_stable_with_wrapped_layers | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_lora_manager_warns_when_all_adapters_pinned | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_scale_keys_are_rounded | @pytest.mark.core_model \| @pytest.mark.cpu |
| diffusion/models/nextstep_1_1/test_nextstep_cfg_parallel_layout.py | test_build_captions_enables_three_way_cfg_when_image_conditions_exist | (无) |
|  | test_build_captions_ignores_image_cfg_without_image_conditions | (无) |
|  | test_decoding_cfg_parallel_mismatch_falls_back_to_non_parallel | (无) |
|  | test_decoding_non_parallel_uses_cfg_mult_for_sampling_and_duplication | (无) |
|  | test_decoding_rejects_incompatible_batch_and_cfg_mult | (无) |
|  | test_flow_matching_head_sample_validates_cfg_mult_divisibility | (无) |
|  | test_resolve_cfg_layout | (无) |
| diffusion/models/z_image/test_zimage_tp_constraints.py | test_validate_zimage_tp_constraints_tp2_ok | (无) |
|  | test_validate_zimage_tp_constraints_tp3_fails_on_ffn_hidden_dim | (无) |
|  | test_validate_zimage_tp_constraints_tp4_fails_on_heads | (无) |
| diffusion/quantization/test_fp8_config.py | test_fp8_config_creation | (无) |
|  | test_fp8_config_with_custom_params | (无) |
|  | test_fp8_delegates_to_vllm_config | (无) |
|  | test_invalid_quantization | (无) |
|  | test_none_quantization | (无) |
|  | test_quantization_conflicting_methods_warning | (无) |
|  | test_quantization_dict_not_mutated | (无) |
|  | test_quantization_integration | (无) |
|  | test_supported_methods | (无) |
|  | test_vllm_config_extraction | (无) |
| diffusion/test_diffusers_loader.py | test_empty_source_prefix_keeps_full_model_strict_check | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
|  | test_strict_check_only_validates_source_prefix_parameters | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
|  | test_strict_check_raises_when_source_parameters_are_missing | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
| diffusion/test_diffusion_model_runner.py | test_execute_model_emits_cache_summary_with_active_cache_dit_backend | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
|  | test_execute_model_skips_cache_summary_without_active_cache_backend | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
|  | test_load_model_clears_cache_backend_for_unsupported_pipeline | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
| diffusion/test_diffusion_worker.py | test_load_weights_calls_pipeline | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
|  | test_load_weights_empty_iterable | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
|  | test_sleep_falls_back_to_device_memory_when_nvml_unavailable | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
|  | test_sleep_level_1 | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
|  | test_sleep_level_2 | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
|  | test_sleep_memory_freed_validation | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
|  | test_wake_up_partial_buffer_restore | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
|  | test_wake_up_with_buffers | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
|  | test_wake_up_without_buffers | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
| diffusion/test_multiproc_executor_concurrency.py | test_collective_rpc_closed_executor_raises | @pytest.mark.diffusion |
|  | test_results_are_correctly_routed | @pytest.mark.diffusion |
|  | test_results_are_correctly_routed | @pytest.mark.diffusion |
|  | test_results_are_correctly_routed | @pytest.mark.diffusion |
|  | test_rpc_times_out_when_add_req_stalled_on_worker | @pytest.mark.diffusion |
|  | test_rpc_times_out_when_lock_held_directly | @pytest.mark.diffusion |
|  | test_rpc_without_timeout_still_waits_for_lock | @pytest.mark.diffusion |
|  | test_serial_add_req_error_propagation | @pytest.mark.diffusion |
|  | test_serial_add_req_multiple_sequential | @pytest.mark.diffusion |
|  | test_serial_add_req_returns_correct_result | @pytest.mark.diffusion |
|  | test_serial_add_req_then_collective_rpc | @pytest.mark.diffusion |
|  | test_serial_collective_rpc_all_ranks | @pytest.mark.diffusion |
|  | test_serial_collective_rpc_error_propagation | @pytest.mark.diffusion |
|  | test_serial_collective_rpc_single_rank | @pytest.mark.diffusion |
| diffusion/test_worker_wrapper_base.py | test_basic_initialization | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
|  | test_custom_pipeline_args_initialization | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
|  | test_custom_pipeline_with_explicit_extension | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
|  | test_execute_method_error | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
|  | test_execute_method_invalid_type | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
|  | test_execute_method_success | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
|  | test_execute_method_with_no_args | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
|  | test_execute_model_delegation | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
|  | test_extension_conflict_warning | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
|  | test_generate_delegation | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
|  | test_getattr_delegation | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
|  | test_getattr_method_access | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
|  | test_getattr_missing_attribute | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
|  | test_load_weights_delegation | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
|  | test_multiple_extensions_same_class | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
|  | test_prepare_worker_class_with_extension_class | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
|  | test_prepare_worker_class_with_extension_string | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
|  | test_prepare_worker_class_without_extension | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
|  | test_re_init_pipeline_basic | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
|  | test_re_init_pipeline_cleanup | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
|  | test_re_init_pipeline_multiple_calls | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
|  | test_re_init_pipeline_none_pipeline | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
|  | test_shutdown_delegation | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
|  | test_sleep_delegation | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
|  | test_wake_up_delegation | @pytest.mark.core_model \| @pytest.mark.diffusion \| @pytest.mark.cpu |
| distributed/omni_connectors/test_adapter_and_flow.py | test_get_connectors_for_stage | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_recv_no_connector | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_recv_success | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_recv_with_missing_metadata | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_send_fail | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_send_success | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_shm_connector_flow | @pytest.mark.core_model \| @pytest.mark.cpu |
| distributed/omni_connectors/test_basic_connectors.py | test_basic_serialization | (无) |
|  | test_create_shm_connector | (无) |
|  | test_create_unknown_connector | (无) |
|  | test_get_invalid_metadata | (无) |
|  | test_ndarray_serialization | (无) |
|  | test_put_get_inline | (无) |
|  | test_put_get_shm | (无) |
|  | test_tensor_serialization | (无) |
| distributed/omni_connectors/test_chunk_transfer_adapter.py | test_cleanup_after_poll_flow | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_cleanup_clears_all_state | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_cleanup_idempotent | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_cleanup_infers_external_id | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_cleanup_only_affects_target_request | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_cleanup_preserves_pending_save | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_cleanup_request_id_reuse_not_polluted | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_create_connector_config_parsing | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_generation_scheduler_calls_cleanup_on_finished | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_load_poll | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_postprocess_scheduler_output | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_process_and_restore_queues | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_save_async | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_update_request_payload | @pytest.mark.core_model \| @pytest.mark.cpu |
| distributed/omni_connectors/test_kv_flow.py | test_integration_flow | @pytest.mark.core_model \| @pytest.mark.cpu \| @pytest.mark.cache |
|  | test_manager_extraction | @pytest.mark.core_model \| @pytest.mark.cpu \| @pytest.mark.cache |
|  | test_manager_extraction_mismatched_kv_block_counts | @pytest.mark.core_model \| @pytest.mark.cpu \| @pytest.mark.cache |
|  | test_manager_extraction_no_connector | @pytest.mark.core_model \| @pytest.mark.cpu \| @pytest.mark.cache |
|  | test_manager_extraction_tuple_layout | @pytest.mark.core_model \| @pytest.mark.cpu \| @pytest.mark.cache |
|  | test_manager_reception | @pytest.mark.core_model \| @pytest.mark.cpu \| @pytest.mark.cache |
|  | test_normalize_layer_kv_rejects_invalid_inputs | @pytest.mark.core_model \| @pytest.mark.cpu \| @pytest.mark.cache |
| distributed/omni_connectors/test_mooncake_transfer_engine_buffer.py | test_alignment | @pytest.mark.cpu \| @pytest.mark.parallel |
|  | test_basic_alloc_free | @pytest.mark.cpu \| @pytest.mark.parallel |
|  | test_context_manager_releases_buffer | @pytest.mark.cpu \| @pytest.mark.parallel |
|  | test_double_free_after_merge_is_safe | @pytest.mark.cpu \| @pytest.mark.parallel \| @pytest.mark.slow |
|  | test_double_free_exact_is_safe | @pytest.mark.cpu \| @pytest.mark.parallel \| @pytest.mark.slow |
|  | test_exhaustion_and_recovery | @pytest.mark.cpu \| @pytest.mark.parallel |
|  | test_fragmentation_and_defrag | @pytest.mark.cpu \| @pytest.mark.parallel \| @pytest.mark.slow |
|  | test_merge_adjacent_blocks | @pytest.mark.cpu \| @pytest.mark.parallel \| @pytest.mark.slow |
|  | test_partial_overlap_raises_corruption | @pytest.mark.cpu \| @pytest.mark.parallel \| @pytest.mark.slow |
|  | test_tensor_view | @pytest.mark.cpu \| @pytest.mark.parallel |
|  | test_thread_safety | @pytest.mark.cpu \| @pytest.mark.parallel |
| distributed/omni_connectors/test_mooncake_transfer_engine_rdma.py | test_auto_cleanup | @pytest.mark.parallel \| @pytest.mark.gpu |
|  | test_bytes_e2e | @pytest.mark.parallel \| @pytest.mark.gpu |
|  | test_cleanup_releases_buffer | @pytest.mark.parallel \| @pytest.mark.gpu |
|  | test_close_releases_resources | @pytest.mark.parallel \| @pytest.mark.gpu |
|  | test_concurrent_put | @pytest.mark.parallel \| @pytest.mark.gpu |
|  | test_concurrent_put_get_integrity | @pytest.mark.parallel \| @pytest.mark.gpu |
|  | test_concurrent_put_get_threaded_both_sides | @pytest.mark.parallel \| @pytest.mark.gpu |
|  | test_context_manager | @pytest.mark.parallel \| @pytest.mark.gpu |
|  | test_double_close_safe | @pytest.mark.parallel \| @pytest.mark.gpu |
|  | test_empty_bytes_rejected | @pytest.mark.parallel \| @pytest.mark.gpu |
|  | test_gpu_e2e_transfer | @pytest.mark.parallel \| @pytest.mark.gpu |
|  | test_gpu_pool_init | @pytest.mark.parallel \| @pytest.mark.gpu |
|  | test_gpu_pool_put_cpu_and_gpu_tensor | @pytest.mark.parallel \| @pytest.mark.gpu |
|  | test_initialization | @pytest.mark.parallel \| @pytest.mark.gpu |
|  | test_large_tensor_100mb | @pytest.mark.parallel \| @pytest.mark.gpu |
|  | test_large_tensor_500mb | @pytest.mark.parallel \| @pytest.mark.gpu |
|  | test_mixed_types_sequential | @pytest.mark.parallel \| @pytest.mark.gpu |
|  | test_object_e2e | @pytest.mark.parallel \| @pytest.mark.gpu |
|  | test_pool_exhaustion_and_recovery | @pytest.mark.parallel \| @pytest.mark.gpu |
|  | test_put_tensor_bytes_object | @pytest.mark.parallel \| @pytest.mark.gpu |
|  | test_rapid_alloc_free_cycle | @pytest.mark.parallel \| @pytest.mark.gpu |
|  | test_small_tensor_1_element | @pytest.mark.parallel \| @pytest.mark.gpu |
|  | test_tensor_e2e | @pytest.mark.parallel \| @pytest.mark.gpu |
|  | test_zero_copy_e2e | @pytest.mark.parallel \| @pytest.mark.gpu |
| distributed/omni_connectors/test_omni_connector_configs.py | test_load_qwen_yaml_configs | @pytest.mark.skipif(len(config_files) == 0, reason="No config files found or directory missing") |
| engine/test_async_omni_engine_abort.py | test_abort | @pytest.mark.core_model \| @pytest.mark.omni \| @hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1) |
| entrypoints/openai_api/test_image_server.py | test_client | @pytest.mark.core_model \| @pytest.mark.cpu \| @pytest.fixture |
|  | test_different_image_sizes | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_encode_image_base64 | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_generate_images_async_omni_sampling_params | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_generate_multiple_images | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_generate_single_image | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_health_endpoint | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_health_endpoint_no_engine | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_image_edit_compression_jpeg | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_image_edit_compression_png | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_image_edit_images_processing | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_image_edit_parameter_default | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_image_edit_parameter_default_single_stage | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_image_edit_parameter_pass | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_image_edit_with_seed_zero | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_image_edit_with_seed_zero_single_stage | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_invalid_n_parameter | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_invalid_size | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_invalid_size_parse_error | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_missing_prompt | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_model_field_omitted_works | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_model_not_loaded | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_models_endpoint | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_models_endpoint_no_engine | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_parameter_validation | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_parameters_passed_through | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_parse_size_edge_cases | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_parse_size_invalid | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_parse_size_negative | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_parse_size_valid | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_url_response_format_not_supported | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_with_custom_parameters | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_with_negative_prompt | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_with_seed | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_with_seed_zero | @pytest.mark.core_model \| @pytest.mark.cpu |
| entrypoints/openai_api/test_serving_chat_metrics.py | test_omni_chat_completion_response_metrics | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_omni_chat_completion_stream_response_metrics | @pytest.mark.core_model \| @pytest.mark.cpu |
| entrypoints/openai_api/test_serving_chat_sampling_params.py | test_apply_request_overrides_applies_values | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_apply_request_overrides_clones_params | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_apply_request_overrides_preserves_defaults | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_get_comprehension_stage_index_finds_first_stage | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_get_comprehension_stage_index_finds_second_stage | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_get_comprehension_stage_index_raises_when_not_found | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_max_tokens_uses_yaml_default_when_not_specified | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_multiple_params_override_together | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_non_comprehension_stages_use_cloned_defaults | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_openai_sampling_fields_contains_expected_fields | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_preserves_yaml_defaults_when_no_request_params | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_request_frequency_penalty_overrides | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_request_max_tokens_overrides_yaml_default | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_request_presence_penalty_overrides | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_request_seed_overrides_yaml_default | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_request_temperature_overrides_yaml_default | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_request_top_p_overrides_yaml_default | @pytest.mark.core_model \| @pytest.mark.cpu |
| entrypoints/openai_api/test_serving_speech.py | test_app | @pytest.mark.core_model \| @pytest.mark.cpu \| @pytest.fixture |
|  | test_build_tts_params | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_create_speech_invalid_format | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_create_speech_mp3_format | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_create_speech_success | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_estimate_prompt_len_fallback | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_is_tts_detection_no_stage | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_is_tts_detection_with_tts_stage | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_list_voices_endpoint | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_load_supported_speakers | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_max_instructions_length_cli_override | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_max_instructions_length_cli_overrides_stage_config | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_max_instructions_length_default | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_max_instructions_length_stage_config | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_mono_audio_preservation | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_non_streaming_unchanged | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_speed_adjustment | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_speed_adjustment_bypass | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_speed_adjustment_stereo_handling | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_speed_parameter_is_used | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_sse_stream_format_is_blocked | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_stereo_audio_preservation | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_stereo_to_mono_conversion | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_stream_valid | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_stream_validation_errors | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_streaming | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_unsupported_format_fallback | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_validate_instructions_length_uses_cached_value | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_validate_tts_request_basic | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_validate_tts_request_task_types | @pytest.mark.core_model \| @pytest.mark.cpu |
| entrypoints/openai_api/test_video_server.py | test_client | @pytest.mark.core_model \| @pytest.mark.cpu \| @pytest.fixture |
|  | test_i2v_video_generation_form | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_invalid_lora_returns_400 | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_invalid_n_raises_validation_error | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_invalid_response_format_raises_validation_error | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_invalid_seconds_returns_422 | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_invalid_size_format_raises_validation_error | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_invalid_size_parse_returns_500 | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_missing_handler_returns_503 | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_missing_prompt_returns_422 | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_negative_prompt_and_seed_pass_through | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_sampling_params_pass_through | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_seconds_defaults_fps_and_frames | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_size_param_sets_width_height | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_t2v_video_generation_form | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_video_request_validation | @pytest.mark.core_model \| @pytest.mark.cpu |
| entrypoints/test_async_omni_diffusion_config.py | test_default_cache_config_used_when_missing | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_default_stage_config_includes_cache_backend | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_default_stage_devices_from_sequence_parallel | @pytest.mark.core_model \| @pytest.mark.cpu |
| entrypoints/test_cfg_companion_tracker.py | test_companion_lifecycle_failure | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_companion_lifecycle_success | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_companion_lifecycle_timeout | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_companion_tracker_initialization | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_expand_prompts_registers_companions | @pytest.mark.core_model \| @pytest.mark.cpu |
| entrypoints/test_omni_diffusion.py | test_close_sends_shutdown_signal | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_generate_handles_error_messages | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_generate_no_final_output_returns_empty | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_generate_pipeline_and_final_outputs | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_generate_pipeline_with_batch_input | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_generate_raises_on_length_mismatch | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_generate_sampling_params_none_use_default | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_initialize_stage_configs_called_when_none | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_wait_for_stages_ready_timeout | @pytest.mark.core_model \| @pytest.mark.cpu |
| entrypoints/test_omni_input_preprocessor.py | test_process_text_keeps_additional_information | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_process_text_multimodal_skips_empty_payloads | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_process_tokens_keeps_additional_information | @pytest.mark.core_model \| @pytest.mark.cpu |
| entrypoints/test_omni_llm.py | test_close_sends_shutdown_signal | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_generate_handles_error_messages | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_generate_no_final_output_returns_empty | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_generate_pipeline_and_final_outputs | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_generate_raises_on_length_mismatch | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_generate_sampling_params_none_use_default | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_initialize_stage_configs_called_when_none | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_wait_for_stages_ready_timeout | @pytest.mark.core_model \| @pytest.mark.cpu |
| entrypoints/test_omni_new_request_data.py | test_omni_new_request_data_allows_missing_payloads | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_omni_new_request_data_copies_payloads | @pytest.mark.core_model \| @pytest.mark.cpu |
| entrypoints/test_omni_stage_diffusion_config.py | test_build_od_config_includes_diffusion_fields | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_build_od_config_respects_explicit_config | @pytest.mark.core_model \| @pytest.mark.cpu |
| entrypoints/test_stage_utils.py | test_set_stage_devices_npu_platform | @pytest.mark.core_model \| @pytest.mark.cpu \| @pytest.mark.usefixtures("clean_gpu_memory_between_tests") |
|  | test_set_stage_devices_respects_logical_ids | @pytest.mark.core_model \| @pytest.mark.cpu \| @pytest.mark.usefixtures("clean_gpu_memory_between_tests") |
| entrypoints/test_utils.py | test_dict_preserves_key_types | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_dict_with_counter_values | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_dict_with_dataclass_values | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_dict_with_mixed_types | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_dict_with_nested_values | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_dict_with_none_values | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_dict_with_recursive_structure | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_dict_with_set_values | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_empty_dict | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_filters_omni_diffusion_config_union_dataclass | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_filters_omni_engine_args_unknown_fields | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_integration_with_convert_dataclasses | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_invalid_dataclass_raises_error | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_invalid_kwargs_type_raises_error | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_simple_dict | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_simple_filtering | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_uses_filter_dict_like_object | @pytest.mark.core_model \| @pytest.mark.cpu |
| metrics/test_stats.py | test_build_and_log_summary_e2e_only | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_build_and_log_summary_multiple_requests | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_orchestrator_aggregator_builds_summary | @pytest.mark.core_model \| @pytest.mark.cpu |
| model_executor/models/qwen2_5_omni/test_audio_length.py | test_cap_and_align_mel_length_no_mismatch | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_resolve_max_mel_frames_default | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_resolve_max_mel_frames_explicit | @pytest.mark.core_model \| @pytest.mark.cpu |
| model_executor/models/qwen2_5_omni/test_qwen2_5_omni_embed.py | test_audio_only | (无) |
|  | test_basic_interleaved | (无) |
|  | test_interleaved | (无) |
|  | test_interleaved_use_audio_in_video | (无) |
|  | test_mixed_modalities_audio_goes_to_audio_pos | (无) |
|  | test_no_audio | (无) |
|  | test_non_interleaved_audio_then_video | (无) |
|  | test_non_interleaved_with_image | (无) |
|  | test_text_positions_unchanged | (无) |
|  | test_video_only | (无) |
| model_executor/models/qwen3_tts/test_cuda_graph_decoder.py | test_batch_size_gt1_falls_back | @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required") |
|  | test_chunked_decode_exact_size_equivalence | @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required") |
|  | test_chunked_decode_shape_match | @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required") |
|  | test_deterministic_across_calls | @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required") |
|  | test_disabled_wrapper_matches_eager | @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required") |
|  | test_exact_size_numerical_equivalence | @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required") |
|  | test_fallback_eager_exact_match | @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required") |
|  | test_padded_interior_positions_close | @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required") |
|  | test_padded_output_bounded | @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required") |
|  | test_padded_output_shape_and_length | @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required") |
|  | test_single_frame | @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required") |
| model_executor/stage_input_processors/test_qwen3_tts_async_chunk.py | test_talker2code2wav_async_chunk_does_not_emit_empty_chunk_when_not_finished | (无) |
|  | test_talker2code2wav_async_chunk_emits_eof_marker_when_finished_with_no_frames | (无) |
|  | test_talker2code2wav_async_chunk_flushes_tail_when_finished_without_pooler_output | (无) |
| test_outputs.py | test_encoder_prompt_token_ids_property | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_from_diffusion | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_from_pipeline | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_multimodal_output_property | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_num_cached_tokens_property | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_outputs_empty_when_no_request_output | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_outputs_property | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_prompt_token_ids_none_when_no_request_output | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_prompt_token_ids_property | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_to_dict_diffusion | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_to_dict_pipeline | @pytest.mark.core_model \| @pytest.mark.cpu |
| worker/test_gpu_generation_model_runner.py | test_sample_tokens_dict_output | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_sample_tokens_list_allows_none_output | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_sample_tokens_list_output | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_sample_tokens_tensor_output | @pytest.mark.core_model \| @pytest.mark.cpu |
| worker/test_omni_gpu_model_runner.py | test_maybe_attach_mimo_audio_req_infos_enriches_dict | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_maybe_attach_mimo_audio_req_infos_no_req_state_returns_input | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_talker_mtp_forward_cpu_empty_batch_noop | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_talker_mtp_forward_cpu_updates_inputs_and_info | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_update_intermediate_buffer_accumulates | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_update_intermediate_buffer_skips_empty_update | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_update_intermediate_buffer_skips_unknown_req_id | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_update_intermediate_buffer_writes_to_buffer_and_setattr | @pytest.mark.core_model \| @pytest.mark.cpu |
| worker/test_process_gpu_memory.py | test_empty | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_integer_indices | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_mig_ids | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_raises_on_invalid_device | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_raises_on_invalid_uuid | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_returns_false_when_nvml_fails | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_returns_memory_for_current_process | @pytest.mark.core_model \| @pytest.mark.cpu \| @pytest.mark.skipif(not os.path.exists("/dev/nvidia0"), reason="No GPU") |
|  | test_returns_none_on_nvml_init_failure | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_returns_true_when_nvml_works | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_returns_zero_when_process_not_found | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_spaces | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_uses_uuid_when_provided | @pytest.mark.core_model \| @pytest.mark.cpu |
|  | test_uuids | @pytest.mark.core_model \| @pytest.mark.cpu |
