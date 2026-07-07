# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Staged smoke test: HiDream-O1-Image pipeline init.

Requires HiDream-O1-Image checkpoint at
/workspace/vllm-omni/.hf_models_cache/HiDream-O1-Image.

Usage:
    python _test_dev/_test_hidream_o1_pipeline_init.py
"""
from __future__ import annotations


def main() -> None:
    model_dir = '/workspace/vllm-omni/.hf_models_cache/HiDream-O1-Image'
    print(f"model_dir[resolved]             : {model_dir!r}")

    # 1: arch resolver
    from vllm_omni.diffusion.data import resolve_model_class_name

    resolved = resolve_model_class_name(model_dir)
    resolved_ok = resolved == "Qwen3VLForConditionalGeneration"
    print(f"1[arch resolve]              : resolved={resolved!r} expected='Qwen3VLForConditionalGeneration' ok={resolved_ok}")
    assert resolved_ok, f"1: arch resolver returned {resolved!r}; registry key/dispatch logic may need updating"

    # 2: registry lookup
    from vllm_omni.diffusion.registry import DiffusionModelRegistry

    pipeline_cls = DiffusionModelRegistry._try_load_model_cls("Qwen3VLForConditionalGeneration")
    reg_not_none = pipeline_cls is not None
    reg_class_ok = reg_not_none and pipeline_cls.__name__ == "HiDreamO1ImagePipeline"
    print(f"2[registry lookup]           : cls_not_none={reg_not_none} cls_name={pipeline_cls.__name__ if reg_not_none else None!r} expected='HiDreamO1ImagePipeline' ok={reg_class_ok}")
    assert reg_class_ok, "2: registry entry missing or wrong class; check _DIFFUSION_MODELS in vllm_omni/diffusion/registry.py"

    # 3: processor + 5 special-token literal encoding (no 30GB load yet)
    from transformers import AutoProcessor, PreTrainedTokenizerBase

    processor = AutoProcessor.from_pretrained(model_dir)
    processor_type = type(processor).__name__
    tokenizer = processor if isinstance(processor, PreTrainedTokenizerBase) else processor.tokenizer
    tokenizer_type = type(tokenizer).__name__

    special_tokens = ["<|boi_token|>", "<|bor_token|>", "<|eor_token|>", "<|bot_token|>", "<|tms_token|>"]
    unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else -1
    token_reports = []
    all_single_token = True
    for literal in special_tokens:
        ids = tokenizer.encode(literal, add_special_tokens=False)
        single = len(ids) == 1
        not_unk = single and ids[0] != unk_id
        token_reports.append(f"{literal}={ids}(single={single},not_unk={not_unk})")
        if not (single and not_unk):
            all_single_token = False
    print(f"3[processor+tokens]          : processor={processor_type!r} tokenizer={tokenizer_type!r} unk_id={unk_id} all_single_token={all_single_token}")
    for report in token_reports:
        print(f"                                    {report}")
    assert all_single_token, "3: at least one special-token literal did not encode to a single non-unk id"

    # 4: full weight load with output_loading_info=True (~30-60s, ~30GB)
    import torch

    from vllm_omni.diffusion.models.hidream_o1_image.hidream_o1_image_transformer import (
        HiDreamO1ImageTransformer,
    )

    model, loading_info = HiDreamO1ImageTransformer.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        output_loading_info=True,
    )
    model.eval()
    missing = loading_info.get("missing_keys", [])
    unexpected = loading_info.get("unexpected_keys", [])
    mismatched = loading_info.get("mismatched_keys", [])
    n_params = sum(p.numel() for p in model.parameters())
    weight_load_ok = len(missing) == 0 and len(unexpected) == 0 and len(mismatched) == 0
    print(f"4[weight loading_info]       : missing={len(missing)} unexpected={len(unexpected)} mismatched={len(mismatched)} n_params={n_params/1e9:.2f}B ok={weight_load_ok}")
    if missing:
        print(f"                                    missing_sample={missing[:5]}")
    if unexpected:
        print(f"                                    unexpected_sample={unexpected[:5]}")
    if mismatched:
        print(f"                                    mismatched_sample={mismatched[:5]}")
    assert weight_load_ok, "4: state_dict alignment broken; backbone/pixel-DiT porting may have missed a key"
    del model, loading_info
    torch.cuda.empty_cache()

    # 5: direct HiDreamO1ImagePipeline(od_config=...) constructor
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.hidream_o1_image import HiDreamO1ImagePipeline

    od_config = OmniDiffusionConfig.from_kwargs(
        model=model_dir,
        dtype=torch.bfloat16,
    )
    pipeline = HiDreamO1ImagePipeline(od_config=od_config, prefix="")

    proc_ok = pipeline.processor is not None
    model_ok = pipeline.model is not None
    eval_ok = model_ok and pipeline.model.training is False
    device_type = None
    if model_ok:
        try:
            device_type = next(pipeline.model.parameters()).device.type
        except StopIteration:
            device_type = "no-params"
    device_ok = device_type in ("cuda", "npu", "xpu")
    dtype_ok = model_ok and next(pipeline.model.parameters()).dtype == pipeline.dtype
    tokenizer_now = pipeline.processor if isinstance(pipeline.processor, PreTrainedTokenizerBase) else pipeline.processor.tokenizer
    all_5_attrs_ok = all(
        getattr(tokenizer_now, attr, None) == literal
        for attr, literal in zip(
            ["boi_token", "bor_token", "eor_token", "bot_token", "tms_token"],
            special_tokens,
        )
    )
    has_ckpt_ok = pipeline.has_real_checkpoint()
    print(f"5[pipeline construct]        : processor_ok={proc_ok} model_ok={model_ok} eval_ok={eval_ok} device_type={device_type!r}(ok={device_ok}) dtype_ok={dtype_ok} 5_special_token_attrs_ok={all_5_attrs_ok} has_real_checkpoint={has_ckpt_ok}")
    assert proc_ok and model_ok and eval_ok and device_ok and dtype_ok and all_5_attrs_ok and has_ckpt_ok, "5: pipeline __init__ post-condition failed"

    # 5b: load_weights() must return the full param-name set — otherwise
    # diffusers_loader's strict "all params covered" check aborts the worker
    # with an EOFError in phase 6, which is hard to trace back.
    expected_names = {name for name, _ in pipeline.named_parameters()}
    loaded_names = pipeline.load_weights(iter(()))
    weights_missing = expected_names - loaded_names
    weights_extra = loaded_names - expected_names
    load_weights_ok = not weights_missing and not weights_extra
    print(f"5b[load_weights coverage]    : returned={len(loaded_names)} expected={len(expected_names)} missing={len(weights_missing)} extra={len(weights_extra)} ok={load_weights_ok}")
    assert load_weights_ok, (
        f"5b: load_weights() must return the full param name set to satisfy "
        f"diffusers_loader strict check; missing_sample={list(weights_missing)[:3]} "
        f"extra_sample={list(weights_extra)[:3]}"
    )
    del pipeline
    torch.cuda.empty_cache()

    # 5c: dummy_run_num_frames = 0 skips DiffusionEngine._dummy_run() warmup
    # so Omni() init doesn't hit our NotImplementedError forward() stub.
    # Remove this workaround once forward() is implemented.
    dummy_frames = getattr(HiDreamO1ImagePipeline, "dummy_run_num_frames", None)
    dummy_frames_ok = dummy_frames == 0
    print(f"5c[dummy_run_num_frames]     : value={dummy_frames} expected=0 ok={dummy_frames_ok} (workaround while forward() is a stub)")
    assert dummy_frames_ok, (
        f"5c: dummy_run_num_frames class attr must be 0 while forward() is a stub; "
        f"got {dummy_frames!r}. See pipeline_hidream_o1_image.py class-level comment."
    )

    # 6: end-to-end Omni(model=...) dispatch (orchestrator + worker).
    # Skip omni.generate: forward() is still a stub.
    from vllm_omni import Omni

    omni = Omni(model=model_dir, dtype=torch.bfloat16)
    omni_ok = omni is not None
    print(f"6[Omni(model=...) init]      : type={type(omni).__name__!r} ok={omni_ok}")
    assert omni_ok, "6: Omni(...) constructor returned falsy"

    print("pass (pipeline init + registry + arch-resolve + weight-load-info + special-token integration all green)")


# __main__ guard is REQUIRED here: Omni() spawns worker processes via
# Python's `spawn` context, which re-imports this module in the child;
# without the guard, the child would re-execute Omni(...) recursively.
if __name__ == "__main__":
    main()


# output:
# model_dir[resolved]             : '/workspace/vllm-omni/.hf_models_cache/HiDream-O1-Image'
# 1[arch resolve]              : resolved='Qwen3VLForConditionalGeneration' expected='Qwen3VLForConditionalGeneration' ok=True
# 2[registry lookup]           : cls_not_none=True cls_name='HiDreamO1ImagePipeline' expected='HiDreamO1ImagePipeline' ok=True
# 3[processor+tokens]          : processor='Qwen3VLProcessor' tokenizer='Qwen2Tokenizer' unk_id=-1 all_single_token=True
#                                     <|boi_token|>=[151669](single=True,not_unk=True)
#                                     <|bor_token|>=[151670](single=True,not_unk=True)
#                                     <|eor_token|>=[151671](single=True,not_unk=True)
#                                     <|bot_token|>=[151672](single=True,not_unk=True)
#                                     <|tms_token|>=[151673](single=True,not_unk=True)
# 4[weight loading_info]       : missing=0 unexpected=0 mismatched=0 n_params=8.80B ok=True
# 5[pipeline construct]        : processor_ok=True model_ok=True eval_ok=True device_type='cuda'(ok=True) dtype_ok=True 5_special_token_attrs_ok=True has_real_checkpoint=True
# 5b[load_weights coverage]    : returned=759 expected=759 missing=0 extra=0 ok=True
# 5c[dummy_run_num_frames]     : value=0 expected=0 ok=True (workaround while forward() is a stub)
# 6[Omni(model=...) init]      : type='Omni' ok=True
# pass (pipeline init + registry + arch-resolve + weight-load-info + special-token integration all green)
