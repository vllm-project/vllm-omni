"""Ray Data batch inference for Qwen3-Omni.

Uses Ray Data's ``vLLMEngineProcessorConfig`` to drive a single vLLM
engine actor

Select the model + dataset via Hydra overrides::
    python batch_inference_ray.py model=qwen3_omni_30b_a3b_thinking
    python batch_inference_ray.py model=qwen3_omni_30b_a3b_thinking dataset=coco_caption_2017
    python batch_inference_ray.py model=qwen3_omni_30b_a3b_thinking \\
        vllm.tensor_parallel_size=4 vllm.max_num_seqs=8 \\
        vllm.max_concurrent_batches=4 batch_size=32
"""

import logging
import os
import re
import time
import traceback
from typing import Any

import hydra
import ray
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

try:
    from ray.data.llm import build_llm_processor
except ImportError:
    from ray.data.llm import build_processor as build_llm_processor
from ray.data.llm import vLLMEngineProcessorConfig
from ray.llm._internal.batch.stages.vllm_engine_stage import vLLMEngineStageUDF
from utils import (
    apply_env,
    compute_benchmark_metrics,
    generate_random_mm_ray_rows,
    load_vision_dataset,
    print_bench_metrics,
    save_outputs,
    save_results_json,
    setup_fd_tee,
    vision_preprocess,
)
from vllm.config import ProfilerConfig

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Modality flags
# ----------------------------------------------------------------------


def _modality_flags(query_type: str) -> tuple[dict, dict]:
    """Translate ``query_type`` into engine + processor modality flags.

    Returns ``(limit_mm_per_prompt, processor_kwargs)`` where the first dict
    goes into ``engine_kwargs`` (controls vLLM's modality capacities) and
    the second into ``vLLMEngineProcessorConfig(...)`` (controls Ray's
    pre-engine image-prep stage).  ``vLLMEngineProcessorConfig`` currently
    only accepts ``has_image`` -- audio/video media URLs are decoded by
    vLLM's own multimodal input parser downstream.
    """
    mods = {"image": 0, "audio": 0, "video": 0}
    if query_type == "image":
        mods["image"] = 1
    elif query_type == "audio":
        mods["audio"] = 1
    elif query_type == "video":
        mods["video"] = 1
    elif query_type == "mixed":
        mods = {"image": 1, "audio": 1, "video": 1}
    want_image_prep = mods["image"] > 0
    # Ray builds differ: some accept has_image / prepare_multimodal_stage on
    # vLLMEngineProcessorConfig; newer Pydantic models forbid extras — filter below.
    processor_kwargs = {
        "has_image": want_image_prep,
        "prepare_multimodal_stage": want_image_prep,
    }
    return mods, processor_kwargs


def _processor_kwargs_for_ray(processor_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only kwargs declared on ``vLLMEngineProcessorConfig`` (extra=forbid)."""
    try:
        allowed = set(vLLMEngineProcessorConfig.model_fields.keys())
    except AttributeError:
        try:
            allowed = set(vLLMEngineProcessorConfig.__fields__.keys())  # type: ignore[attr-defined]
        except AttributeError:
            return processor_kwargs
    filtered = {k: v for k, v in processor_kwargs.items() if k in allowed}
    dropped = set(processor_kwargs) - set(filtered)
    if dropped:
        log.warning(
            "vLLMEngineProcessorConfig omits unsupported keys on this Ray build: %s",
            sorted(dropped),
        )
    return filtered


# ----------------------------------------------------------------------
# vLLM runtime env (worker-side; layered on top of cfg.env via Ray runtime_env)
# ----------------------------------------------------------------------


def get_vllm_runtime_envvars(cfg: DictConfig) -> dict:
    """Build the vLLM worker env-var dict.

    Vendor-conditional on torch's HIP vs CUDA build.  Reads
    ``cfg.vllm.enable_compile_cache`` from the namespaced framework config.
    """
    enable_compile_cache = bool(cfg.vllm.get("enable_compile_cache", False))
    if torch.version.hip:
        # ROCm-specific settings with AITER disabled for Qwen3-Omni compatibility.
        return dict(
            VLLM_WORKER_MULTIPROC_METHOD="spawn",
            VLLM_ROCM_USE_AITER="0",  # Disabled due to GEMM issues
            VLLM_ROCM_USE_AITER_MHA="0",  # Disabled due to GEMM issues
            VLLM_ROCM_USE_AITER_RMSNORM="0",
            VLLM_ROCM_USE_TRITON_ROPE="1",
            VLLM_DISABLE_COMPILE_CACHE="0" if enable_compile_cache else "1",
            VLLM_WORKER_TIMEOUT="7200",
            VLLM_LOGGING_LEVEL="INFO",
            VLLM_RPC_TIMEOUT="7200",
            VLLM_ENGINE_ITERATION_TIMEOUT_S="7200",
            VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="7200",
        )
    if torch.version.cuda:
        return dict(
            VLLM_WORKER_MULTIPROC_METHOD="spawn",
            VLLM_DISABLE_COMPILE_CACHE="0" if enable_compile_cache else "1",
            VLLM_WORKER_TIMEOUT="7200",
            VLLM_LOGGING_LEVEL="INFO",
        )
        raise OSError("Unsupported GPU backend. Only ROCm and CUDA are supported.")


# ----------------------------------------------------------------------
# Ray vLLMEngineProcessorConfig builder
# ----------------------------------------------------------------------


def create_omni_config(cfg: DictConfig) -> vLLMEngineProcessorConfig:
    """Build the Ray Data vLLM engine processor config from cfg.vllm.* and cfg.*."""
    v = cfg.vllm
    if v.enforce_eager:
        compilation_config = {"mode": 0, "custom_ops": ["-rms_norm", "-quant_fp8"]}
    else:
        compilation_config = {
            "mode": 3,
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "custom_ops": ["-rms_norm", "-quant_fp8"],
        }
        if cfg.profiler.enable:
            compilation_config["cudagraph_capture_sizes"] = [v.max_num_seqs]

    limit_mm_per_prompt, processor_kwargs = _modality_flags(cfg.query_type)
    ray_processor_kwargs = _processor_kwargs_for_ray(processor_kwargs)

    engine_kwargs = dict(
        tensor_parallel_size=v.tensor_parallel_size,
        data_parallel_size=v.data_parallel_size,
        trust_remote_code=True,
        limit_mm_per_prompt=limit_mm_per_prompt,
        gpu_memory_utilization=v.gpu_memory_utilization,
        max_model_len=v.max_model_len,
        max_num_seqs=v.max_num_seqs,
        max_num_batched_tokens=v.max_num_batched_tokens,
        kv_cache_dtype=v.kv_cache_dtype,
        mm_encoder_tp_mode="data",
        enable_expert_parallel=v.enable_expert_parallel,
        async_scheduling=v.enable_async_scheduling,
        distributed_executor_backend="mp" if v.enable_async_scheduling else "ray",
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        compilation_config=compilation_config,
    )
    if v.get("all2all_backend") is not None:
        engine_kwargs["all2all_backend"] = v.all2all_backend

    if cfg.profiler.enable:
        profiler_dir = os.path.join(os.path.abspath(cfg.output_dir), "profiler_logs")
        os.makedirs(profiler_dir, exist_ok=True)
        engine_kwargs["profiler_config"] = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=profiler_dir,
            torch_profiler_record_shapes=True,
            torch_profiler_with_stack=bool(cfg.profiler.get("with_stack", False)),
            torch_profiler_use_gzip=True,
            ignore_frontend=True,
        )
        log.info("Profiling enabled via engine_kwargs profiler_config (dir=%s)", profiler_dir)

    return vLLMEngineProcessorConfig(
        model_source=cfg.model_name,
        engine_kwargs=engine_kwargs,
        runtime_env=dict(env_vars=get_vllm_runtime_envvars(cfg)),
        batch_size=cfg.batch_size,
        max_concurrent_batches=v.max_concurrent_batches,
        concurrency=1,
        **ray_processor_kwargs,
        experimental={
            "max_tasks_in_flight_per_actor": 2 * v.max_concurrent_batches,
        },
    )


# ----------------------------------------------------------------------
# Profiler wrapper for the Ray vLLMEngineStage UDF
# ----------------------------------------------------------------------


class vLLMEngineStageUDFWithProfiler(vLLMEngineStageUDF):
    def __init__(self, *args, **kwargs):
        """Wrap ``_generate_async`` with profile start/stop calls.

        Wraps the original method rather than replacing it, so upstream
        changes to ``_generate_async`` are automatically inherited.
        """
        super().__init__(*args, **kwargs)
        original_generate = self.llm._generate_async
        self.llm._generate_async = self._make_profiled_wrapper(original_generate)

    def _make_profiled_wrapper(self, original_method):
        async def profiled_generate_async(request):
            await self.llm.engine.start_profile()
            try:
                result = await original_method(request)
            finally:
                await self.llm.engine.stop_profile()
            return result

        return profiled_generate_async


# ----------------------------------------------------------------------
# Ray Data stats parser (kept; produces the `ray_data_stats` side field)
# ----------------------------------------------------------------------


def parse_ray_data_stats(stats_str: str) -> dict:
    """Summarise Ray Data's stats string into a structured dict."""
    result: dict = {}

    runtime_section = re.search(r"Runtime Metrics:\n(.*?)(?:\n\n|$)", stats_str, re.DOTALL)
    if runtime_section:
        result["runtime_metrics"] = [
            {
                "operator": name.strip(),
                "time": float(time_s),
                "time_unit": unit,
                "percent": float(pct),
            }
            for name, time_s, unit, pct in re.findall(
                r"\*\s+(.+?):\s+([\d.]+)(us|ms|s)\s+\(([\d.]+)%\)",
                runtime_section.group(1),
            )
        ]
    else:
        result["runtime_metrics"] = []

    return result


def _log_ray_data_stats(ray_data_stats: dict | None) -> None:
    """Log Ray Data per-operator runtime metrics (rendered alongside the
    shared benchmark metrics for parity with the pre-Hydra output)."""
    if not ray_data_stats or not ray_data_stats.get("runtime_metrics"):
        return
    log.info("Ray Data Runtime Metrics:")
    ops = ray_data_stats["runtime_metrics"]
    max_name_len = max(len(op["operator"]) for op in ops) + 1
    for op in ops:
        log.info(
            "  %s %s%s (%.3f%%)",
            f"{op['operator']:<{max_name_len}}",
            op["time"],
            op["time_unit"],
            op["percent"],
        )


# ----------------------------------------------------------------------
# Pipeline runner
# ----------------------------------------------------------------------


def run_ray_pipeline(cfg: DictConfig, repeat_idx: int = 0):
    """Run one Ray Data pipeline iteration.

    Returns ``(results, elapsed, ray_data_stats)`` where:
      * ``results`` is a flat list of Omni-schema entries
        (``request_id`` / ``final_output_type`` / ``text`` / ``num_input_tokens`` /
        ``num_output_tokens`` / ``prompt`` / ``batch_uuid``).
      * ``elapsed`` is the wall time around streaming consumption of the
        processor output via
        ``processor(ds).iter_batches(batch_size=cfg.batch_size)``, which
        lets Ray Data drain blocks incrementally while vLLM keeps
        generating (and avoids a full ``take_all()`` materialisation).
      * ``ray_data_stats`` is the dict produced by ``parse_ray_data_stats``
        (used as a top-level side field in the saved JSON).
    """
    config = create_omni_config(cfg)

    # Dataset selection -- driven by the Hydra config group choice
    # (cfg.dataset_kind is stashed from runtime.choices in main()).
    if cfg.dataset_kind == "coco_caption_2017":
        ds = load_vision_dataset()
        if ds is None:
            return [], 0.0, None
        dataset_count = ds.count()
        log.info("Total dataset size: %d samples", dataset_count)
        n_samples = cfg.num_prompts if 0 < cfg.num_prompts < dataset_count else dataset_count
        ds = ds.limit(n_samples)
        preprocess_fn = vision_preprocess
    else:  # random_mm
        n_samples = cfg.num_prompts if cfg.num_prompts > 0 else cfg.batch_size
        bucket_spec = OmegaConf.to_container(cfg.bucket_config, resolve=True)
        rows, limit_mm = generate_random_mm_ray_rows(
            model_name=cfg.model_name,
            seed=cfg.seed,
            num_requests=n_samples,
            input_len=cfg.input_len,
            output_len=cfg.output_len,
            range_ratio=cfg.range_ratio,
            query_type=cfg.query_type,
            bucket_spec=bucket_spec,
        )
        log.info(
            "Sampled %d random-mm rows (query_type=%s, limit_mm_per_prompt=%s)",
            len(rows),
            cfg.query_type,
            limit_mm,
        )
        ds = ray.data.from_items(rows)
        preprocess_fn = None  # rows are already shaped for build_llm_processor

    if repeat_idx > 0:
        ds = ds.map(lambda row: {**row, "_meta_repeat": repeat_idx})

    log.info("Will process: %d samples", n_samples)
    ctx = ray.data.DataContext.get_current()
    ctx.verbose_stats_logs = True
    ctx.enable_auto_log_stats = True

    if preprocess_fn is not None:
        processor = build_llm_processor(config, preprocess=preprocess_fn)
    else:
        processor = build_llm_processor(config)

    # Only patch the vLLMEngineStage on the 2nd iteration (actual profiling run).
    # First iteration (repeat_idx=0) is warmup only -- no profiling overhead.
    if cfg.profiler.enable and repeat_idx > 0:
        if "vLLMEngineStage" not in processor.stages:
            raise ValueError(
                "vLLMEngineStage not found in processor stages. Available: " + str(list(processor.stages.keys()))
            )
        log.info("=" * 70)
        log.info(">>> PROFILING MEASUREMENT ITERATION %d", repeat_idx)
        log.info(">>> Patching Ray vLLM EngineStage for profiling support")
        log.info(">>> Available stages: %s", list(processor.stages.keys()))
        log.info("=" * 70)
        processor.stages["vLLMEngineStage"].fn = vLLMEngineStageUDFWithProfiler
    elif cfg.profiler.enable and repeat_idx == 0:
        log.info("=" * 70)
        log.info(">>> WARMUP ITERATION - No profiling enabled")
        log.info(">>> This iteration warms up the model, kernels, and cache")
        log.info("=" * 70)

    log.info("Omni processor configured successfully")
    log.info("Model: %s", config.model_source)
    log.info(
        "Modalities (engine limit_mm_per_prompt=%s)",
        config.engine_kwargs.get("limit_mm_per_prompt"),
    )

    processor_ds = processor(ds)
    start_time = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for batch in processor_ds.iter_batches(batch_size=cfg.batch_size):
        rows.extend(dict(zip(batch.keys(), values)) for values in zip(*batch.values()))
    elapsed = time.perf_counter() - start_time

    # Translate each Ray row into an Omni-schema entry.  ``batch_uuid``
    # preserves Ray Data's per-batch identity in the flat list (consumers
    # can recover per-batch breakdowns via groupby).
    results: list[dict[str, Any]] = []
    for row in rows:
        row_metrics = row.get("metrics") or {}
        entry = {
            "request_id": row.get("request_id"),
            "final_output_type": "text",
            "text": row.get("generated_text", ""),
            "num_output_tokens": int(row_metrics.get("num_generation_tokens", 0)),
            "num_input_tokens": int(row.get("num_input_tokens", 0)),
            "prompt": "",
            "batch_uuid": row.get("batch_uuid"),
        }
        results.append(entry)

    ray_data_stats = None
    try:
        stats_str = processor_ds.stats() if processor_ds is not None else None
        if stats_str:
            ray_data_stats = parse_ray_data_stats(stats_str)
    except Exception as e:
        log.warning("Failed to parse ray_data_stats: %s", e)

    return results, elapsed, ray_data_stats


# ----------------------------------------------------------------------
# Hydra entry point
# ----------------------------------------------------------------------


@hydra.main(config_path="../config", config_name="benchmark", version_base=None)
def main(cfg: DictConfig) -> None:
    """Ray Data batch inference driver."""
    cfg.mode = "ray"
    cfg.model_short_name = HydraConfig.get().runtime.choices.get("model", "unknown_model")
    cfg.dataset_kind = HydraConfig.get().runtime.choices.get("dataset", "random_mm")

    os.makedirs(cfg.output_dir, exist_ok=True)
    log_path = os.path.join(cfg.output_dir, f"{cfg.log_name}.log")
    setup_fd_tee(log_path)

    apply_env(cfg)

    if not (torch.cuda.is_available() or torch.version.hip):
        log.warning("Skipping Ray Omni inference run (no GPU available)")
        return

    log.info("=" * 70)
    log.info("Qwen3-Omni Ray Data Batch Inference")
    log.info("=" * 70)
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

    try:
        for repeat_idx in range(cfg.n_repeats):
            log.info("\n%s", "=" * 60)
            log.info("Run %d/%d", repeat_idx + 1, cfg.n_repeats)
            log.info("%s", "=" * 60)

            results, elapsed, ray_data_stats = run_ray_pipeline(cfg, repeat_idx=repeat_idx)

            if not results:
                log.warning("[FAILED] No results returned for run %d", repeat_idx + 1)
            else:
                log.info("[SUCCESS] Inference completed with %d results.", len(results))
                metrics = compute_benchmark_metrics(results, elapsed)
                print_bench_metrics(metrics, idx=repeat_idx)
                _log_ray_data_stats(ray_data_stats)

                is_last = repeat_idx == cfg.n_repeats - 1
                if is_last:
                    if cfg.get("save_results_json", False):
                        extras = {"ray_data_stats": ray_data_stats} if ray_data_stats else None
                        save_results_json(
                            cfg,
                            results,
                            metrics,
                            repeat_idx=repeat_idx,
                            extras=extras,
                        )
                    if cfg.get("save_text_output", False):
                        save_outputs(cfg.output_dir, results)

            # Release Ray resources between repeats.
            if ray.is_initialized():
                log.info("Shutting down Ray from this iteration...")
                ray.shutdown()
                time.sleep(5)

        log.info("Benchmark complete.")
    except Exception as e:
        log.error("Ray Omni inference failed: %s", e)
        traceback.print_exc()
        if ray.is_initialized():
            log.info("Shutting down Ray after error...")
            ray.shutdown()
        raise


if __name__ == "__main__":
    main()
