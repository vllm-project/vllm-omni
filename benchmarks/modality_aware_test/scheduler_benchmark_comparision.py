import asyncio
import time
import torch
import numpy as np
import os
import math
import random
import csv
import gc
import traceback
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

# vLLM / Omni imports
from vllm_omni.entrypoints.async_omni_llm import AsyncOmniLLM
from vllm_omni.engine.arg_utils import AsyncOmniEngineArgs
from vllm_omni.entrypoints.utils import load_stage_configs_from_yaml
from vllm.usage.usage_lib import UsageContext
from vllm import SamplingParams
from omegaconf import OmegaConf
from vllm.model_executor.models.qwen3_omni_moe_thinker import  _get_feat_extract_output_lengths



logger = logging.getLogger("benchmark")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


MODEL_PATH = "/root/data/hf_cache/Qwen3-Omni-30B-A3B-Instruct"
TARGET_STAGE_ID = 0
CSV_RESULT_PATH = "/root/data/benchmark_results.csv"
GLOBAL_SEED = 42  
BASE_STAGE_CONFIGS_PATH = "/root/data/vllm-omni/benchmarks/modality_aware_test/stage_configs/baseline_fcfs.yaml"
OPT_STAGE_CONFIGS_PATH = "/root/data/vllm-omni/benchmarks/modality_aware_test/stage_configs/modality_aware.yaml"


QPS_LIST = [5,10,20,50]
DURATION_LIST = [20,10,5,2]

# The "starve_threshold" should be close to the time required for a batch of multimodal data
# to finish encoding + prefilling under single-step scheduling.
# In the alpha system tp=2 test setting, where overall throughput is low and the TP size is small,
# encoding + prefilling is relatively slow, so we set starve_threshold = 10s here.
# In an optimized production system, this value can be tuned to the average duration
# of a single-step encoding + prefilling, without significantly affecting system fairness.
OPT_OVERRIDES_LIST = [
    {"additional_config": {"encoder_gain_threshold": 512, "starve_threshold": 10}},
    {"additional_config": {"encoder_gain_threshold": 768, "starve_threshold": 10}},
    {"additional_config": {"encoder_gain_threshold": 1024, "starve_threshold": 10}},
    {"additional_config": {"encoder_gain_threshold": 1024, "starve_threshold": 10}},
]


@dataclass
class RequestMetrics:
    req_id: str
    arrival_time: float
    start_execution_time: float = -1
    first_token_time: float = -1
    end_time: float = -1
    output_len: int = 0
    success: bool = False

    @property
    def ttft(self):
        if self.first_token_time > 0 and self.arrival_time > 0:
            return (self.first_token_time - self.arrival_time) * 1000  # ms
        return None

    @property
    def e2e_latency(self):
        if self.end_time > 0 and self.arrival_time > 0:
            return self.end_time - self.arrival_time
        return None


def load_engine_args_from_stage_config(
    model: str,
    stage_configs_path: str,
    stage_id: int = 0,
    override_args: dict = None,
) -> AsyncOmniEngineArgs:
    stage_args_list = load_stage_configs_from_yaml(stage_configs_path)
    
    target_stage = None
    for stage_arg in stage_args_list:
        if stage_arg.stage_id == stage_id:
            target_stage = stage_arg
            break
    
    if target_stage is None:
        raise ValueError(f"Stage ID {stage_id} not found in {stage_configs_path}")
    
    engine_args_dict = OmegaConf.to_container(target_stage.engine_args, resolve=True)
    
    if override_args:
        logger.info(f"Applying override args: {override_args}")
        engine_args_dict.update(override_args)
    
    print(engine_args_dict)
    return AsyncOmniEngineArgs(model=model, **engine_args_dict)


def prepare_dataset(total_requests: int, seed: int) -> List[tuple]:
    """
    基于固定种子生成请求数据，确保 Baseline 和 Optimized 跑的是完全相同的请求内容。
    """
    # 设定random和torch的局部随机种子
    rng = random.Random(seed)
    torch.manual_seed(seed)
    
    logger.info(f"⚡ Pre-generating {total_requests} requests with seed {seed}...")
    
    dataset = []
    DEFAULT_SYSTEM = "You are a helpful assistant."
    PROMPT_SUFFIX = "Describe the content in detail."
    
    for i in range(total_requests):
        rid = f"req_{i}"
        
        # 1. 确定模态
        rand_val = rng.random()
        if rand_val < 0.5:
            mod_type = "text"
        elif rand_val < 0.7:
            mod_type = "image"
        elif rand_val < 0.9:
            mod_type = "audio"
        else:
            mod_type = "video"
        
        ein = {}

        if mod_type == "text":
            prompt = (
                f"<|im_start|>system\n{DEFAULT_SYSTEM}<|im_end|>\n"
                f"<|im_start|>user\nQuestion: {rid} What is 1+1?<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            ein = {"prompt": prompt}
            
        elif mod_type == "image":
            mm_placeholder_prompt_ids=[[151655]*256]
            attention_mask=torch.ones((1,256), dtype=torch.int64)
            pixel_values = torch.randn((1024, 1536), dtype=torch.bfloat16)
            image_grid_thw = torch.tensor([[1,32,32]], dtype=torch.int64)
            use_audio_in_video= torch.tensor([False], dtype=torch.bool)
            
            prompt = (
                f"<|im_start|>system\n{DEFAULT_SYSTEM}<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n{PROMPT_SUFFIX}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            
            ein = {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": {
                        "mm_placeholder_prompt_ids":mm_placeholder_prompt_ids,
                        "attention_mask":attention_mask,
                        "pixel_values": pixel_values,
                        "image_grid_thw": image_grid_thw,
                        "use_audio_in_video": use_audio_in_video,
                    }
                }
            }
            
        elif mod_type == "video":
            mm_placeholder_prompt_ids=[[151656]*576]
            attention_mask=torch.ones((1,576), dtype=torch.int64)
            pixel_values_videos = torch.randn((2304, 1536), dtype=torch.bfloat16)
            video_grid_thw = torch.tensor([[4, 24, 24]], dtype=torch.int64)
            video_second_per_grid = torch.tensor([2.0], dtype=torch.bfloat16)
            second_per_grid_ts = torch.tensor([2.0], dtype=torch.bfloat16)
            use_audio_in_video = torch.tensor([False],dtype=torch.bool)
            
            prompt = (
                f"<|im_start|>system\n{DEFAULT_SYSTEM}<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>\n{PROMPT_SUFFIX}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            
            ein = {
                "prompt": prompt,
                "multi_modal_data": {
                    "video": {
                        "mm_placeholder_prompt_ids":mm_placeholder_prompt_ids,
                        "attention_mask":attention_mask,
                        "pixel_values_videos": pixel_values_videos,
                        "video_grid_thw": video_grid_thw,
                        "video_second_per_grid": video_second_per_grid,
                        "second_per_grid_ts": second_per_grid_ts,
                        "use_audio_in_video": use_audio_in_video,
                    }
                }
            }

        elif mod_type == "audio":
            feature_len = rng.randint(1800,2300)
            audio_feature_lengths = torch.tensor([feature_len], dtype=torch.int64)
            feat_lengths, output_lengths=_get_feat_extract_output_lengths(audio_feature_lengths)
            num_tokens = output_lengths.item()

            mm_placeholder_prompt_ids = [[151675] * num_tokens]
            attention_mask = torch.ones((1, num_tokens), dtype=torch.int64)
            feature_attention_mask = [torch.ones((feature_len,), dtype=torch.float32)]
            input_audio_features = torch.randn((128, feature_len), dtype=torch.bfloat16)
            use_audio_in_video = torch.tensor([False],dtype=torch.bool)
            
            prompt = (
                f"<|im_start|>system\n{DEFAULT_SYSTEM}<|im_end|>\n"
                f"<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>\n{PROMPT_SUFFIX}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            
            ein = {
                "prompt": prompt,
                "multi_modal_data": {
                    "audio": {
                        "mm_placeholder_prompt_ids":mm_placeholder_prompt_ids,
                        "attention_mask": attention_mask,
                        "feature_attention_mask": feature_attention_mask,
                        "input_audio_features": input_audio_features,
                        "audio_feature_lengths": audio_feature_lengths,
                        "use_audio_in_video": use_audio_in_video,
                    }
                }
            }

        # 为了可复现，SamplingParams 的 seed 也要固定
        sp = SamplingParams(max_tokens=20, temperature=0.0, ignore_eos=True, seed=seed)
        dataset.append((rid, ein, sp))

    return dataset


async def run_experiment_round(
    exp_name: str,
    config_path: str,
    qps: float,
    duration: int,
    seed: int,
    override_args: dict = None
):
    """
    Init Engine -> Run Benchmark -> Collect Metrics -> Shutdown -> Cleanup
    """
    logger.info(f"\n{'='*60}\n🧪 Starting Experiment: {exp_name} | QPS={qps} | Dur={duration}s\n{'='*60}")
    
    estimated_requests = int(qps * duration * 1.5) + 10
    dataset = prepare_dataset(estimated_requests, seed)
    

    rng_arrival = random.Random(seed)

    arrival_intervals = [rng_arrival.expovariate(qps) for _ in range(estimated_requests)]
    
    try:
        engine_args = load_engine_args_from_stage_config(
            model=MODEL_PATH,
            stage_configs_path=config_path,
            stage_id=TARGET_STAGE_ID,
            override_args=override_args,
        )
        usage_context = UsageContext.OPENAI_API_SERVER
        vllm_config = engine_args.create_engine_config(usage_context=usage_context)
        
        torch.manual_seed(seed)
        
        stage_engine = AsyncOmniLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            engine_args=engine_args,
        )
        await stage_engine.reset_mm_cache()
    except Exception as e:
        logger.error(f"Failed to init engine for {exp_name}: {e}")
        traceback.print_exc()
        return None

    req_metrics: Dict[str, RequestMetrics] = {}
    pending_tasks = []
    
    async def _process_req(rid, ein, sp, arrival_ts):
        rm = RequestMetrics(req_id=rid, arrival_time=arrival_ts)
        req_metrics[rid] = rm
        
        try:
            gen_iterator = stage_engine.generate(ein, sp, request_id=rid)
            
            async for output in gen_iterator:
                now = time.time()

                if rm.first_token_time == -1:
                    rm.first_token_time = now
                    rm.start_execution_time = now 
                
                if output.outputs:
                    rm.output_len = len(output.outputs[0].token_ids)
                
                if output.finished:
                    rm.end_time = now
                    rm.success = True
                    return
        except Exception as e:
            logger.error(f"Req {rid} failed:\n{traceback.format_exc()}")

    logger.info("🚀 Workload Started...")
    start_time = time.time()
    

    current_time = 0.0
    req_count = 0
    
    while current_time < duration and req_count < len(dataset):
        interval = arrival_intervals[req_count]
        await asyncio.sleep(interval)
        current_time += interval
        
        real_arrival_time = time.time()
        
        rid, ein, sp = dataset[req_count]
        
        task = asyncio.create_task(_process_req(rid, ein, sp, real_arrival_time))
        pending_tasks.append(task)
        req_count += 1

    logger.info(f"🛑 Stopped sending requests after {duration}s. Waiting for pending tasks...")
    if pending_tasks:
        await asyncio.gather(*pending_tasks)
    
    total_benchmark_time = time.time() - start_time
    
    success_reqs = [m for m in req_metrics.values() if m.success]
    total_tokens = sum([m.output_len for m in success_reqs])
    
    ttfts = [m.ttft for m in success_reqs if m.ttft is not None]

    actual_qps = req_count / total_benchmark_time
    avg_ttft = np.mean(ttfts) if ttfts else 0
    p99_ttft = np.percentile(ttfts, 99) if ttfts else 0
    throughput = total_tokens / total_benchmark_time
    
    result = {
        "scenario": exp_name,
        "target_qps": qps,
        "duration": duration,
        "actual_qps": round(actual_qps, 2),
        "req_count": req_count,
        "throughput_tokens_per_s": round(throughput, 2),
        "avg_ttft_ms": round(avg_ttft, 2),
        "p99_ttft_ms": round(p99_ttft, 2)
    }
    
    logger.info(f"📊 Result for {exp_name}: {result}")
    
    stage_engine.shutdown()
    del stage_engine
    
    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()
        await asyncio.sleep(1) 
        
    logger.info("🧹 Engine shutdown and cache cleared.")
    
    return result


async def main():
    file_exists = os.path.isfile(CSV_RESULT_PATH)
    csv_headers = [
        "scenario", "target_qps", "duration", "actual_qps", 
        "req_count", "throughput_tokens_per_s", "avg_ttft_ms", "p99_ttft_ms"
    ]
    
    if not file_exists:
        with open(CSV_RESULT_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()

    
    for (qps, dur), overrides in zip(zip(QPS_LIST, DURATION_LIST), OPT_OVERRIDES_LIST):
        
        # --- 1. Baseline Experiment ---
        base_res = await run_experiment_round(
            exp_name="Baseline",
            config_path=BASE_STAGE_CONFIGS_PATH,
            qps=qps,
            duration=dur,
            seed=GLOBAL_SEED,
            override_args=None 
        )
        
        if base_res:
            with open(CSV_RESULT_PATH, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers)
                writer.writerow(base_res)

        await asyncio.sleep(5) 

        # --- 2. Optimized Experiment ---
        opt_res = await run_experiment_round(
            exp_name="ModalityAware",
            config_path=OPT_STAGE_CONFIGS_PATH,
            qps=qps,
            duration=dur,
            seed=GLOBAL_SEED, 
            override_args=overrides
        )

        if opt_res:
            with open(CSV_RESULT_PATH, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers)
                writer.writerow(opt_res)
        
        await asyncio.sleep(5)

    logger.info(f"✅ All benchmarks completed. Results saved to {CSV_RESULT_PATH}")


if __name__ == "__main__":
    # pip install -e . --no-deps --no-build-isolation
    # python3 -u benchmarks/modality_aware_test/scheduler_benchmark_comparision.py 2>&1 | tee /root/data/benchmark_comparision.log
    asyncio.run(main())

    