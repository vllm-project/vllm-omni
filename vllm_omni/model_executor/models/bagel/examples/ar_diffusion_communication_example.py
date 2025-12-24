import argparse
import multiprocessing
import os
import random
import time

import numpy as np
import torch

# 导入 LLM 相关
from vllm.sampling_params import SamplingParams

# 导入 Diffusion 相关
from vllm_omni.entrypoints.omni_diffusion import OmniDiffusion
from vllm_omni.entrypoints.omni_llm import OmniLLM


# ==========================================
# 1. 逻辑封装：OmniLLM 运行函数
# ==========================================
def run_omni_llm_process(args, gpu_id):
    """
    LLM 进程入口
    """
    # 核心：设置显卡可见性，实现物理隔离
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["VLLM_USE_V1"] = "1"

    # 确定性设置
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"进程 [OmniLLM] 启动，分配 GPU: {gpu_id}, PID: {os.getpid()}")

    # 1. 加载 Prompts
    model_name = args.model
    prompts = args.prompts
    if getattr(args, "txt_prompts", None) and args.prompt_type == "text":
        try:
            with open(args.txt_prompts, encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines()]
                prompts = [ln for ln in lines if ln != ""]
                print(f"[LLM Info] Loaded {len(prompts)} prompts from {args.txt_prompts}")
        except Exception as e:
            print(f"[LLM Error] Failed to load prompts: {e}")
            return

    if prompts is None:
        prompts = ["<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"]

    # 2. 初始化 OmniLLM
    omni_llm = OmniLLM(
        model=model_name,
        log_stats=args.enable_stats,
        log_file=("omni_llm_pipeline.log" if args.enable_stats else None),
        init_sleep_seconds=args.init_sleep_seconds,
        batch_timeout=args.batch_timeout,
        init_timeout=args.init_timeout,
        shm_threshold_bytes=args.shm_threshold_bytes,
        worker_backend=args.worker_backend,
        ray_address=args.ray_address,
        stage_configs_path=args.stage_configs_path,
    )

    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=20, stop=["<|im_end|>"])

    # 3. 推理
    t1 = time.time()
    # 注意：这里为了匹配你之前的代码逻辑，只取第一个 prompt 演示
    input_data = [{"prompt": prompts[0]}]
    omni_outputs = omni_llm.generate(input_data, [sampling_params])
    t2 = time.time()

    print(f"==========> [OmniLLM] 推理耗时: {t2 - t1:.4f}s")
    print(f"[OmniLLM Result]: {omni_outputs}")


# ==========================================
# 2. 逻辑封装：Diffusion 运行函数
# ==========================================
def run_diffusion_process(model_path, prompt, gpu_id, use_cache=False, seed=41):
    """
    Diffusion 进程入口
    """
    # 核心：设置显卡可见性
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"进程 [Diffusion] 启动，分配 GPU: {gpu_id}, PID: {os.getpid()}")

    cache_config = {}
    cache_backend = "none"

    if use_cache:
        cache_backend = "cache_dit"
        cache_config = {
            "Fn_compute_blocks": 1,
            "Bn_compute_blocks": 0,
            "max_warmup_steps": 4,
            "residual_diff_threshold": 0.24,
            "max_continuous_cached_steps": 3,
            "enable_taylorseer": False,
            "taylorseer_order": 1,
            "scm_steps_mask_policy": None,
            "scm_steps_policy": "dynamic",
        }
        print("--- [Diffusion] Running WITH Cache-DiT ---")
    else:
        print("--- [Diffusion] Running WITHOUT Cache (Baseline) ---")

    # 初始化 Pipeline
    pipeline = OmniDiffusion(model=model_path, cache_backend=cache_backend, cache_config=cache_config)

    start_time = time.time()
    image = pipeline.generate(prompt, seed=seed)
    end_time = time.time()

    elapsed = end_time - start_time
    print(f"[Diffusion] Generation time: {elapsed:.4f}s")

    output_file = f"bagel_{'cache' if use_cache else 'baseline'}_output.png"
    image.save(output_file)
    print(f"[Diffusion] Image saved to {output_file}")

    pipeline.close()


# ==========================================
# 3. 参数解析与主进程调度
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser()
    # LLM 必需参数
    parser.add_argument("--model", default="/workspace/ByteDance-Seed/BAGEL-7B-MoT/", help="LLM model path")
    # Diffusion 必需参数
    parser.add_argument("--diff-model", default="/workspace/ByteDance-Seed/BAGEL-7B-MoT/", help="Diffusion model path")

    # GPU 分配参数 (默认 LLM 用 0, Diffusion 用 1)
    parser.add_argument("--llm-gpu", type=int, default=0)
    parser.add_argument("--diff-gpu", type=int, default=1)

    # 其他 LLM 选项
    parser.add_argument("--prompts", nargs="+", default=None)
    parser.add_argument("--prompt_type", choices=["text", "audio"], default="text")
    parser.add_argument("--txt-prompts", type=str, default=None)
    parser.add_argument("--enable-stats", action="store_true")
    parser.add_argument("--init-sleep-seconds", type=int, default=20)
    parser.add_argument("--batch-timeout", type=int, default=5)
    parser.add_argument("--init-timeout", type=int, default=300)
    parser.add_argument("--shm-threshold-bytes", type=int, default=65536)
    parser.add_argument("--worker-backend", type=str, default="process", choices=["process", "ray"])
    parser.add_argument("--ray-address", type=str, default=None)
    parser.add_argument("--stage-configs-path", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    # 设置 Multiprocessing 启动方式 (在 CUDA 环境下推荐使用 spawn)
    multiprocessing.set_start_method("spawn", force=True)

    args = parse_args()

    # 1. 准备 Diffusion 参数
    diff_prompt = "A futuristic city skyline at twilight, cyberpunk style"

    # 2. 创建进程
    # 进程1: LLM
    p1 = multiprocessing.Process(target=run_omni_llm_process, args=(args, args.llm_gpu))

    # 进程2: Diffusion
    p2 = multiprocessing.Process(
        target=run_diffusion_process, args=(args.diff_model, diff_prompt, args.diff_gpu, False, 41)
    )

    print(f"主进程启动: 同时派发任务到 GPU {args.llm_gpu} (LLM) 和 GPU {args.diff_gpu} (Diffusion)")

    # 3. 启动
    p1.start()
    p2.start()

    # 4. 等待结束
    p1.join()
    p2.join()

    print("所有推理任务已完成。")
