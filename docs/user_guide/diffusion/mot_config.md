This directory contains auto-tuned Triton kernel configurations for the
MoT (Mixture-of-Tokens) GEMM and RMSNorm operators used by BAGEL and other
MoT-architecture diffusion models.

File naming convention:
    device_name=<GPU_NAME>,dtype=<DTYPE>.json

For example:
    device_name=NVIDIA_A100-SXM4-80GB,dtype=w16a16.json

Each JSON file maps (K, N) matrix shapes to a dictionary of batch sizes (M)
and their optimal Triton tile configurations:

    {
        "3584_4608": {              // K=3584, N=4608 (QKV projection)
            "1024": {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "num_warps": 4,
                "num_stages": 3
            },
            ...
        },
        "3584_3584": { ... }        // K=3584, N=3584 (output projection)
    }

Config loading order (3-tier, see ops/mot_gemm.py):
    1. $VLLM_TUNED_CONFIG_FOLDER/<filename>  (env override)
    2. This directory: vllm_omni/diffusion/layers/mot/configs/<filename>
    3. Conservative default config (compiles everywhere, sub-optimal perf)

If no config file matches the current device, a warning is printed with
instructions to run the auto-tuning benchmark.

To generate configs for your hardware:

    python benchmarks/kernels/mot_linear_benchmarks.py \
        --model ByteDance-Seed/BAGEL-7B-MoT \
        --tp-size 1 --dtype w16a16 --tune \
        --save-dir vllm_omni/diffusion/layers/mot/configs/

For multi-GPU tuning (uses Ray for parallel search):

    python benchmarks/kernels/mot_linear_benchmarks.py \
        --model ByteDance-Seed/BAGEL-7B-MoT \
        --tp-size 2 --tune

See benchmarks/kernels/mot_linear_benchmarks.py for full options.
