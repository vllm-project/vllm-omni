# ROCm INT8 verification

Use `run_all.sh` from an AMD Docker container to verify the online INT8 changes on ROCm. The script uses the container's active `python`, so it does not require a virtual environment.

The script runs the following checks:

1. It confirms that PyTorch and vLLM-Omni detect ROCm and at least two GPUs.
2. It runs the affected INT8, BitsAndBytes, MiniMax, and BAGEL test files.
3. It checks the Triton INT8 kernel on one GPU.
4. It checks the AITER INT8 kernel when AITER is installed or explicitly requested.
5. It checks exact full weight and two rank sharded weight parity.
6. It runs MiniMax with DiT TP2 and text encoder TP2.
7. It runs BAGEL single stage image generation with TP2.

## Run the checks

Run the script from the repository root. Set the MiniMax FL2VA checkpoint path before starting.

```bash
VLLM_TEST_MINIMAX_H3_FL2VA_MODEL=/models/MiniMax-H3/FL2VA \
bash rocm_int8_verification/run_all.sh
```

The script stops on the first failed check. It prints the result directory when it starts and when it exits.

## AITER selection

`RUN_AITER=auto` is the default. The script runs the AITER check when Python can find the `aiter` package.

Use the following command when the production container must use AITER:

```bash
VLLM_TEST_MINIMAX_H3_FL2VA_MODEL=/models/MiniMax-H3/FL2VA \
RUN_AITER=1 \
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_USE_AITER_LINEAR=1 \
bash rocm_int8_verification/run_all.sh
```

Use `RUN_AITER=0` when the container does not include AITER.

## Optional settings

| Variable | Default | Purpose |
| --- | --- | --- |
| `GPU_IDS` | `0,1` | Two logical AMD GPU IDs used by the tests and model runs |
| `BAGEL_MODEL` | `ByteDance-Seed/BAGEL-7B-MoT` | BAGEL model ID or local checkpoint path |
| `RUN_AITER` | `auto` | Run the AITER kernel check with `auto`, `1`, or `0` |
| `ROCM_VERIFY_RUN_ID` | UTC timestamp and process ID | Name of the result directory |

## Output locations

The script keeps generated files under `rocm_int8_verification`:

```text
rocm_int8_verification/
├── cache/
│   ├── huggingface/
│   ├── torch_extensions/
│   ├── torchinductor/
│   ├── triton/
│   ├── vllm/
│   └── xdg/
├── runtime/
│   └── run-<random-id>/
└── results/
    └── <run-id>/
        ├── pytest-cache/
        ├── pytest-temp/
        ├── bagel-int8-tp2.png
        ├── minimax-int8-tp2-output.npz
        ├── run-all.log
        └── step logs
```

The script sets `TMPDIR`, `TMP`, and `TEMP` to a short directory under `rocm_int8_verification/runtime`. vLLM creates Unix sockets under this path, so keeping it short avoids the 107 character Unix socket path limit. The script also passes a result directory to pytest through `--basetemp` and `cache_dir`. Model downloads, runtime files, and compiler caches stay in the verification folder.

The model runs are smoke tests. A quality claim still requires a fixed seed BF16 and INT8 comparison with a recorded threshold.
