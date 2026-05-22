#!/bin/bash
if nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader 2>/dev/null | grep -q .; then
    echo "ERROR: GPUs are occupied by the following processes:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    exit 1
fi

export PYTHONPATH="/app/vllm-omni${PYTHONPATH:+:$PYTHONPATH}"

# source .venv/bin/activate
# export TORCH_COMPILE_DISABLE=1
# export VLLM_DISABLE_COMPILE_CACHE=1
# export CUDA_VISIBLE_DEVICES=0,1,4,5
PP_SIZE="${1:-2}"
CFG_SIZE="${2:-1}"
OUTPUT_SUFFIX=""
if [[ "$*" == *"--enable-pipefusion"* ]]; then
    OUTPUT_SUFFIX="_pf"
fi
if [[ "$*" == *"--vae-patch-parallel-size"* ]]; then
    VAE_SIZE=$(echo "$*" | grep -oP '(?<=--vae-patch-parallel-size=)[0-9]+')
    OUTPUT_SUFFIX="${OUTPUT_SUFFIX}_vae${VAE_SIZE}"
fi
if [[ "$*" == *"--ulysses-degree"* ]]; then
    ULYSSES_SIZE=$(echo "$*" | grep -oP '(?<=--ulysses-degree=)[0-9]+')
    OUTPUT_SUFFIX="${OUTPUT_SUFFIX}_sp${ULYSSES_SIZE}"
fi
if [[ "$*" == *"--tensor-parallel-size"* ]]; then
    TP_SIZE=$(echo "$*" | grep -oP '(?<=--tensor-parallel-size=)[0-9]+')
    OUTPUT_SUFFIX="${OUTPUT_SUFFIX}_tp${TP_SIZE}"
fi
python examples/offline_inference/text_to_video/text_to_video.py --model=Wan-AI/Wan2.2-TI2V-5B-Diffusers --width=1280 --height=704 --guidance-scale=5.0 --prompt="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --output=results/5B/t2v_5B_pp"${PP_SIZE}"_cfg"${CFG_SIZE}"${OUTPUT_SUFFIX}.mp4 --pipeline-parallel-size="${PP_SIZE}" --cfg-parallel-size="${CFG_SIZE}" "${@:3}"
