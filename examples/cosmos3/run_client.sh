#!/usr/bin/env bash
# Launch ONLY the baseline client; never start/stop a server or another profile.
set -euo pipefail
export COSMOS3_RUN_STARTED_AT="${COSMOS3_RUN_STARTED_AT:-$(date +%s.%N)}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python="${COSMOS3_CLIENT_PYTHON:-${VLLM_PYTHON:-$root/.venv/bin/python}}"
export PYTHONPATH="$root/plugins/cosmos3_client${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}" no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
# Huawei default: remote draws on 168; CPU mode uses local OSMesa.
# GPU users can explicitly select COSMOS3_RENDER=egl.
default_render=egl
if [[ -n "${REMOTE_RENDER_URL:-}" || "${COSMOS3_DEVICE:-${VLLM_TARGET_DEVICE:-cuda}}" == npu || "${COSMOS3_DEVICE:-}" == ascend ]]; then
  default_render=remote
fi
render="${COSMOS3_RENDER:-$default_render}"
case "$render" in
  remote|cpu)
    export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBGL_ALWAYS_SOFTWARE=1
    export GALLIUM_DRIVER=llvmpipe LP_NUM_THREADS="${LP_NUM_THREADS:-2}"
    render_device=-1
    if [[ "$render" == remote ]]; then
      export REMOTE_RENDER_URL="${REMOTE_RENDER_URL:-http://127.0.0.1:19960}"
    else
      unset REMOTE_RENDER_URL
    fi
    ;;
  egl)
    unset REMOTE_RENDER_URL LIBGL_ALWAYS_SOFTWARE
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-$root/plugins/cosmos3_client/vllm_omni_cosmos3_client/egl_vendor.json}"
    render_device="${MUJOCO_EGL_DEVICE_ID:-0}"
    ;;
  *) echo "COSMOS3_RENDER must be remote, cpu or egl" >&2; exit 2 ;;
esac
export EVAL_OUT="${EVAL_OUT:-$root/outputs/baseline_$(date -u +%Y%m%dT%H%M%SZ)}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$EVAL_OUT/mpl_cache}"
cd "$root"
exec "$python" -u -m vllm_omni_cosmos3_client \
  --server_url "${COSMOS3_SERVER_URL:-http://127.0.0.1:8101}" \
  --task_suite libero_10 --num_trials_per_task 1 --num_envs 1 --seed 0 \
  --image_size 256 --env_image_size 256 --action_horizon 16 \
  --camera agentview,wrist --action_space frame_wise_relative \
  --rotation_space 6d --action_dim 10 --gripper_mode zero_one \
  --mujoco_gl "$MUJOCO_GL" --render_gpu_device_id "$render_device" \
  --timeout 1800 --wait_timeout 1800 --output_dir "$EVAL_OUT"  "$@"
