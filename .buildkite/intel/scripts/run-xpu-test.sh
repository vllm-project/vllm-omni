#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# This script build the XPU docker image and run the offline inference inside the container.
set -ex

omni_source_dir=$(git rev-parse --show-toplevel)

: "${VLLM_VERSION:?VLLM_VERSION must be set}"
# Official upstream XPU image. It is published from vllm's own
# docker/Dockerfile.xpu, which is exactly what build_vllm_base() clones and
# builds below, so pulling it is equivalent to building the fallback base.
upstream_base_image="vllm/vllm-openai-xpu:${VLLM_VERSION}"
local_base_image="xpu/vllm-omni-ci-base:${VLLM_VERSION}"
image_name="xpu/vllm-omni-ci:${BUILDKITE_COMMIT:?BUILDKITE_COMMIT must be set}"
container_name="xpu_${BUILDKITE_COMMIT}_$(
    tr -dc A-Za-z0-9 </dev/urandom | head -c 10
    echo
)"

cd "${omni_source_dir}"

# The XPU base image is ~37GB; the default gzip layer exporter is single-threaded
# and takes ~16min to compress it. zstd is multi-threaded (uses all cores) at a
# similar ratio; set EXPORT_COMPRESSION=uncompressed to skip compression entirely
# for local-only images. This requires the containerd image store (buildx docker
# driver), which is the default here.
EXPORT_COMPRESSION="${EXPORT_COMPRESSION:-zstd}"
if [ "${EXPORT_COMPRESSION}" = "uncompressed" ]; then
    export_args=(--output "type=image,name={{IMAGE}},compression=uncompressed")
else
    export_args=(--output "type=image,name={{IMAGE}},compression=${EXPORT_COMPRESSION},compression-level=3,force-compression=true")
fi

docker_build() {
    # $1 = image name; remaining args passed through to docker build.
    local image="$1"
    shift
    local out=("${export_args[@]/'{{IMAGE}}'/${image}}")
    docker build "${out[@]}" "$@" -f docker/Dockerfile.xpu .
}

build_vllm_base() (
    local vllm_source_dir
    vllm_source_dir=$(mktemp -d)
    trap 'rm -rf "${vllm_source_dir}"' EXIT

    git clone --depth 1 --branch "${VLLM_VERSION}" \
        https://github.com/vllm-project/vllm "${vllm_source_dir}"

    local out=("${export_args[@]/'{{IMAGE}}'/${base_image_name}}")
    docker build "${out[@]}" \
        --target vllm-openai \
        -f "${vllm_source_dir}/docker/Dockerfile.xpu" \
        "${vllm_source_dir}"
)

# Resolve the vLLM base image, in priority order:
#   1. the published upstream image for VLLM_VERSION, pulled fresh (so moving
#      tags like nightly are refreshed rather than served stale from disk);
#   2. a local copy of that same image, when the registry is unreachable;
#   3. VLLM_BASE, if that image is already on disk;
#   4. otherwise build VLLM_BASE from the matching vLLM source tag (~30min).
# VLLM_BASE names the fallback base and defaults to ${local_base_image}; set it
# to reuse or produce a base under a different tag.
resolve_vllm_base() {
    if docker pull "${upstream_base_image}"; then
        base_image_name="${upstream_base_image}"
        return
    fi

    # Registry unreachable, but a previous run may have left the image behind.
    if [ -n "$(docker images -q "${upstream_base_image}")" ]; then
        echo "WARNING: could not pull ${upstream_base_image}; using local copy" >&2
        base_image_name="${upstream_base_image}"
        return
    fi

    echo "WARNING: ${upstream_base_image} unavailable remotely and locally" >&2
    base_image_name="${VLLM_BASE:-${local_base_image}}"
    if [ -z "$(docker images -q "${base_image_name}")" ]; then
        echo "WARNING: building ${base_image_name} from vLLM ${VLLM_VERSION} source" >&2
        build_vllm_base
    fi
}

resolve_vllm_base

# Try building the docker image
docker_build "${image_name}" --build-arg "VLLM_BASE=${base_image_name}" --build-arg "VLLM_VERSION=${VLLM_VERSION}"

# Setup cleanup
remove_docker_container() {
    docker rm -f "${container_name}" || true
    docker image rm -f "${image_name}" || true
    docker system prune -f || true
}
trap remove_docker_container EXIT

HF_CACHE="${HF_CACHE:-$(realpath ~)/.cache/huggingface}"
mkdir -p "${HF_CACHE}"
HF_MOUNT="/root/.cache/huggingface"

# One hour of tests inside the container. The stage budgets below sum to less
# than this; timeout_in_minutes on the step has to cover this plus the image
# build (check the "time docker build" line above when tuning either).
XPU_TEST_BUDGET="${XPU_TEST_BUDGET:-60m}"

# pr    = core_model + the mxfp8 config check (~20m; the only weights it needs
#         are the tiny CI models, tiny-random/Qwen-Image and qwen_image_random).
# broad = also advanced_model and omni, which load full checkpoints (~20m more).
# Set XPU_TEST_SCOPE=broad on the nightly or a manually triggered build.
export XPU_TEST_SCOPE="${XPU_TEST_SCOPE:-pr}"

# Container variables below must expand in the container shell.
# shellcheck disable=SC2016
time timeout -k 30 "${XPU_TEST_BUDGET}" docker run \
    --device /dev/dri:/dev/dri \
    --net=host \
    --ipc=host \
    -v /dev/dri/by-path:/dev/dri/by-path \
    -v "${HF_CACHE}:${HF_MOUNT}" \
    --security-opt seccomp=unconfined \
    --entrypoint="" \
    -e VLLM_LOGGING_LEVEL \
    -e VLLM_OMNI_LOGGING_LEVEL \
    -e HF_TOKEN \
    -e HF_HUB_OFFLINE \
    -e XPU_TEST_SCOPE \
    -e XPU_CPU_WORKERS \
    -e ZE_AFFINITY_MASK \
    --name "${container_name}" \
    "${image_name}" \
    bash -c '
    set -uo pipefail
    pip install tblib==3.1.0 || exit 1
    cd /workspace/vllm-omni || exit 1

    # One card per job. The agent sets ZE_AFFINITY_MASK; keep only its first
    # entry so a cards_2+ test cannot take cards another agent on this host is
    # using - with a single visible device the hardware_test(num_cards>=2)
    # skipif fires instead. Every marker expression below also pins cards_1.
    mask="${ZE_AFFINITY_MASK:-0}"
    export ZE_AFFINITY_MASK="${mask%%,*}"
    echo "ZE_AFFINITY_MASK=${mask} -> using card ${ZE_AFFINITY_MASK}"

    # Paths that hold XPU-tagged tests. Scoped explicitly: a bare
    # "pytest tests/" still crashes during collection on XPU (#6175).
    paths=(tests/diffusion tests/dfx tests/e2e tests/platforms
           tests/model_executor tests/entrypoints tests/engine
           tests/worker tests/worker_v2)

    # Resolves a model directory at import time, so collection of the whole
    # path list fails (pytest exit 2) whenever those weights are absent from
    # the mounted cache. It carries no XPU marker, so skip the file outright.
    ignore=(--ignore tests/e2e/offline_inference/test_cosyvoice3_expansion.py)

    # core_model, but each case starts an engine on a real checkpoint: 507s for
    # 4 nodes and 142s for 1. They carry no signal the kernel tests do not, so
    # the blocking lane leaves them to the broad scope.
    pr_ignore=(--ignore tests/diffusion/batching
               --ignore tests/diffusion/cache/test_teacache.py)
    [ "${XPU_TEST_SCOPE:-pr}" = "broad" ] && pr_ignore=()

    rc=0
    # stage <name> <budget> <pytest args...>
    stage() {
        local name="$1" budget="$2"
        shift 2
        local t0=${SECONDS} code=0
        echo "--- :pytest: ${name}"
        timeout -k 30 "${budget}" pytest -v -s --durations=25 "$@" || code=$?
        # 5 = nothing collected for this marker set, which is not a failure.
        if [ "${code}" = 0 ] || [ "${code}" = 5 ]; then
            echo "+++ PASS ${name} ($((SECONDS - t0))s)"
        else
            echo "FAIL ${name} ($((SECONDS - t0))s, rc=${code})"
            rc=1
        fi
    }

    # Device-agnostic lane: cpu-marked tests, which take no card but do run
    # with current_platform == xpu. That is coverage no other CI lane has, and
    # it reaches the areas the card lane cannot afford - engine, config, the
    # scheduler, entrypoints, distributed. The skip list pins the cases that
    # need a real checkpoint or assert CUDA-specific behaviour (VLLMZ-3554).
    cpu_paths=(tests/platforms tests/config tests/core tests/utils tests/attention
               tests/engine tests/worker tests/worker_v2 tests/distributed
               tests/model_executor tests/diffusion)
    mapfile -t cpu_skip < <(grep -v "^#" .buildkite/intel/xpu-cpu-lane-skip.txt | grep .)
    cpu_ignore=()
    for f in "${cpu_skip[@]}"; do cpu_ignore+=(--ignore "${f}"); done
    stage "device-agnostic (no card)" 30m -n "${XPU_CPU_WORKERS:-8}" \
        "${cpu_paths[@]}" "${cpu_ignore[@]}" \
        -m "core_model and cpu and not (cards_2 or cards_3 or cards_4 or cards_5 or cards_6 or cards_7 or cards_8)"

    # core_model excludes omni: the async-omni tests carry both markers and
    # would otherwise run twice, and they need their own process (broad scope).
    stage "core_model" 30m "${paths[@]}" "${ignore[@]}" "${pr_ignore[@]}" \
        -m "core_model and not omni and xpu and B60 and cards_1" --run-level core_model
    stage "mxfp8 config" 2m tests/diffusion/quantization/test_mxfp8_config.py

    # Everything past this point loads a full checkpoint (Qwen3-TTS, Z-Image,
    # Wan2.1, Voxtral) from the mounted HF cache and costs about 20 minutes, so
    # it is off on PRs and runs on the nightly or a manually triggered build
    # (XPU_TEST_SCOPE=broad).
    if [ "${XPU_TEST_SCOPE:-pr}" = "broad" ]; then
        # full_model stays out until the cache is seeded: those cases pull tens
        # of GB of weights.
        stage "advanced_model" 15m "${paths[@]}" "${ignore[@]}" \
            -m "advanced_model and not full_model and xpu and B60 and cards_1" --run-level core_model
        # The async-omni engine leaves worker VRAM mapped in the pytest parent,
        # so a 16 GiB thinker started after the other tests in the same process
        # fails to initialise. CUDA splits this test out for the same reason;
        # keep it last and alone.
        stage "omni" 10m "${paths[@]}" "${ignore[@]}" \
            -m "omni and xpu and B60 and cards_1" --run-level core_model
    else
        echo "--- :pytest: skipping advanced_model and omni (XPU_TEST_SCOPE=pr)"
    fi

    exit "${rc}"
'
