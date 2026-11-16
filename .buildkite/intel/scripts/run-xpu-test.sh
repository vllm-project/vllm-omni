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

# Test budget inside the container; the pipeline gives the broad scope more.
# Each stage also has its own cap, summing to 59m in the pr scope and 94m in the
# broad scope, so one runaway stage cannot eat the whole budget. Measured end to
# end in the pr scope on two cards: 3m40s dispatch + 10m05s omni + 23m01s core
# 1-card + 23s mxfp8 + 1m41s core 2-card = 39m31s including the reaps.
# timeout_in_minutes on the step has to cover the budget plus the image build
# (check the "time docker build" line above when tuning either).
XPU_TEST_BUDGET="${XPU_TEST_BUDGET:-60m}"

# pr    = the no-card XPU-dispatch stage, every core_model case on one and on
#         two cards, and the omni models. Needs tiny-random/Qwen-Image (30MB),
#         riverclouds/qwen_image_random (9.9GB), Qwen2.5-Omni-3B (12GB) and
#         Qwen2.5-Omni-7B (21GB) in the mounted HF cache.
# broad = also advanced_model on one card (Qwen3-TTS, Z-Image, Wan2.1, Voxtral,
#         ~13m) and on two (11 sleep/wake cases, which additionally need
#         BAGEL-7B-MoT).
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

    # Card budget. The agent sets ZE_AFFINITY_MASK; take its first entry for the
    # cards_1 stages and its first two for the cards_2 ones, so a test can never
    # reach a card another agent on this host is using - with only the allotted
    # devices visible, hardware_test(num_cards=N) skips itself when N is higher.
    # Every marker expression below also pins cards_1 or cards_2 explicitly.
    mask="${ZE_AFFINITY_MASK:-0}"
    IFS="," read -r -a mask_cards <<< "${mask}"
    one_card="${mask_cards[0]}"
    two_cards="${mask_cards[0]}${mask_cards[1]+,${mask_cards[1]}}"
    echo "ZE_AFFINITY_MASK=${mask} -> cards_1 on ${one_card}, cards_2 on ${two_cards}"

    # How many cards this lane is entitled to. The pipeline asks for 2 so the
    # cards_2 stages always run; an agent that hands over fewer fails the step
    # instead of quietly dropping those stages. Set XPU_REQUIRE_CARDS=1 on a
    # single-card queue to opt out of the 2-card stages on purpose.
    require_cards="${XPU_REQUIRE_CARDS:-2}"
    # Which card classes this step owns. A dedicated 2-card step sets
    # XPU_TEST_CARDS=2 so it does not repeat the 1-card lane that the
    # single-card agents already run; "1,2" (the default) runs everything.
    test_cards="${XPU_TEST_CARDS:-1,2}"
    case ",${test_cards}," in *,1,*) run_1card=1 ;; *) run_1card=0 ;; esac
    case ",${test_cards}," in *,2,*) run_2card=1 ;; *) run_2card=0 ;; esac
    echo "XPU_TEST_CARDS=${test_cards} -> 1-card stages=${run_1card}, 2-card stages=${run_2card}"
    if [ "${#mask_cards[@]}" -lt "${require_cards}" ]; then
        echo "ERROR: this step wants ${require_cards} cards but the agent gave" \
             "${#mask_cards[@]} (ZE_AFFINITY_MASK=${mask}). Point the step at an" \
             "agent with ${require_cards} cards, or set XPU_REQUIRE_CARDS to" \
             "${#mask_cards[@]} to run the smaller lane deliberately." >&2
        exit 1
    fi

    # Paths that hold XPU-tagged tests. Scoped explicitly: a bare
    # "pytest tests/" still crashes during collection on XPU (#6175).
    paths=(tests/diffusion tests/dfx tests/e2e tests/platforms
           tests/model_executor tests/entrypoints tests/engine
           tests/worker tests/worker_v2)

    # Resolves a model directory at import time, so collection of the whole
    # path list fails (pytest exit 2) whenever those weights are absent from
    # the mounted cache. It carries no XPU marker, so skip the file outright.
    ignore=(--ignore tests/e2e/offline_inference/test_cosyvoice3_expansion.py)


    # The GPU stages load these from the mounted cache. Resolving them up front
    # turns "LocalEntryNotFoundError six minutes into the omni stage" into one
    # clear message naming what the agent is missing.
    if [ "${run_1card}" = 1 ]; then
        python - <<PY || exit 1
from huggingface_hub import snapshot_download

needed = ["tiny-random/Qwen-Image", "riverclouds/qwen_image_random",
          "Qwen/Qwen2.5-Omni-3B", "Qwen/Qwen2.5-Omni-7B"]
missing = []
for repo in needed:
    try:
        snapshot_download(repo, local_files_only=True)
    except Exception as exc:
        missing.append(f"{repo}: {type(exc).__name__}")
if missing:
    raise SystemExit(
        "these checkpoints are not in the mounted HF cache:\n  "
        + "\n  ".join(missing)
        + "\nSeed them on the agent cache that HF_CACHE points at."
    )
print("checkpoint preflight ok:", ", ".join(needed))
PY
    fi

    rc=0
    # A stage that starts an engine can leave a worker process behind, and its
    # device memory with it: a 19 GiB leftover once left 11 of 30 GiB free and
    # failed the next stage at startup. Reap our own strays between stages and
    # report what the card holds, so a leak shows up in the log instead of as a
    # mysterious memory-profiling failure.
    reap() {
        pkill -f StageEngineCoreProc || true
        pkill -f "VLLM::EngineCore" || true
        pkill -f "vllm_omni.*worker" || true
        sleep 5
        ZE_AFFINITY_MASK="$1" python - <<PY || true
import torch
free, total = torch.xpu.mem_get_info()
print(f"[card {torch.xpu.current_device()}] {free / 2**30:.2f} of {total / 2**30:.2f} GiB free")
PY
    }

    rc=0
    # stage <name> <budget> <cards> <pytest args...>; cards is the
    # ZE_AFFINITY_MASK the stage runs under, or "" for the no-card stage.
    stage() {
        local name="$1" budget="$2" cards="$3"
        shift 3
        local t0=${SECONDS} code=0
        echo "--- :pytest: ${name}"
        ZE_AFFINITY_MASK="${cards}" timeout -k 30 "${budget}" \
            pytest -v -s --durations=25 "$@" || code=$?
        # 5 = nothing collected for this marker set, which is not a failure.
        if [ "${code}" = 0 ] || [ "${code}" = 5 ]; then
            echo "+++ PASS ${name} ($((SECONDS - t0))s)"
        else
            echo "FAIL ${name} ($((SECONDS - t0))s, rc=${code})"
            rc=1
        fi
        [ -n "${cards}" ] && reap "${cards}"
    }

    # No-card stage: the cpu-marked tests that can observe which platform they
    # run on, so they exercise XPU dispatch without taking a card. The file list
    # is pinned rather than globbed - the rest of the cpu suite is platform
    # independent and the CUDA and ROCm lanes already run it. Note --ignore does
    # not apply to files named explicitly, which is why the exclusions live in
    # the list itself. See the header of both files for the measurements.
    if [ "${run_1card}" = 1 ]; then
        mapfile -t xpu_path < <(grep -v "^#" .buildkite/intel/xpu-path-tests.txt | grep .)
        stage "xpu dispatch (no card)" 6m "${one_card}" -n "${XPU_CPU_WORKERS:-8}" \
            "${xpu_path[@]}" \
            -m "core_model and cpu and not (cards_2 or cards_3 or cards_4 or cards_5 or cards_6 or cards_7 or cards_8)"

        # The omni models go first and in their own process. They want the most
        # device memory of anything here - Qwen2.5-Omni colocates three stages
        # on one card at utils 0.45/0.25/0.15 - so they run on a clean card.
        stage "omni · 1 card" 15m "${one_card}" "${paths[@]}" "${ignore[@]}" \
            -m "omni and xpu and B60 and cards_1" --run-level core_model

        # core_model excludes omni: the async-omni tests carry both markers and
        # would otherwise run twice.
        stage "core_model · 1 card" 28m "${one_card}" "${paths[@]}" "${ignore[@]}" \
            -m "core_model and not omni and xpu and B60 and cards_1" --run-level core_model
        stage "mxfp8 config" 2m "${one_card}" tests/diffusion/quantization/test_mxfp8_config.py

        # advanced_model loads the larger checkpoints (Qwen3-TTS, Z-Image,
        # Wan2.1, Voxtral), so it waits for the nightly or a manual build.
        # full_model stays out of both until the cache is seeded.
        if [ "${XPU_TEST_SCOPE:-pr}" = "broad" ]; then
            stage "advanced_model · 1 card" 15m "${one_card}" "${paths[@]}" "${ignore[@]}" \
                -m "advanced_model and not full_model and xpu and B60 and cards_1" --run-level core_model
        fi
    fi

    # Two cards: distributed init over XCCL plus the step pipeline in each
    # parallel mode. The SKU set is wider here because the only 2-card XPU
    # suites in-tree are marked B70 (sleep mode), and a B70 agent satisfies the
    # B60-marked cases too.
    if [ "${run_2card}" = 1 ]; then
        stage "core_model · 2 cards" 8m "${two_cards}" "${paths[@]}" "${ignore[@]}" \
            -m "core_model and xpu and (B60 or B70) and cards_2" --run-level core_model

        # 11 sleep/wake cases. Not measured yet: they additionally need
        # BAGEL-7B-MoT in the mounted cache.
        if [ "${XPU_TEST_SCOPE:-pr}" = "broad" ]; then
            stage "advanced_model · 2 cards" 20m "${two_cards}" "${paths[@]}" "${ignore[@]}" \
                -m "advanced_model and not full_model and xpu and (B60 or B70) and cards_2" --run-level core_model
        fi
    fi

    exit "${rc}"
'
