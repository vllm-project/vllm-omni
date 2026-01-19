#!/bin/bash

# ==============================================================================
# [USER CONFIGURATION SECTION]
# Please modify the variables below to match your Kubernetes environment.
# ==============================================================================
# TODO: change the yaml path to your own local path
DUAL_SERVER_YAML="./dual_server_start.yaml"
TESTER_YAML="./tester_start.yaml"
LOCAL_OUTPUT_DIR="./results"

# TODO: PVC claim name & mount path in your cluster
# it is recommended to mount the pvc to the same path for both servers & client
# make sure it is a ssd pvc.
PVC_CLAIM_NAME="omnitest"
PVC_DIR="/root/data"

# TODO: change the vllm code & data dir to your remote cluster path
VLLM_OMNI_PATH="${PVC_DIR}/vllm-omni"
HF_CACHE_PATH="${PVC_DIR}/hf_cache"
DATA_DIR="${PVC_DIR}/datasets"
OUTPUT_DIR="${PVC_DIR}/benchmark_results"

# TODO: images: because we rollout for a lot of time in this experiment,
# it is recommended to download the image from docker hub to a local registry
SERVER_IMAGE_NAME="vllm/vllm-omni:v0.12.0rc1"
CLIENT_IMAGE_NAME="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"


# TODO: GPU labels & driver versions in your cluster
# it is important to use the same hardware & driver for production level e2e test
GPU_LABEL_KEY="kubernetes.io/hostname"
GPU_LABEL="<YOUR_GPU_NODE_HOSTNAME>"
GPU_DRIVER_KEY="nvidia.com/cuda.driver.major"
GPU_DRIVER="<YOUR_DRIVER_VERSION>"

# TODO: in some tricky k8s cluster, it is recommended to specify a python path.
# you can just use "python3" in most cases
CLIENT_PYTHON_CMD="python3"

# ==============================================================================
# [END OF USER CONFIGURATION]
# Do not modify the script below unless you know what you are doing.
# ==============================================================================

OUTPUT_NAME="benchmark_results"
LOG_NAME="error_packets"

BASELINE_DEPLOYMENT="vllm-baseline-backend"
OPTIMIZED_DEPLOYMENT="vllm-opt-backend"
CLIENT_DEPLOYMENT="vllm-benchmark-client"
CLIENT_LABEL="app=vllm-benchmark-client"

export VLLM_OMNI_PATH
export HF_CACHE_PATH

export PVC_DIR
export PVC_CLAIM_NAME

export SERVER_IMAGE_NAME
export CLIENT_IMAGE_NAME

export GPU_LABEL
export GPU_DRIVER
export GPU_LABEL_KEY
export GPU_DRIVER_KEY

export CLIENT_PYTHON_CMD

get_client_pod() {
    kubectl get pod -l $CLIENT_LABEL --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null
}

exec_with_retry() {
    local cmd="$1"
    local max_retries=10
    local retry_count=0
    local sleep_sec=10

    while [ $retry_count -lt $max_retries ]; do
        CURRENT_POD=$(get_client_pod)

        if [ -z "$CURRENT_POD" ]; then
            echo "Client pod not found or not running. Waiting ${sleep_sec}s..."
        else

            echo "Executing on $CURRENT_POD: $cmd"
            if kubectl exec "$CURRENT_POD" -- bash -c "$cmd"; then
                return 0
            else
                EXIT_CODE=$?
                echo " Command failed with exit code $EXIT_CODE."
                if [ $EXIT_CODE -eq 137 ]; then
                    echo "💥 Detected OOM Kill (137). Waiting for pod restart..."
                fi
            fi
        fi

        ((retry_count++))
        echo " Retry $retry_count/$max_retries in ${sleep_sec}s..."
        sleep $sleep_sec

        echo " Waiting for pod to be ready..."
        kubectl wait --for=condition=ready pod -l $CLIENT_LABEL --timeout=300s >/dev/null 2>&1
    done

    echo " Failed to execute command after $max_retries retries."
    return 1
}

cleanup() {
    echo ""
    echo " [Trap Triggered] Script is exiting..."

    FINAL_POD=$(get_client_pod)
    if [ ! -z "$FINAL_POD" ]; then
        echo " Attempting to save current CSV results from $FINAL_POD before deleting..."
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        kubectl cp $FINAL_POD:${OUTPUT_DIR}/${OUTPUT_NAME}.csv ${LOCAL_OUTPUT_DIR}/${OUTPUT_NAME}_interrupted_${TIMESTAMP}.csv >/dev/null 2>&1 || true
    else
        echo " Could not find a running pod to save interrupted results."
    fi

    echo " Cleaning up Kubernetes resources (Releasing GPUs)..."
    envsubst < $DUAL_SERVER_YAML | kubectl delete -f - --wait=false >/dev/null 2>&1 || true
    kubectl delete -f $TESTER_YAML --wait=false >/dev/null 2>&1 || true

    echo " Cleanup complete. Resources deleted."
}

trap cleanup EXIT INT TERM

echo "======= [step 1] deploy and initialize the environment ======="

envsubst '${VLLM_OMNI_PATH} ${HF_CACHE_PATH} ${PVC_DIR} ${SERVER_IMAGE_NAME} ${GPU_LABEL} ${GPU_DRIVER} ${GPU_LABEL_KEY} ${GPU_DRIVER_KEY} ${PVC_CLAIM_NAME}' < $DUAL_SERVER_YAML | kubectl apply -f -

envsubst '${CLIENT_PYTHON_CMD} ${CLIENT_IMAGE_NAME} ${PVC_DIR} ${PVC_CLAIM_NAME}' < $TESTER_YAML | kubectl apply -f -



echo "wait for the client pod to be ready..."
kubectl wait --for=condition=ready pod -l $CLIENT_LABEL --timeout=600s

echo "Wait for dependencies to be installed inside the pod..."
exec_with_retry "while [ ! -f /tmp/ready ]; do sleep 1; done; echo 'Dependencies ready'"

echo " Running pre-flight python environment check..."
PROBE_CMD="$CLIENT_PYTHON_CMD -c 'import aiohttp; import numpy; import argparse; print(\" Python Environment is GOOD\")'"

if ! exec_with_retry "$PROBE_CMD"; then
    echo " FATAL ERROR: Python environment check failed!"
    echo "   Please check the pip install step above."
    echo "   Exiting early to save time."
    exit 1
fi
echo " Pre-flight check passed. Proceeding..."


exec_with_retry "mkdir -p ${OUTPUT_DIR}"


exec_with_retry "echo 'Scenario,QPS,Baseline_TPS,Optimized_TPS,Baseline_ErrorRate,Optimized_ErrorRate' > ${OUTPUT_DIR}/${OUTPUT_NAME}.csv"

echo "the client is ready to run the benchmark..."


echo "======= [step 2] wait for the servers to be ready ======="

# pulling image for the first time might use a lot of time
INIT_TIMEOUT=3600s

echo " Waiting for deployment rollout (timeout: $INIT_TIMEOUT)..."


kubectl rollout status deployment/$BASELINE_DEPLOYMENT --timeout=$INIT_TIMEOUT &
PID_BASE=$!

kubectl rollout status deployment/$OPTIMIZED_DEPLOYMENT --timeout=$INIT_TIMEOUT &
PID_OPT=$!


wait $PID_BASE
STATUS_BASE=$?
wait $PID_OPT
STATUS_OPT=$?

if [ $STATUS_BASE -ne 0 ] || [ $STATUS_OPT -ne 0 ]; then
    echo " Error: Initial deployment failed or timed out. Images might still be pulling."
    echo "   Please check pod events with: kubectl describe pod -l app=$BASELINE_DEPLOYMENT"
    exit 1
fi

echo " Server is ready! Starting benchmark..."


echo "======= [step 3] run the actual benchmark ======="

TEST_CONFIGS=(
    "5:20"
    "10:10"
    "20:5"
    "50:2"
)
SCENARIO_NAMES=("MIXED")
SCENARIO_WEIGHTS=(
    '{"image":0.2,"audio":0.2,"video":0.1,"text":0.5}'
)

FIRST_RUN=true

for config in "${TEST_CONFIGS[@]}"; do
    qps=${config%%:*}
    duration=${config##*:}

    for i in "${!SCENARIO_NAMES[@]}"; do
        NAME=${SCENARIO_NAMES[$i]}
        WEIGHTS=${SCENARIO_WEIGHTS[$i]}

        echo "================================================"
        echo "  TESTING: SCENARIO=$NAME | QPS=$qps"
        echo "================================================"

        if [ "$FIRST_RUN" = true ]; then
            echo "⚡ First run detected. Skipping restart, using fresh deployment."
            FIRST_RUN=false
        else
            echo "🧹 Hard restarting server to release GPU memory and clean KV cache..."

            kubectl scale deployment/$BASELINE_DEPLOYMENT --replicas=0
            kubectl scale deployment/$OPTIMIZED_DEPLOYMENT --replicas=0

            echo "   Waiting for pods to terminate..."
            kubectl wait --for=delete pod -l app=$BASELINE_DEPLOYMENT --timeout=120s >/dev/null 2>&1
            kubectl wait --for=delete pod -l app=$OPTIMIZED_DEPLOYMENT --timeout=120s >/dev/null 2>&1


            kubectl scale deployment/$BASELINE_DEPLOYMENT --replicas=1
            kubectl scale deployment/$OPTIMIZED_DEPLOYMENT --replicas=1

            echo "   Waiting for server to be ready..."

            kubectl wait --for=condition=ready pod -l app=$BASELINE_DEPLOYMENT --timeout=1000s &
            PID_BASE=$!

            kubectl wait --for=condition=ready pod -l app=$OPTIMIZED_DEPLOYMENT --timeout=1000s &
            PID_OPT=$!

            wait $PID_BASE
            wait $PID_OPT

            sleep 15
        fi

        echo " run the benchmark script..."

        BENCHMARK_CMD="$CLIENT_PYTHON_CMD ${VLLM_OMNI_PATH}/benchmarks/modality_aware_test/e2e_benchmark/modality_aware_benchmark.py \
            --scenario '$NAME' \
            --qps '$qps' \
            --duration '$duration' \
            --weights '$WEIGHTS' \
            --data_dir '$DATA_DIR' \
            --output_dir '${OUTPUT_DIR}' \
            --csv_file '${OUTPUT_NAME}.csv' \
            --log_file '${LOG_NAME}.log'"


        exec_with_retry "$BENCHMARK_CMD" || { echo "Benchmark failed after max retries"; exit 1; }


        echo " the benchmark is completed."
    done
done

echo "======= [step 4] clean up and save the results ======="

FINAL_POD=$(get_client_pod)

if [ -n "$FINAL_POD" ]; then
    OUTPUT_FILE="${LOCAL_OUTPUT_DIR}/${OUTPUT_NAME}_$(date +%Y%m%d_%H%M).csv"
    OUTPUT_LOG="${LOCAL_OUTPUT_DIR}/${LOG_NAME}_$(date +%Y%m%d_%H%M).log"
    echo "Saving results from $FINAL_POD to $OUTPUT_FILE and $OUTPUT_LOG..."
    kubectl cp $FINAL_POD:${OUTPUT_DIR}/${OUTPUT_NAME}.csv $OUTPUT_FILE
    kubectl cp $FINAL_POD:${OUTPUT_DIR}/${LOG_NAME}.log $OUTPUT_LOG
else
    echo "Error: Could not find a running client pod to copy results from."
fi

echo " All benchmarks are completed."
