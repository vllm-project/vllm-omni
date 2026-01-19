# Benchmark Instructions

This directory contains two types of benchmarks to evaluate the performance of the Modality-Aware Scheduler.

## 1. `e2e_benchmark` (Production-Level End-to-End Test)

This is a production-level, end-to-End (E2E) benchmark designed to simulate real-world scenarios. It deploys the following components on Kubernetes:
* Two separate server pods (with identical hardware and driver versions).
* One client pod.
* A shared SSD PVC mounted by all three pods to store multimodal datasets.

**Design Philosophy:**
The client constructs requests using local file paths residing on the shared PVC and communicates via ClusterIP. This architecture minimizes the latency uncertainty caused by public network transmissions, ensuring that the request arrival time difference between the two servers is within one millisecond. Furthermore, it ensures that I/O operations for loading multimodal data do not become a bottleneck.

**Current Performance Note:**
Currently, the system uses the standard HuggingFace processor for preprocessing, which is CPU-intensive and acts as the primary bottleneck for system throughput in this alpha stage. 
* **Result:** Under `qps=20, max_token=10`, both the `ModalityAwareScheduler` and the default scheduler show similar throughput (~16.24 tokens/s), as the scheduler's potential is masked by the preprocessing latency.

### How to Run

**Step 1: Prepare Data**

Download the necessary datasets:

#### For users in mainland China:
    python3 vllm-omni/benchmarks/modality_aware_test/e2e_benchmark/prepare_data.py \
    --data_dir /root/data/datasets \
    --use_mainland_hf_mirror

#### For users in other regions:
    python3 vllm-omni/benchmarks/modality_aware_test/e2e_benchmark/prepare_data.py \
    --data_dir /root/data/datasets

**Step 2: Pre-process Media**

To prevent OOM (Out of Memory) errors on recommended hardware, resize raw images/videos and truncate video data:

    python3 vllm-omni/benchmarks/modality_aware_test/e2e_benchmark/resize_image_video.py \
    --data_dir /root/data/datasets

    python3 vllm-omni/benchmarks/modality_aware_test/e2e_benchmark/truncate_video.py \
    --data_dir /root/data/datasets

**Step 3: Cluster Configuration**

Claim an SSD PVC in your cloud cluster and upload the entire project folder to it.

Keep a local copy of the following files:

    e2e_benchmark/dual_server_start.yaml
    e2e_benchmark/tester_start.yaml
    e2e_benchmark/run_benchmark.sh

**Step 4: Configure the Script **

Open run_benchmark.sh and modify the [USER CONFIGURATION SECTION] to match your Kubernetes environment:

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
    PVC_CLAIME_NAME="omnitest"
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


**Step 5: Deploy and Test**

Run the script locally to initiate deployment and the E2E test:

    bash run_benchmark.sh


## 2. micro_benchmark_bypass_cpu_bottleneck (Component Test)

This is a specialized component test designed to verify the core engine performance by bypassing the CPU bottleneck caused by the HuggingFace preprocessor. It constructs a tensor dictionary (simulating the output of the processor) as a dummy request and passes it directly to the engine core.

Performance Gains: This benchmark generates plots for avg_ttft, p99_ttft, and throughput (located in this directory).

Result: At qps=20, max_token=10, the ModalityAwareScheduler achieves 44 tokens/s, compared to 30 tokens/s for the default scheduler.

Improvement: This demonstrates a ~46% increase in throughput when the CPU bottleneck is removed.

    ⚠️ Note on Compatibility: This test was developed to validate the scheduler's theoretical limits. Due to recent upstream changes in the HuggingFace preprocessing logic, the dummy data generation in this script is currently out of sync with the latest codebase.

    This script is provided for reference and logic verification purposes. Developers wishing to run this specific micro-benchmark may need to manually update the dummy data bypass logic to align with the current multimodal preprocessing steps.