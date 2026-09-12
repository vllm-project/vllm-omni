# Wan2.2 TI2V 5B RTX 5090

## Purpose

---

Add RTX 5090 validation notes to the existing Wan-AI recipe.

This document contains the test environment, commands, and test results for Wan2.2-TI2V-5B on 1x NVIDIA GeForce RTX 5090.

Contributes to [#2645](https://github.com/vllm-project/vllm-omni/issues/2645).
## Environment

---
```bash
python collect-env.py
```

```bash
==============================
        System Info
==============================
OS                           : Ubuntu 24.04.4 LTS (x86_64)
GCC version                  : (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Clang version                : Could not collect
CMake version                : version 3.28.3
Libc version                 : glibc-2.39

==============================
       PyTorch Info
==============================
PyTorch version              : 2.11.0+cu130
Is debug build               : False
CUDA used to build PyTorch   : 13.0
ROCM used to build PyTorch   : N/A

==============================
      Python Environment
==============================
Python version               : 3.12.13 | packaged by conda-forge | (main, Mar  5 2026, 16:50:00) [GCC 14.3.0] (64-bit runtime)
Python platform              : Linux-5.15.0-181-generic-x86_64-with-glibc2.39

==============================
       CUDA / GPU Info
==============================
Is CUDA available            : True
CUDA runtime version         : 13.0.88
CUDA_MODULE_LOADING set to   :
GPU models and configuration : GPU 0: NVIDIA GeForce RTX 5090
Nvidia driver version        : 580.95.05
cuDNN version                : Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.9.14.0
/usr/lib/x86_64-linux-gnu/libcudnn_adv.so.9.14.0
/usr/lib/x86_64-linux-gnu/libcudnn_cnn.so.9.14.0
/usr/lib/x86_64-linux-gnu/libcudnn_engines_precompiled.so.9.14.0
/usr/lib/x86_64-linux-gnu/libcudnn_engines_runtime_compiled.so.9.14.0
/usr/lib/x86_64-linux-gnu/libcudnn_graph.so.9.14.0
/usr/lib/x86_64-linux-gnu/libcudnn_heuristic.so.9.14.0
/usr/lib/x86_64-linux-gnu/libcudnn_ops.so.9.14.0
HIP runtime version          : N/A
MIOpen runtime version       : N/A
Is XNNPACK available         : True

==============================
          CPU Info
==============================
Architecture:                            x86_64
CPU op-mode(s):                          32-bit, 64-bit
Address sizes:                           43 bits physical, 48 bits virtual
Byte Order:                              Little Endian
CPU(s):                                  64
On-line CPU(s) list:                     0-63
Vendor ID:                               AuthenticAMD
Model name:                              AMD EPYC 7302 16-Core Processor
CPU family:                              23
Model:                                   49
Thread(s) per core:                      2
Core(s) per socket:                      16
Socket(s):                               2
Stepping:                                0
Frequency boost:                         enabled
CPU(s) scaling MHz:                      50%
CPU max MHz:                             3000.0000
CPU min MHz:                             1500.0000
BogoMIPS:                                5999.91
Flags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba ibrs ibpb stibp vmmcall fsgsbase bmi1 avx2 smep bmi2 cqm rdt_a rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local clzero irperf xsaveerptr rdpru wbnoinvd arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif v_spec_ctrl umip rdpid overflow_recov succor smca sev sev_es ibpb_exit_to_user
Virtualization:                          AMD-V
L1d cache:                               1 MiB (32 instances)
L1i cache:                               1 MiB (32 instances)
L2 cache:                                16 MiB (32 instances)
L3 cache:                                256 MiB (16 instances)
NUMA node(s):                            2
NUMA node0 CPU(s):                       0-15,32-47
NUMA node1 CPU(s):                       16-31,48-63
Vulnerability Gather data sampling:      Not affected
Vulnerability Indirect target selection: Not affected
Vulnerability Itlb multihit:             Not affected
Vulnerability L1tf:                      Not affected
Vulnerability Mds:                       Not affected
Vulnerability Meltdown:                  Not affected
Vulnerability Mmio stale data:           Not affected
Vulnerability Reg file data sampling:    Not affected
Vulnerability Retbleed:                  Mitigation; untrained return thunk; SMT enabled with STIBP protection
Vulnerability Spec rstack overflow:      Mitigation; safe RET
Vulnerability Spec store bypass:         Mitigation; Speculative Store Bypass disabled via prctl and seccomp
Vulnerability Spectre v1:                Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:                Mitigation; Retpolines; IBPB conditional; STIBP always-on; RSB filling; PBRSB-eIBRS Not affected; BHI Not affected
Vulnerability Srbds:                     Not affected
Vulnerability Tsa:                       Not affected
Vulnerability Tsx async abort:           Not affected
Vulnerability Vmscape:                   Mitigation; IBPB before exit to userspace

==============================
Versions of relevant libraries
==============================
[pip3] flashinfer-python==0.6.12
[pip3] numpy==2.3.5
[pip3] nvidia-cublas==13.1.0.3
[pip3] nvidia-cuda-cccl==13.3.3.4.1
[pip3] nvidia-cuda-crt==13.3.73
[pip3] nvidia-cuda-cupti==13.0.85
[pip3] nvidia-cuda-nvcc==13.2.78
[pip3] nvidia-cuda-nvrtc==13.0.88
[pip3] nvidia-cuda-runtime==13.0.96
[pip3] nvidia-cuda-tileiras==13.2.78
[pip3] nvidia-cudnn-cu13==9.19.0.56
[pip3] nvidia-cudnn-frontend==1.26.0
[pip3] nvidia-cufft==12.0.0.61
[pip3] nvidia-cufile==1.15.1.6
[pip3] nvidia-curand==10.4.0.35
[pip3] nvidia-cusolver==12.0.4.66
[pip3] nvidia-cusparse==12.6.3.3
[pip3] nvidia-cusparselt-cu13==0.8.0
[pip3] nvidia-cutlass-dsl==4.5.2
[pip3] nvidia-cutlass-dsl-libs-base==4.5.2
[pip3] nvidia-cutlass-dsl-libs-cu13==4.5.2
[pip3] nvidia-ml-py==13.610.43
[pip3] nvidia-nccl-cu13==2.28.9
[pip3] nvidia-nvjitlink==13.0.88
[pip3] nvidia-nvshmem-cu13==3.4.5
[pip3] nvidia-nvtx==13.0.85
[pip3] nvidia-nvvm==13.2.78
[pip3] onnxruntime==1.27.0
[pip3] pyzmq==27.1.0
[pip3] tokenspeed-triton==3.7.10.post20260531
[pip3] torch==2.11.0+cu130
[pip3] torch_c_dlpack_ext==0.1.5
[pip3] torch-einops-utils==0.1.6
[pip3] torchaudio==2.11.0+cu130
[pip3] torchsde==0.2.6
[pip3] torchvision==0.26.0+cu130
[pip3] transformers==5.13.0
[pip3] triton==3.6.0
[pip3] x-transformers==2.23.3
[conda] pyzmq                      27.1.0           pypi_0                pypi

==============================
         vLLM Info
==============================
ROCM Version                 : Could not collect
vLLM Version                 : 0.24.0
vLLM-Omni Version            : 0.24.1.dev15+g3a64da194 (git sha: 3a64da194)
vLLM Build Flags:
  CUDA Archs: Not Set; ROCm: Disabled
GPU Topology:
  	[4mGPU0	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	16-31,48-63	1		N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

==============================
     Environment Variables
==============================
NVIDIA_VISIBLE_DEVICES=GPU-f9bd060a-83ad-efea-f40a-f82397e3e8e7
NVIDIA_REQUIRE_CUDA=cuda>=13.0 brand=unknown,driver>=535,driver<536 brand=grid,driver>=535,driver<536 brand=tesla,driver>=535,driver<536 brand=nvidia,driver>=535,driver<536 brand=quadro,driver>=535,driver<536 brand=quadrortx,driver>=535,driver<536 brand=nvidiartx,driver>=535,driver<536 brand=vapps,driver>=535,driver<536 brand=vpc,driver>=535,driver<536 brand=vcs,driver>=535,driver<536 brand=vws,driver>=535,driver<536 brand=cloudgaming,driver>=535,driver<536 brand=unknown,driver>=550,driver<551 brand=grid,driver>=550,driver<551 brand=tesla,driver>=550,driver<551 brand=nvidia,driver>=550,driver<551 brand=quadro,driver>=550,driver<551 brand=quadrortx,driver>=550,driver<551 brand=nvidiartx,driver>=550,driver<551 brand=vapps,driver>=550,driver<551 brand=vpc,driver>=550,driver<551 brand=vcs,driver>=550,driver<551 brand=vws,driver>=550,driver<551 brand=cloudgaming,driver>=550,driver<551 brand=unknown,driver>=565,driver<566 brand=grid,driver>=565,driver<566 brand=tesla,driver>=565,driver<566 brand=nvidia,driver>=565,driver<566 brand=quadro,driver>=565,driver<566 brand=quadrortx,driver>=565,driver<566 brand=nvidiartx,driver>=565,driver<566 brand=vapps,driver>=565,driver<566 brand=vpc,driver>=565,driver<566 brand=vcs,driver>=565,driver<566 brand=vws,driver>=565,driver<566 brand=cloudgaming,driver>=565,driver<566 brand=unknown,driver>=570,driver<571 brand=grid,driver>=570,driver<571 brand=tesla,driver>=570,driver<571 brand=nvidia,driver>=570,driver<571 brand=quadro,driver>=570,driver<571 brand=quadrortx,driver>=570,driver<571 brand=nvidiartx,driver>=570,driver<571 brand=vapps,driver>=570,driver<571 brand=vpc,driver>=570,driver<571 brand=vcs,driver>=570,driver<571 brand=vws,driver>=570,driver<571 brand=cloudgaming,driver>=570,driver<571 brand=unknown,driver>=575,driver<576 brand=grid,driver>=575,driver<576 brand=tesla,driver>=575,driver<576 brand=nvidia,driver>=575,driver<576 brand=quadro,driver>=575,driver<576 brand=quadrortx,driver>=575,driver<576 brand=nvidiartx,driver>=575,driver<576 brand=vapps,driver>=575,driver<576 brand=vpc,driver>=575,driver<576 brand=vcs,driver>=575,driver<576 brand=vws,driver>=575,driver<576 brand=cloudgaming,driver>=575,driver<576
NCCL_VERSION=2.28.3-1
NVIDIA_DRIVER_CAPABILITIES=all
NVIDIA_PRODUCT_NAME=CUDA
CUDA_VERSION=13.0.3
LD_LIBRARY_PATH=/root/.venv/lib/python3.12/site-packages/cv2/../../lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
CUDA_HOME=/usr/local/cuda
CUDA_HOME=/usr/local/cuda
PYTORCH_NVML_BASED_CUDA_CHECK=1
TORCHINDUCTOR_COMPILE_THREADS=1
TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_root
```


## Pytest result

---
To save computational costs and disk space, I modified the code from lines 37 to 42 in test_wan22_expansion.py, to focus on the pytest for the Wan-AI/Wan2.2-TI2V-5B model.
```bash
WAN22_MODELS = [
    ("Wan-AI/Wan2.2-T2V-A14B-Diffusers", "t2v"),
    ("Wan-AI/Wan2.2-I2V-A14B-Diffusers", "i2v"),
    ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", "ti2v"),
]
NPU_MODELS = [("Wan-AI/Wan2.2-I2V-A14B-Diffusers", "i2v")]
```
to
```bash
WAN22_MODELS = [
    ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", "ti2v"),
]
NPU_MODELS = []
```
This modification can be ignored if the disk space is enough.

```bash
uv pip install pytest pytest-asyncio
pytest -s -v tests/e2e/online_serving/test_wan22_expansion.py
```

```bash
================================================== warnings summary ==================================================
<frozen importlib._bootstrap>:488
  <frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute

<frozen importlib._bootstrap>:488
  <frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyObject has no __module__ attribute

../.venv/lib/python3.12/site-packages/torch/jit/_script.py:365: 14 warnings
  /root/.venv/lib/python3.12/site-packages/torch/jit/_script.py:365: DeprecationWarning: `torch.jit.script_method` is deprecated. Please switch to `torch.compile` or `torch.export`.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
--- Running Summary
=============================== 2 passed, 5 skipped, 16 warnings in 512.09s (0:08:32) ================================
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

## Test plan

---
The documented commands were run locally on 1x NVIDIA GeForce RTX 5090:

* Started in server`vllm serve Wan-AI/Wan2.2-TI2V-5B-Diffusers --omni --port 8092 --tensor-parallel-size 1`
* Checked status in client`curl -v http://127.0.0.1:8092/health`
* Got test image`wget https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg`
* Sent Text-only generation request
* Sent Text-Image generation request

**vLLM Version:** 0.24.0

### Text-only generation request
```bash
curl -X POST "http://127.0.0.1:8092/v1/videos/sync" ^
  -F "prompt=In a cherry blossom forest, the cherry blossoms sway gently in the breeze, and petals fall onto the path beneath the trees. A child go through the path. Cinematic quality, smooth motion." ^
  -F "negative_prompt=blurry, low quality, distorted, artifacts, low resolution, pixelated, deformed" ^
  -F "width=832" ^
  -F "height=480" ^
  -F "num_frames=48" ^
  -F "fps=16" ^
  -F "num_inference_steps=40" ^
  -F "guidance_scale=5.0" ^
  -F "flow_shift=12.0" ^
  -F "seed=42" ^
  --output t2v_output.mp4
```
### Text-Image generation request
```bash
curl -X POST "http://127.0.0.1:8092/v1/videos/sync" ^
  -F "input_reference=@./cherry_blossom.jpg" ^
  -F "prompt=A bird comes and stands on the branch. Cherry blossoms are blown off by the strong wind." ^
  -F "negative_prompt=blurry, low quality, distorted, artifacts, low resolution, pixelated, deformed" ^
  -F "width=832" ^
  -F "height=480" ^
  -F "num_frames=48" ^
  -F "fps=16" ^
  -F "num_inference_steps=40" ^
  -F "guidance_scale=5.0" ^
  -F "flow_shift=12.0" ^
  -F "seed=42" ^
  --output t2v_output.mp4
```

## Test result

---

|  | image | result |
| --- | --- | --- |
| Text-only generation | - | ![cherry_t.mp4](./child_seed42.mp4) |
| Text-Image generation | ![cherry.jpg](./cherry_blossom.jpg) | ![cherry_ti.mp4](./bird_seed42.mp4) |
