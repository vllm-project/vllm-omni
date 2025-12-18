# \# Profiling vLLM-Omni

# \## profiling hooks for omni\&vllm\&diffusion pipeline

# 

# \## 1.Usage of Log Statistics for Single-Pipeline Diffusion Scheduling  

# 

# 

# In this project, tasks such as text-to-image and text-to-video follow a single-pipeline diffusion scheduling paradigm.  

# Each request triggers the diffusion pipeline as a whole, executing text encoding, denoising iterations, and decoding in a tightly coupled, end-to-end manner.

# 

# • The entire workflow is launched in one shot via `Omni.generate(...)`.

# 

# • Execution proceeds sequentially within the diffusion engine.

# 

# • Performance and behavior can be directly inspected through:

# 

# &nbsp; • Diffusion-level logs (e.g., denoising steps, post-processing),

# 

# &nbsp; • vLLM runtime logs (e.g., worker startup, device allocation).

# 

# > Text-to-Image / Text-to-Video → \*Single diffusion pipeline, single execution path\*  

# \### The log usage method is as follows:

# \### 1.Print the vllm feature.

# 1)vllm feature integration

# ```bash

# export VLLM\_LOGGING\_LEVEL=DEBUG

# ```

# 2)Run script(Taking image\_to\_image as an example, the usage method for other models is the same.):

# 

# ```python

# &nbsp;   python image\_edit.py \\

# &nbsp;       --image input.png \\

# &nbsp;       --prompt "Let this mascot dance under the moon, surrounded by floating stars and poetic bubbles such as 'Be Kind'" \\

# &nbsp;       --output output\_image\_edit.png \\

# &nbsp;       --num\_inference\_steps 50 \\

# &nbsp;       --cfg\_scale 4.0

# ```

# We can see the vLLM logs in the console and the diffusion logs in path/omni\_diffusion\_stats/omni\_diffusion\_%Y%m%d\_%H%M%S\_xx\_pidxxxx.jsonl.

# 

# ```json

# DEBUG 12-17 09:21:42 \[plugins/\_\_init\_\_.py:28] No plugins for group vllm.platform\_plugins found.

# DEBUG 12-17 09:21:42 \[platforms/\_\_init\_\_.py:34] Checking if TPU platform is available.

# DEBUG 12-17 09:21:42 \[platforms/\_\_init\_\_.py:52] TPU platform is not available because: No module named 'libtpu'

# DEBUG 12-17 09:21:42 \[platforms/\_\_init\_\_.py:58] Checking if CUDA platform is available.

# DEBUG 12-17 09:21:42 \[platforms/\_\_init\_\_.py:78] Confirmed CUDA platform is available.

# DEBUG 12-17 09:21:42 \[platforms/\_\_init\_\_.py:106] Checking if ROCm platform is available.

# DEBUG 12-17 09:21:42 \[platforms/\_\_init\_\_.py:120] ROCm platform is not available because: No module named 'amdsmi'

# DEBUG 12-17 09:21:42 \[platforms/\_\_init\_\_.py:127] Checking if XPU platform is available.

# DEBUG 12-17 09:21:42 \[platforms/\_\_init\_\_.py:146] XPU platform is not available because: No module named 'intel\_extension\_for\_pytorch'

# DEBUG 12-17 09:21:42 \[platforms/\_\_init\_\_.py:153] Checking if CPU platform is available.

# DEBUG 12-17 09:21:42 \[platforms/\_\_init\_\_.py:58] Checking if CUDA platform is available.

# DEBUG 12-17 09:21:42 \[platforms/\_\_init\_\_.py:78] Confirmed CUDA platform is available.

# INFO 12-17 09:21:42 \[platforms/\_\_init\_\_.py:216] Automatically detected platform cuda.

# DEBUG 12-17 09:21:47 \[compilation/decorators.py:155] Inferred dynamic dimensions for forward method of <class 'vllm.model\_executor.models.qwen3\_moe.Qwen3MoeModel'>: \['input\_ids', 'positions', 'intermediate\_tensors', 'inputs\_embeds']

# WARNING 12-17 09:21:47 \[mooncake\_connector.py:18] Mooncake not available, MooncakeOmniConnector will not work

# DEBUG 12-17 09:21:47 \[factory.py:35] Registered connector: MooncakeConnector

# DEBUG 12-17 09:21:47 \[factory.py:35] Registered connector: SharedMemoryConnector

# DEBUG 12-17 09:21:48 \[distributed/device\_communicators/shm\_broadcast.py:313] Connecting to ipc:///tmp/5c30e5fa-26de-43e1-bd35-d551269b0fe2

# DEBUG 12-17 09:21:48 \[distributed/device\_communicators/shm\_broadcast.py:243] Binding to ipc:///tmp/7c1c23a5-2d1c-4f83-a6f2-36d8c4c71644

# INFO 12-17 09:21:48 \[distributed/device\_communicators/shm\_broadcast.py:289] vLLM message queue communication handle: Handle(local\_reader\_ranks=\[0], buffer\_handle=(1, 10485760, 10, 

# INFO 12-17 09:22:16 \[diffusers\_loader.py:214] Loading weights took 17.82 seconds

# INFO 12-17 09:22:16 \[gpu\_worker.py:81] Model loading took 53.7462 GiB and 27.811149 seconds

# INFO 12-17 09:22:16 \[gpu\_worker.py:86] Worker 0: Model loaded successfully.

# INFO 12-17 09:22:16 \[gpu\_worker.py:237] Worker 0: Scheduler loop started.

# INFO 12-17 09:22:16 \[gpu\_worker.py:175] Worker 0 ready to receive requests via shared memory

# DEBUG 12-17 09:22:16 \[diffusion\_engine.py:147] All workers are ready

# DEBUG 12-17 09:22:16 \[distributed/device\_communicators/shm\_broadcast.py:313] Connecting to ipc:///tmp/7c1c23a5-2d1c-4f83-a6f2-36d8c4c71644

# INFO 12-17 09:22:16 \[scheduler.py:45] SyncScheduler initialized result MessageQueue

# INFO 12-17 09:22:16 \[omni\_diffusion.py:114] OmniDiffusion initialized: model=path/models/Qwen-Image-Edit, class=QwenImageEditPipeline, init\_ms=36702.19

# Pipeline loaded

# ```

# ---

# \## omni\_diffusion\_%Y%m%d\_%H%M%S\_xx\_pidxxxx.jsonl

# ```json

# {"model": "path/models/Qwen-Image", "model\_class": "QwenImagePipeline", "init\_ms": 17562.917941002524, "event": "engine\_load", "ts": 1766019712.405319, "pid": 18635, "host": "xxxx"}

# {"n\_requests": 1, "prompt\_chars": 28, "height": 1024, "width": 1024, "generator": "<torch.\_C.Generator object at 0x7fc71d96e8f0>", "true\_cfg\_scale": 4.0, "num\_inference\_steps": 50, "num\_outputs\_per\_prompt": 1, "event": "request\_scheduled", "ts": 1766019712.405916, "pid": 18635, "host": "xxxx"}

# {"n\_requests": 1, "total\_ms": 42437.41700099781, "diffusion\_total\_ms": 42437.13191100687, "denoise\_avg\_ms": 848.7426382201375, "input\_tokens": 28, "input\_tokens\_per\_s": 0.6597951048562086, "event": "request\_finished", "ts": 1766019754.8433862, "pid": 18635, "host": "xxxx"}

# 

# ```

# \### 2.The vllm feature is not printed..

# Run script:

# ```python

# &nbsp;   python image\_edit.py \\

# &nbsp;       --image input.png \\

# &nbsp;       --prompt "Let this mascot dance under the moon, surrounded by floating stars and poetic bubbles such as 'Be Kind'" \\

# &nbsp;       --output output\_image\_edit.png \\

# &nbsp;       --num\_inference\_steps 50 \\

# &nbsp;       --cfg\_scale 4.0

# ```

# We can see the vLLM logs in the console .The characteristics of diffusion models will still be statistically analyzed.

# 

# ```json

# INFO 12-17 09:28:58 \[\_\_init\_\_.py:216] Automatically detected platform cuda.

# WARNING 12-17 09:29:03 \[mooncake\_connector.py:18] Mooncake not available, MooncakeOmniConnector will not work

# Loaded input image from input.png (size: (514, 556))

# INFO 12-17 09:29:06 \[shm\_broadcast.py:289] vLLM message queue communication handle: Handle(local\_reader\_ranks=\[0], buffer\_handle=(1, 10485760, 10, 'psm\_0c8120b1'), local\_subscribe\_addr='ipc:///tmp/7f7c25ae-cf87-4c4d-b79d-17cbb4ea00e2', remote\_subscribe\_addr=None, remote\_addr\_ipv6=False)

# INFO 12-17 09:29:06 \[diffusion\_engine.py:92] Starting server...

# .......

# INFO 12-17 09:29:26 \[diffusion\_engine.py:43] Pre-processing completed in 0.0564 seconds

# INFO 12-17 09:30:26 \[shm\_broadcast.py:466] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation).

# INFO 12-17 09:31:17 \[diffusion\_engine.py:48] Generation completed successfully.

# INFO 12-17 09:31:17 \[diffusion\_engine.py:53] Post-processing completed in 0.0651 seconds

# INFO 12-17 09:31:17 \[omni\_diffusion.py:177] request\_finished: n\_requests=1, total\_ms=111360.70

# INFO 12-17 09:31:17 \[omni\_diffusion.py:184] request\_scheduled: n\_requests=1, kwargs\_keys=\['pil\_image', 'negative\_prompt', 'generator', 'true\_cfg\_scale', 'num\_inference\_steps', 'num\_outputs\_per\_prompt'], kwargs\_detail={'generator': '<torch.\_C.Generator object at 0x7f0c71d328d0>', 'true\_cfg\_scale': 4.0, 'num\_inference\_steps': 50, 'num\_outputs\_per\_prompt': 1}

# INFO 12-17 09:31:17 \[omni\_diffusion.py:190] OMNI\_DIFFUSION\_METRICS {"prompt\_chars": 103, "input\_tokens": 103, "input\_tokens\_per\_s": 0.9249223665998467, "num\_inference\_steps": 50, "diffusion\_total\_ms": 111360.43418200097, "denoise\_avg\_ms": 2227.2086836400194, "total\_ms": 111360.69763200067}

# Total generation time: 111.3614 seconds (111361.44 ms)

# Saved edited image to path/vllm-omni/examples/offline\_inference/image\_to\_image/output\_image\_edit.png

# INFO 12-17 09:31:17 \[gpu\_worker.py:190] Worker 0: Received shutdown message

# INFO 12-17 09:31:17 \[gpu\_worker.py:214] event loop terminated.

# INFO 12-17 09:31:17 \[gpu\_worker.py:114] Worker 0: Destroyed process group

# INFO 12-17 09:31:17 \[gpu\_worker.py:245] Worker 0: Shutdown complete.

# 

# ```

# ---

# \## 2.Usage of Log Statistics for Multi-Process, Multi-Stage Scheduling  

# 

# In contrast, Qwen2.5-Omni and Qwen3-Omni adopt a multi-process, multi-stage scheduling model driven by OmniLLM.  

# Rather than executing a single pipeline, the system decomposes the task into multiple stages, each running as an independent process.

# 

# • The core abstraction is a stage-based pipeline:

# 

# &nbsp; • Each stage performs a specific function (e.g., reasoning, generation, modality transformation).

# 

# &nbsp; • Stages are connected via inter-process communication (IPC).

# 

# • Scheduling is pipeline-oriented:

# 

# &nbsp; • Downstream stages are activated once upstream stages complete.

# 

# &nbsp; • Multiple stages can overlap in time, enabling pipeline parallelism.

# 

# • System behavior can be observed through:

# 

# &nbsp; • Omni-level logs (stage transitions and orchestration),

# 

# &nbsp; • Diffusion logs (if diffusion is involved in a stage),

# 

# &nbsp; • vLLM logs (process lifecycle, execution and resource usage).

# 

# > Qwen2.5-Omni / Qwen3-Omni → \*Multi-process, multi-stage pipeline with explicit scheduling\*

# 

# 1\. Setting the log switch.:

# 

# ```python

# &nbsp;   omni\_llm = Omni(

# &nbsp;       model=model\_name,

# &nbsp;       log\_stats=args.enable\_stats,#Setting  enable\_stats=True 

# &nbsp;       log\_file=(os.path.join(log\_dir, "omni\_llm\_pipeline.log") if args.enable\_stats else None)

# &nbsp;   )

# ```

# or

# ```python

# &nbsp;   omni\_llm = Omni(

# &nbsp;       model=model\_name,

# &nbsp;       log\_stats=True 

# &nbsp;       log\_file=os.path.join(log\_dir, "omni\_llm\_pipeline.log") 

# &nbsp;   )

# 

# ```

# 2\. Run  script:

# 

# ```bash

# sh run\_multiple\_prompts.sh

# ```

# or

# ```bash

# run\_single\_prompt.sh

# ```

# 4\. vllm feature integration

# ```bash

# export VLLM\_LOGGING\_LEVEL=DEBUG

# ```

# We can see the debug log（vllm+omni+diffusion）in omni\_llm\_pipeline.log:

# ```json

# 2025-12-16 01:24:23,021 \[PID:17815] DEBUG: \[Orchestrator] generate() called

# 2025-12-16 01:24:23,021 \[PID:17815] DEBUG: \[Orchestrator] Seeding 1 requests into stage-0

# 2025-12-16 01:24:23,022 \[PID:17815] DEBUG: \[Orchestrator] Enqueued request 0\_b3b2dcb1-4c75-42de-a073-dcef52b9e557 to stage-0

# 2025-12-16 01:24:23,023 \[PID:17815] DEBUG: \[Orchestrator] Entering scheduling loop: total\_requests=1, stages=3

# 2025-12-16 01:24:26,527 \[PID:17815] INFO: \[StageMetrics] stage=0 req=0\_b3b2dcb1-4c75-42de-a073-dcef52b9e557 metrics={'num\_tokens\_out': 52, 'stage\_gen\_time\_ms': 3490.6439781188965, 'batch\_id': 1, 'rx\_decode\_time\_ms': 0.036716461181640625, 'rx\_transfer\_bytes': 339, 'rx\_in\_flight\_time\_ms': 0.0}

# 2025-12-16 01:24:26,527 \[PID:17815] DEBUG: \[Orchestrator] Stage-0 completed request 0\_b3b2dcb1-4c75-42de-a073-dcef52b9e557; forwarding or finalizing

# 2025-12-16 01:24:26,527 \[PID:17815] DEBUG: \[Orchestrator] Request 0\_b3b2dcb1-4c75-42de-a073-dcef52b9e557 finalized at stage-0

# 2025-12-16 01:24:26,780 \[PID:17815] DEBUG: \[Orchestrator] Forwarded request 0\_b3b2dcb1-4c75-42de-a073-dcef52b9e557 to stage-1

# 2025-12-16 01:24:44,789 \[PID:17815] INFO: \[StageMetrics] stage=1 req=0\_b3b2dcb1-4c75-42de-a073-dcef52b9e557 metrics={'num\_tokens\_out': 170, 'stage\_gen\_time\_ms': 17991.965770721436, 'batch\_id': 1, 'rx\_decode\_time\_ms': 5.737543106079102, 'rx\_transfer\_bytes': 3148794, 'rx\_in\_flight\_time\_ms': 1.1227130889892578}

# 2025-12-16 01:24:44,789 \[PID:17815] DEBUG: \[Orchestrator] Stage-1 completed request 0\_b3b2dcb1-4c75-42de-a073-dcef52b9e557; forwarding or finalizing

# 2025-12-16 01:24:44,790 \[PID:17815] DEBUG: \[Orchestrator] Forwarded request 0\_b3b2dcb1-4c75-42de-a073-dcef52b9e557 to stage-2

# 2025-12-16 01:24:44,914 \[PID:17815] INFO: \[StageMetrics] stage=2 req=0\_b3b2dcb1-4c75-42de-a073-dcef52b9e557 metrics={'num\_tokens\_out': 0, 'stage\_gen\_time\_ms': 117.71297454833984, 'batch\_id': 1, 'rx\_decode\_time\_ms': 0.43487548828125, 'rx\_transfer\_bytes': 8393, 'rx\_in\_flight\_time\_ms': 0.5235671997070312}

# 2025-12-16 01:24:44,915 \[PID:17815] DEBUG: \[Orchestrator] Stage-2 completed request 0\_b3b2dcb1-4c75-42de-a073-dcef52b9e557; forwarding or finalizing

# 2025-12-16 01:24:44,915 \[PID:17815] DEBUG: \[Orchestrator] Request 0\_b3b2dcb1-4c75-42de-a073-dcef52b9e557 finalized at stage-2

# 2025-12-16 01:24:44,915 \[PID:17815] DEBUG: \[Orchestrator] Request 0\_b3b2dcb1-4c75-42de-a073-dcef52b9e557 fully completed (1/1)

# 2025-12-16 01:24:44,915 \[PID:17815] DEBUG: \[Orchestrator] All requests completed

# 2025-12-16 01:24:44,915 \[PID:17815] INFO: \[Summary] {'e2e\_requests': 1, 'e2e\_total\_time\_ms': 21893.684148788452, 'e2e\_sum\_time\_ms': 21892.935752868652, 'e2e\_total\_tokens': 0, 'e2e\_avg\_time\_per\_request\_ms': 21892.935752868652, 'e2e\_avg\_tokens\_per\_s': 0.0, 'wall\_time\_ms': 21893.684148788452, 'final\_stage\_id': 2, 'stages': \[{'stage\_id': 0, 'requests': 1, 'tokens': 52, 'total\_time\_ms': 3505.100727081299, 'avg\_time\_per\_request\_ms': 3505.100727081299, 'avg\_tokens\_per\_s': 14.835522299897058}, {'stage\_id': 1, 'requests': 1, 'tokens': 170, 'total\_time\_ms': 18008.86106491089, 'avg\_time\_per\_request\_ms': 18008.86106491089, 'avg\_tokens\_per\_s': 9.43979740791238}, {'stage\_id': 2, 'requests': 1, 'tokens': 0, 'total\_time\_ms': 124.7246265411377, 'avg\_time\_per\_request\_ms': 124.7246265411377, 'avg\_tokens\_per\_s': 0.0}], 'transfers': \[{'from\_stage': 0, 'to\_stage': 1, 'samples': 1, 'total\_bytes': 3148794, 'total\_time\_ms': 5.67626953125, 'tx\_mbps': 4437.835776, 'rx\_samples': 1, 'rx\_total\_bytes': 3148794, 'rx\_total\_time\_ms': 5.737543106079102, 'rx\_mbps': 4390.442308539705, 'total\_samples': 1, 'total\_transfer\_time\_ms': 12.53652572631836, 'total\_mbps': 2009.3567029593396}, {'from\_stage': 1, 'to\_stage': 2, 'samples': 1, 'total\_bytes': 8393, 'total\_time\_ms': 0.35572052001953125, 'tx\_mbps': 188.7549247828418, 'rx\_samples': 1, 'rx\_total\_bytes': 8393, 'rx\_total\_time\_ms': 0.43487548828125, 'rx\_mbps': 154.39821698245615, 'total\_samples': 1, 'total\_transfer\_time\_ms': 1.3141632080078125, 'total\_mbps': 51.092588493468796}]}

# 

# ```

# 

# \## If you do not need to print the vLLM features, you can run the script directly, or unset VLLM\_LOGGING\_LEVEL. 

# ```bash

# unset VLLM\_LOGGING\_LEVEL

# ```

