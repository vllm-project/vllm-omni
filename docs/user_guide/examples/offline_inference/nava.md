# NAVA Offline Inference

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/nava>.


This example documents the native `NAVAPipeline` request shape in vLLM-Omni.
The text/image-conditioned audio-video and speaker timbre paths are wired for
real-checkpoint E2E validation. Speaker references use local ReDimNet assets
prepared by the download script; runtime inference does not fetch speaker code.

Prepare a local model directory:

```bash
python examples/offline_inference/nava/download_nava.py --local-dir /models/nava
```

Expected text-to-audio-video command shape:

```bash
python examples/offline_inference/nava/end2end.py \
  --model /models/nava \
  --prompt "A person speaks while standing near the sea." \
  --output outputs/nava_t2av.pt
```

Expected image-conditioned command shape:

```bash
python examples/offline_inference/nava/end2end.py \
  --model /models/nava \
  --prompt "Continue this first frame with natural speech." \
  --image first_frame.png \
  --output outputs/nava_i2av.pt
```

Expected speaker timbre command shape:

```bash
python examples/offline_inference/nava/end2end.py \
  --model /models/nava \
  --prompt "A person says <S>Hello from NAVA.<E>" \
  --spk-wavs speaker.wav \
  --output outputs/nava_timbre.pt
```

Acceleration and multi-GPU modes for NAVA are not listed as supported until
they are verified with real checkpoints.

## Example materials

??? abstract "download_nava.py"
    ``````py
    --8<-- "examples/offline_inference/nava/download_nava.py"
    ``````
??? abstract "end2end.py"
    ``````py
    --8<-- "examples/offline_inference/nava/end2end.py"
    ``````
