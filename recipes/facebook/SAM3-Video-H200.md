# SAM 3 video concept segmentation and tracking — H200

## Summary

- Vendor: Meta
- Model: `facebook/sam3` (SAM 3 detector + memory tracker, Transformers `Sam3VideoModel` checkpoint)
- Task: Find every instance of one or more text concepts in a video and track each one through all frames
- Mode: Online serving (`POST /v1/videos/segmentations`) and offline `Omni.generate`
- Hardware: 1x NVIDIA H200 141GB
- Maintainer: Community

## When to use this recipe

Serve SAM 3 to turn a video and a few concept names ("person", "car") into
per-object masklets: one stable id per instance, with a mask, a box and a score
on every frame. Typical uses are layer separation, matting and per-object edits
before video editing or generation with other vLLM-Omni models. The optional
per-object mask videos plug directly into compositing tools.

## Supported model contract

| Item | Contract |
| ---- | -------- |
| Endpoint | `POST /v1/videos/segmentations` ([Video Segmentation API](../../docs/serving/video_segmentation_api.md)); the same server also answers `POST /v1/images/segmentations` |
| Input | One uploaded video (up to 512 MiB, any container PyAV decodes); every frame is processed in decode order, resized internally to 1008x1008 |
| Prompts | Up to 16 text concepts per request (`prompt` plus the JSON list `prompts`); objects of different prompts are tracked independently |
| Output | Per frame, each visible object's id, COCO RLE mask, `[x0, y0, x1, y1]` box and score at the input resolution; optionally one H.264 mask video per object |
| Checkpoints | Transformers-format `Sam3VideoModel` (`facebook/sam3`, `model.safetensors`). Checkpoints without such tracker weights (Meta's native layout, SAM 3.1 multiplex) serve images only and answer 400 for video |
| Not covered | Point/box prompts on a chosen frame, interactive refinement, starting from a frame other than the first, the SAM 3.1 multiplex tracker |

## References

- Upstream: <https://github.com/facebookresearch/sam3>, <https://huggingface.co/facebook/sam3>
- Reference implementation used for parity: Transformers `Sam3VideoModel` (5.19.0)

## Hardware

- Accelerator model and per-device memory: NVIDIA H200, 141 GB
- Number of devices: 1
- Device interconnect: not used
- Qualification scope: single-request online serving in BF16

## Software environment

- OS: Ubuntu 22.04.5 LTS
- Python: 3.12.15
- Driver / runtime: NVIDIA 590.48.01, CUDA 13.2, PyTorch 2.13.0
- vLLM version: 0.32.0
- vLLM-Omni version or commit: `de2c39b9a` plus this change

## Command

`facebook/sam3` is gated: accept its terms on Hugging Face and export
`HF_TOKEN` first. The command was qualified with `jetjodh/sam3`, an ungated
copy of the same repository.

```bash
vllm serve facebook/sam3 --omni --port 8091
```

## Verification

```bash
curl -s http://localhost:8091/v1/videos/segmentations \
  -F "video=@dance.mp4" -F "prompt=person" \
  | jq '{num_frames, objects: [.objects[].id], first: [.frames[0].instances[] | {id, score, box}]}'
```

Expected: one object id per dancer, the same ids on later frames, scores
around 0.9 and boxes in input pixels.

## Notes

- Speed (BF16, first request after startup, end to end over HTTP):
    - a 124-frame 1344x768 clip with one object took 9.1 s including the mask
    video;
    - a 270-frame 1280x720 clip with 7 tracked people took 27.8 s.

  The tracker updates objects one at a time, as the reference does, so time
  grows with the number of tracked objects.
- Memory usage: 1.63 GiB of weights in BF16; 2.24 GiB process-scoped GPU memory
  after loading and 3.7 GiB after the requests above. Tracking memory is
  bounded by the memory window instead of growing with the video length.
- Accuracy: in FP32 the outputs match the Transformers reference loaded from
  the same checkpoint frame by frame. Five runs cover one to three prompts, 60
  to 270 frames and up to 50 objects, with 14 hot-start removals: 8,320
  instances in total. On every frame without a removal, all instances have the
  same object ids, mask IoU 1.0, identical boxes and a max score difference of
  4e-7. Transformers needs its Hub `cv_utils` kernel for detection NMS and hole
  filling; the reference runs used vLLM-Omni's in-tree implementations of
  both.
- The ten frames that differ are frames where hot-start removes an object that
  is followed by later objects. Transformers drops the object from its id list
  before pairing the id list with the tracked masks, so later objects receive
  their neighbour's mask on that frame (for example its object 17 shows object
  12's mask). This port pairs the masks with the ids tracked before the
  removal, as Meta's original implementation does.
- In BF16 the port stays within the reference's own FP32-vs-BF16 drift: mean
  mask IoU 0.993 against the BF16 reference, and 0.992 between the
  reference's two precisions.
- Known limitations: one video per request, at most 2048 frames; text prompts
  only; tracking starts at the first frame.
