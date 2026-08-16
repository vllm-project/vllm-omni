# HunyuanImage-3.0-Instruct

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/hunyuan_image3>.

This directory holds `end2end.py`, a full-featured offline script covering
functionality the shared task examples don't yet have (streaming
chain-of-thought display, diffusion KV-cache dtype / skip-steps control) and
`reproduce.sh`, a thin repro script for the shared-example path below.

For the standard four-modality path (text-to-image, image editing,
image-to-text, text-to-text), HunyuanImage-3.0-Instruct runs through the
**shared task examples**, with all model-specific knobs declared centrally in
`vllm_omni/model_extras/hunyuan_image3.py` and routed via `--extra-body` /
`--extra-args`:

| Modality | How to run |
| :--- | :--- |
| Text to image (`t2i`) | shared `examples/offline_inference/text_to_image/text_to_image.py` |
| Image editing (`it2i`) | shared `examples/offline_inference/image_to_image/image_edit.py` |
| Image to text (`i2t`) | shared `examples/offline_inference/x_to_text/x_to_text.py --image ...` |
| Text to text (`t2t`) | shared `examples/offline_inference/x_to_text/x_to_text.py` |

See
[`recipes/Tencent/HunyuanImage-3.0-Instruct.md`](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Tencent/HunyuanImage-3.0-Instruct.md)
for full run commands, deploy-config selection, declared `extra_body`
parameters, hardware-specific configurations, and benchmark data -- that
recipe is the single documentation home for this model; this README stays a
short pointer to it plus the local `end2end.py` / `reproduce.sh` scripts.

## Example materials

??? abstract "end2end.py"
    Large file omitted from the rendered docs. View it on GitHub: <https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/hunyuan_image3/end2end.py>.

??? abstract "reproduce.sh"
    ``````sh
    --8<-- "examples/offline_inference/hunyuan_image3/reproduce.sh"
    ``````
