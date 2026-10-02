# WF-05: MiniMax-H3 latent-mask editing

Edit a source video with MiniMax-H3 using spatial or temporal masks. The workflow includes object removal, inpainting, continuation, and extension, with a mask preview for each case.

## Requirements

- ComfyUI with ComfyUI-vLLM-Omni installed.
- An H3 service using the FL2VA checkpoint partition with latent-mask editing support. Follow the [H3 recipe](../../../recipes/MiniMaxAI/MiniMax-H3.md) for server setup.

## Use

Open **vLLM-Omni MiniMax-H3 Latent Mask Editing.json** from the example workflows and set the Generate Video URL/model for the case you want to run.

1. Upload your source video in Load Video.
2. Adjust the case's mask. For removal and inpainting, set Mask Canvas to the video dimensions, Mask Rectangle to the region size, and Combine Masks x/y to its position. For continuation and extension, set MiniMax-H3 Temporal Mask's mode and duration; continuation also uses `preserve_fraction`. Match the duration in Generate Video, and use a duration longer than the source for extension. Mask values are `0` to preserve and `1` to regenerate.
3. Adjust the prompt to describe the desired content. Set the output dimensions and duration in Generate Video. If editing audio, update the sound description and audio mask (`0` preserves, `1` regenerates the whole clip).
4. Select the case's Mask Preview Save Video node and choose **Execute to selected output nodes** to preview the mask as a red overlay without inference, or select its Result Save Video node to generate the edited video. Running the entire workflow generates all four cases.
