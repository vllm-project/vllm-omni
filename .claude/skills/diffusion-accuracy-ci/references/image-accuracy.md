# Image accuracy cases

Use this reference when adding or extending a real-model image accuracy pytest.
Apply [the reference contract](reference-contract.md) for provenance, input alignment,
repeated-run calibration, and threshold selection before choosing a metric.

## Choose the closest existing case

| Output contract | Repository example | Reusable behavior |
| --- | --- | --- |
| Text-to-image | [Qwen Image](../../../../tests/e2e/accuracy/test_qwen_image.py) | Native HTTP generation followed by an independent Diffusers reference; SSIM/PSNR or pixel error |
| Single or multiple input edits | [Qwen Image Edit](../../../../tests/e2e/accuracy/test_qwen_image_edit.py) | Ordered multipart inputs and equivalent reference image arguments |
| Ordered transparent layers | [Qwen Image Layered](../../../../tests/e2e/accuracy/test_qwen_image_layered.py) | Explicit layer count and pairwise RGBA comparison |

Use these files for lifecycle and interface patterns. Their prompts, resolutions,
steps, guidance values, backends, and numerical thresholds are model-specific.
Changing a model ID alone does not establish architecture or backend compatibility.

## Generate both sides in the current run

1. Define one case and map its fields to the native request and reference call.
   Include the pipeline class and task, rather than relying on automatic detection
   when the same model family has generation, editing, and layered variants.
2. Use [OmniServer](../../../../tests/helpers/runtime.py) as a context manager or
   use the repository's server fixture. Keep the request and output retrieval
   inside its lifetime. Check HTTP status before decoding the response.
3. For generation, follow `/v1/images/generations`; for editing, follow
   `/v1/images/edits`. Verify the model's actual supported fields rather than
   assuming `guidance_scale` and `true_cfg_scale` have interchangeable meanings.
4. Close the native server before loading a same-device reference if concurrent
   residency is unnecessary. Follow the examples' `try/finally` reference cleanup:
   release model hooks, drop the pipeline, collect garbage, and clean the test
   environment. Cleanup must also run after a request or assertion fails.
5. Decode and fully load each image while its byte stream is alive. Assert the
   response container, output count, dimensions, and expected channel semantics
   before metric conversion. Save both sides as lossless PNGs.

Reuse [accuracy fixtures](../../../../tests/e2e/accuracy/conftest.py) and
[helpers](../../../../tests/e2e/accuracy/helpers.py). `model_output_dir` creates
a directory but does not clear old files. Give parameterized cases and workers
distinct output paths; a previous file must never substitute for generation.

## Editing inputs and dimensions

- Materialize the same input bytes for both paths. Reuse repository assets where
  appropriate, and record any crop, resize, interpolation, orientation, or color
  conversion applied before inference. Check any model-side preprocessing too.
- Preserve the order of multiple conditioning images. Qwen's native edit example
  sends repeated `image` multipart fields; the reference receives the matching
  ordered list. A single-image reference may expect a PIL image instead of a list.
- Pass the same effective width and height to both paths. For model-specific
  `size="auto"` or `resolution`, state the resulting geometry and verify it;
  resizing generated outputs to make them comparable can hide a preprocessing bug.
- Preserve alpha when it belongs to the task. The RGB conversion used by a normal
  editing example is not appropriate for transparent input or output contracts.

## Layered and RGBA outputs

Treat the layers as an ordered sequence, not a batch of interchangeable pictures.
For Qwen Layered, `n=1` selects one result while `layers` selects its layer count;
the HTTP response contains one image per layer. Check both paths against the
requested count before comparing corresponding indexes.

Normalize only the documented output container shape. The existing layered helper
filters PIL elements from a nested result; a new case should first assert all
element types and expected nesting so malformed output cannot disappear silently.
Check layer dimensions and preserve layer order and all RGBA channels in PNGs.

Call `assert_image_sequence_similarity(..., compare_mode="RGBA")` for the current
RGBA pattern. Its default is RGB and its count assertion only checks equality
between the two lists. The shared image metrics compare the converted channels
directly; they do not composite or premultiply alpha. If the official contract
uses a different representation, align it explicitly and retain alpha diagnostics.

## Select metrics and retain evidence

| Helper | What it gates | Use when |
| --- | --- | --- |
| `assert_similarity` | Full-image SSIM and PSNR after normalized channel conversion | Structural and pixel fidelity with a calibrated tolerance |
| `assert_images_pixel_close` | Mean and p99 absolute channel error in `[0, 1]` | Close numerical alignment, as in the Qwen-Image-2512 case |
| `assert_image_sequence_similarity` | Per-index SSIM/PSNR | Every requested image or layer must meet the contract |
| `CLIPScorer` | Image-text or image-image embedding similarity | A declared semantic check that complements the numerical gate |

Pass `compare_mode` deliberately. Pixel p99 measures the tail across channel
samples; its printed mismatch ratios are diagnostics, not additional gates.
CLIP similarity alone cannot establish numerical fidelity to a reference.
`assert_similarity` can skip when the reference size differs from a supplied
requested size. Treat that skip as an invalid baseline and report it explicitly.

Retain the effective request, resolved reference provenance, both PNG outputs,
metrics, thresholds, device/backend, and logs with the failing case. For layers,
identify the failing index and retain every layer, including alpha. Follow the
[reference contract](reference-contract.md) when diagnosing drift or calibrating
a new model; do not loosen thresholds merely to accept a mismatch.
