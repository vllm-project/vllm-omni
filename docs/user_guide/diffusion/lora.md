# Diffusion LoRA

vLLM-Omni provides a shared LoRA backend for diffusion pipelines. It supports
startup fusion, dynamic execution, request-scoped adapter selection, weighted
multi-adapter composition, and an immutable startup registry.

For one linear layer, adapter $i$ may provide a low-rank update $B_iA_i$ and an
optional per-layer bias update $d_i$. For prefused adapters $P$ and dynamic
adapters $D$, both execution modes share this definition:

$$
y = \left(W + \sum_{i \in P} s_i B_i A_i\right)x
  + \left(b + \sum_{i \in P} s_i d_i\right)
  + \sum_{j \in D} s_j\left(B_j A_j x + d_j\right)
$$

An adapter without a bias update has $d_i = 0$. One adapter may carry separate
bias updates for many target layers; $d_i$ is not one global bias.

The modes differ in when this expression is evaluated:

| Mode | Startup option | Weight lifecycle | Request switching | Quantized base weights |
|---|---|---|---|---|
| Dynamic | `--dynamic-lora PATH` | Loads an inactive adapter at startup and evaluates the low-rank branch only when selected | Yes | Yes |
| Prefused | `--prefused-lora PATH=SCALE` | Merges the delta into dense weights once at startup | No | No |

Dynamic LoRA is the recommended default for serving. Prefusion is useful only
when permanent dense weights are required and its output quality has been
validated for the model and dtype.

## Adapter formats

The generic loader accepts a PEFT directory containing `adapter_config.json`
and adapter weights:

```text
lora_adapter/
├── adapter_config.json
└── adapter_model.safetensors
```

The PEFT configuration describes the rank, alpha, and target modules. A local
path or Hugging Face repository ID may be registered at startup. Requests may
only select adapters registered with `--dynamic-lora`; they never resolve a
new local path or trigger a Hugging Face download.

Single-file `.safetensors` adapters are also supported when their checkpoint
layout is compatible with the selected pipeline. Check the corresponding
recipe for supported adapter repositories and formats.

Model-owned single-file adapters currently include MiniMax-H3 Turbo,
Qwen-Image, Wan2.1 T2V, and Wan2.2 T2V. Wan2.2 assigns adapters containing
`high_noise` and `low_noise` in their filenames to the corresponding
transformers, so the two files can be passed in either CLI order. Current
Wan2.2 I2V LightX2V files also contain dense and bias deltas; use the offline
assembly workflow below because those tensors are not low-rank LoRA terms.

## Dynamic serving

Register one request-selectable adapter at startup:

```bash
vllm serve BASE_MODEL \
  --omni \
  --dynamic-lora '{"path":"/path/to/adapter","name":"accelerator"}' \
  --max-cpu-loras 1
```

Repeat `--dynamic-lora` to register multiple adapters. The registry capacity
must cover all of them:

```bash
vllm serve BASE_MODEL \
  --omni \
  --dynamic-lora '{"path":"/path/to/accelerator.safetensors","name":"accelerator"}' \
  --dynamic-lora '{"path":"org/style-adapter","name":"style"}' \
  --max-cpu-loras 2
```

Dynamic specifications accept `PATH` or a JSON object with `path` and optional
`name`. A bare path derives its name from the final path component. Registered
names must be unique. Registration never activates an adapter and does not
accept a scale; clients set composition scales per request. Every selectable
adapter must be listed before the server starts.

### Request selection

The Images and Videos APIs accept a `lora` object or list. Each entry selects a
startup `--dynamic-lora` adapter by `name`; `scale` defaults to `1.0`:

```json
{
  "lora": [
    {"name": "accelerator", "scale": 1.0},
    {"name": "style", "scale": 0.6}
  ]
}
```

Request behavior is explicit:

| Request `lora` value | Dynamic adapters used by the request |
|---|---|
| Field omitted or `null` | None |
| `[]` | None |
| One object | That registered adapter only |
| List of objects | That registered weighted composition |

For example, a synchronous video request can select and scale an adapter:

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/videos/sync \
  -F 'model=BASE_MODEL' \
  -F 'prompt=A cinematic wide shot of a singer on an open-air stage.' \
  -F 'lora={"name":"accelerator","scale":0.8}'
```

The same object or list can be passed as the `lora` field of an Images API
JSON request. Unknown names are rejected before scheduler admission, and
request handling resolves names to server-owned canonical paths without
invoking the loader. Request paths are rejected even when they match a
registration.

Duplicate adapter names in one composition have their scales added, zero-scale
results are removed, and non-finite scales are rejected. Requests with
different adapter compositions are scheduled in separate diffusion batches.

## Prefusion

Use `--prefused-lora` to merge one or more weighted adapters into the dense
weights at startup:

```bash
vllm serve BASE_MODEL \
  --omni \
  --prefused-lora /path/to/accelerator.safetensors=1.0 \
  --prefused-lora /path/to/style-adapter=0.6
```

The backend accumulates each dense delta in FP32 and copies the merged result
to the base dtype once. The fused contribution is permanent for the lifetime
of the process: request-level `lora=[]` disables only dynamic adapters and
cannot remove a prefused delta.

Prefused and dynamic adapters may be used together according to the equation
above. Do not specify the same adapter in both sets unless applying its delta
twice is intentional. Prefusion is rejected for quantized diffusion weights
because their serialized or runtime representation is not a dense
floating-point weight that can safely receive an in-place LoRA delta.

## Compile, offload, and registry capacity

Dynamic adapters are installed before compilation, CPU/layerwise offload, and
diffusion cache wrapping. Requests may select, disable, compose, or reweight
those registered adapters, but cannot load another adapter, introduce a new
target layer, or expand the allocated rank. This rule is the same in eager and
compiled deployments. An explicit empty composition remains valid.

`--max-cpu-loras` bounds the immutable per-worker dynamic registry. Registered
adapters remain resident; request handling never inserts, removes, or evicts
them. Set it to at least the number of `--dynamic-lora` entries.

Dynamic LoRA executes the dense base layer through its configured quantization
method and adds the low-rank branch separately, so it can be combined with a
quantized base model. This does not imply bitwise equality with an unquantized
reference.

Distributed layerwise offload supports startup-preloaded dynamic LoRA. It uses
the ordinary CPU loader before installing LoRA wrappers, then DLO streams only
the dense base weights while the small A/B runtime slots remain on the
execution device. This combination does not use DLO's direct-mmap startup
optimization. Prefused LoRA with DLO is rejected; use `--dynamic-lora`.

## Sampling remains model-owned

Loading an acceleration or distilled LoRA does not change timesteps, guidance,
the scheduler, or the number of denoising steps. Set those independently
through deployment defaults or request sampling parameters according to the
adapter's published usage instructions.

## Model integration and extension points

The shared backend owns startup loading, immutable registration, weighted
composition, layer installation, dynamic execution, and prefusion. Models only
describe checkpoint normalization and how logical adapter tensors bind to their
modules:

| Model interface | Model-provided fields | Shared backend behavior |
|---|---|---|
| `get_lora_load_plan(adapter_path, tensor_keys)` | PEFT metadata, `weights_mapper`, `state_dict_converter` | Download, deserialize, validate, and register |
| `get_lora_apply_plan()` | `component_names`, `target_modules`, `packed_modules_mapping` | Install TP-aware layers, bind packed slices, compose, activate, and fuse |

The generic PEFT-directory path normally needs no model-specific load plan. A
raw single-file checkpoint can return `DiffusionLoRALoadPlan` when its keys or
layout need conversion. Its `state_dict_converter` returns either normalized A/B
tensors or `ConvertedLoRAState`, which additionally carries a tuple of typed
auxiliary updates.

`get_lora_load_plan()` may be implemented by the pipeline or one of its declared
components; incompatible plans are rejected. `get_lora_apply_plan()` is the
pipeline's declarative binding contract. These plans customize checkpoint
recognition, alpha handling, tensor conversion, component routing, target
selection, and packed-projection mapping. They do not inject arbitrary forward
callbacks or take ownership of registration and composition.

The backend currently implements `AdditiveBiasUpdate`. A converter may return
any number of these updates, one for each affected module or packed slice. Each
is shape-checked, scaled with its adapter, and supported by both dynamic and
prefused execution. New auxiliary mathematics requires a new typed update plus
shared validation and application support; unknown update types and unsupported
nonzero dense deltas are rejected.

### Wan example

Wan uses both plans without adding a separate backend. The load plan recognizes
the publication format, folds alpha, converts keys, and routes Wan2.2
`high_noise` and `low_noise` files to `transformer` and `transformer_2`. Its
converter returns ordinary A/B tensors plus every supported `.lora_B.bias` as a
separate typed update:

```python
def convert(state_dict):
    # Wan-specific alpha, key, and component normalization occurs first.
    lora_tensors = {}
    auxiliary_updates = []
    for key, tensor in state_dict.items():
        if key.endswith(".lora_B.bias"):
            auxiliary_updates.append(
                AdditiveBiasUpdate(
                    module_name=key.removesuffix(".lora_B.bias"),
                    tensor=tensor,
                )
            )
        else:
            lora_tensors[key] = tensor
    return ConvertedLoRAState(
        lora_tensors=lora_tensors,
        auxiliary_updates=tuple(auxiliary_updates),
    )

return DiffusionLoRALoadPlan(
    peft_config={
        "lora_alpha": None,
        "target_modules": list(_WAN_LORA_TARGETS),
    },
    state_dict_converter=convert,
)
```

Its apply plan declares both transformer components, supported target modules,
and the logical `to_q`/`to_k`/`to_v` to packed `to_qkv` mapping. MiniMax-H3 uses
the same interfaces for metadata alpha, key mapping, and FFN row reordering;
Qwen-Image uses them for publication-format conversion and packed QKV binding.

## Offline inference

The same request types are available through the Python API:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.diffusion.lora.types import registered_lora_request

adapter_path = "/path/to/lora_adapter"
omni = Omni(
    model="stabilityai/stable-diffusion-3.5-medium",
    dynamic_lora=[f'{{"path":"{adapter_path}","name":"style"}}'],
)

params = OmniDiffusionSamplingParams(
    num_inference_steps=28,
    lora_request=registered_lora_request("style"),
    lora_scale=0.8,
)
outputs = omni.generate("A piece of cheesecake", params)
```

For multiple adapters, pass matching tuples of registered name selectors and scales.

## Wan2.2 LightX2V Offline Assembly

This workflow is LoRA-adjacent: it uses external LightX2V conversion plus
`Wan2.2-Distill-Loras` to bake converted Wan2.2 I2V checkpoints into a local
Diffusers directory, instead of loading LoRA adapters at runtime.

### Required assets

- Base model: `Wan-AI/Wan2.2-I2V-A14B`
- Diffusers skeleton: `Wan-AI/Wan2.2-I2V-A14B-Diffusers`
- Optional external converter from the LightX2V project (not shipped in this repository)
- Optional LoRA weights: `lightx2v/Wan2.2-Distill-Loras`

### Step 1: Optional - convert high/low-noise DiT weights with LightX2V

Install or clone LightX2V from the upstream repository
(`https://github.com/ModelTC/LightX2V`). After cloning, the converter used
below is available at `<lightx2v_root>/tools/convert/converter.py`.

```bash
python /path/to/lightx2v/tools/convert/converter.py \
  --source /path/to/Wan2.2-I2V-A14B/high_noise_model \
  --output /tmp/wan22_lightx2v/high_noise_out \
  --output_ext .safetensors \
  --output_name diffusion_pytorch_model \
  --model_type wan_dit \
  --direction forward \
  --lora_path /path/to/wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors \
  --lora_key_convert auto \
  --single_file

python /path/to/lightx2v/tools/convert/converter.py \
  --source /path/to/Wan2.2-I2V-A14B/low_noise_model \
  --output /tmp/wan22_lightx2v/low_noise_out \
  --output_ext .safetensors \
  --output_name diffusion_pytorch_model \
  --model_type wan_dit \
  --direction forward \
  --lora_path /path/to/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors \
  --lora_key_convert auto \
  --single_file
```

If you are not using LightX2V, skip this step and either keep the original
Diffusers weights from the skeleton or point Step 2 at any other converted
`transformer/` and `transformer_2/` checkpoints.

### Step 2: Assemble a final Diffusers-style directory

```bash
python tools/wan22/assemble_wan22_i2v_diffusers.py \
  --diffusers-skeleton /path/to/Wan2.2-I2V-A14B-Diffusers \
  --transformer-weight /tmp/wan22_lightx2v/high_noise_out \
  --transformer-2-weight /tmp/wan22_lightx2v/low_noise_out \
  --output-dir /path/to/Wan2.2-I2V-A14B-Custom-Diffusers \
  --asset-mode symlink \
  --overwrite
```

`--transformer-weight` and `--transformer-2-weight` are optional. If you omit
them, the tool keeps the original weights from the Diffusers skeleton.

### Step 3: Run offline inference

```bash
python examples/offline_inference/image_to_video/image_to_video.py \
  --model /path/to/Wan2.2-I2V-A14B-Custom-Diffusers \
  --image /path/to/input.jpg \
  --prompt "A cat playing with yarn" \
  --num-frames 81 \
  --num-inference-steps 4 \
  --tensor-parallel-size 4 \
  --height 480 \
  --width 832 \
  --flow-shift 12 \
  --sample-solver euler \
  --guidance-scale 1.0 \
  --guidance-scale-high 1.0 \
  --boundary-ratio 0.875
```

Notes:

- This route avoids runtime LoRA loading changes in vLLM-Omni when you choose to bake converted weights into a local Diffusers directory.
- Output quality and speed depend on the replacement checkpoints and sampling params you choose.


## See Also

- [Text-to-Image Offline Example](../examples/offline_inference/text_to_image.md#lora) - Complete offline LoRA example
- [Text-to-Image Online Example](../examples/online_serving/text_to_image.md#lora) - Complete online LoRA example
