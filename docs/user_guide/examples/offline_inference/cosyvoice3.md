# CosyVoice3

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/cosyvoice3>.


## Setup

Install dependencies:
```
uv pip install -e .
```

> **Note:** This includes required libraries such as `librosa`, `soundfile`,
> `onnxruntime`, `x-transformers`, and `einops` via
> `requirements/common.txt` and platform-specific requirements files.

Download the model snapshot:
```
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
```

Add `config.json` in `pretrained_models/Fun-CosyVoice3-0.5B/`:
```json
{
    "model_type": "cosyvoice3",
    "architectures": [
        "CosyVoice3Model"
    ]
}
```

> **Why `config.json` is required:**
> `AutoConfig.register("cosyvoice3", CosyVoice3Config)` only registers a class mapping.
> The loader still needs `model_type: "cosyvoice3"` from `config.json` to select that class.
> If no `config.json` is present, model type cannot be inferred automatically.
> If your downloaded checkpoint already includes a valid `config.json` with
> `model_type: "cosyvoice3"`, this manual step can be skipped.

Run the offline verification script:
```
python examples/offline_inference/cosyvoice3/verify_e2e_cosyvoice.py \
  --model pretrained_models/Fun-CosyVoice3-0.5B \
  --tokenizer pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN
```

## Implementation Overview

CosyVoice3 runs as a **2-stage Omni pipeline**.

- **Stage 0** (`talker`) converts text + prompt audio to speech tokens.
- **Stage 1** (`code2wav`) converts speech tokens + prompt features to waveform via flow matching and HiFiGAN.

Key components live in `vllm_omni/model_executor/models/cosyvoice3/cosyvoice3.py`.

- `CosyVoice3MultiModalProcessor` builds the multimodal inputs:
    - Tokenizes `prompt` and `prompt_text`.
    - Extracts speech tokens and mel features from the prompt audio.
    - Extracts a speaker embedding.
- `CosyVoice3Model` implements both stages:
    - Stage 0 uses `CosyVoice3LM` and outputs speech tokens + conditioning features.
    - Stage 1 runs the flow model (DiT-based CFM) and HiFiGAN to synthesize waveform.

Stage wiring is configured in `vllm_omni/model_executor/stage_configs/cosyvoice3.yaml`.

- Stage 0 emits latent speech tokens.
- Stage 1 consumes them via `custom_process_input_func` and outputs audio.

## Example materials

??? abstract "verify_e2e_cosyvoice.py"
    ``````py
    --8<-- "examples/offline_inference/cosyvoice3/verify_e2e_cosyvoice.py"
    ``````
