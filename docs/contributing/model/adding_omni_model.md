# Adding an Omni-Modality Model

This guide walks through the process of adding a new multi-stage model to vLLM-Omni, using **Qwen3-Omni** as a comprehensive example. Qwen3-Omni is a multi-stage omni-modality model that demonstrates the full capabilities of vLLM-Omni's architecture.

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Step-by-Step Implementation](#step-by-step-implementation)
4. [Key Components](#key-components)
5. [Model Registration](#model-registration)
6. [Stage Configuration](#stage-configuration)
7. [Stage Input Processors](#stage-input-processors)
8. [Testing](#testing)
9. [Adding a Model Recipe](#adding-a-model-recipe)
10. [Summary](#summary)

## Overview

vLLM-Omni supports multi-stage model architectures where different stages can run on different devices and process different modalities. The Qwen3-Omni model exemplifies this with three stages:

1. **Thinker Stage**: Multimodal understanding (text + audio + video) → text generation
2. **Talker Stage**: Text embeddings → RVQ codec codes
3. **Code2Wav Stage**: RVQ codes → audio waveform

Each stage is implemented as a separate model class that can be configured independently.

## Directory Structure

When adding a new model, you'll need to create the following structure:

```
vllm_omni/model_executor/models/
└── your_model_name/              # Model directory (e.g., qwen3_omni)
    ├── __init__.py               # Exports main model class
    ├── your_model.py             # Main unified model class
    ├── your_model_stage1_implementation.py      # Stage 1 implementation (e.g., thinker)
    ├── your_model_stage2_implementation.py      # Stage 2 implementation (e.g., talker)
    └── your_model_stage3_implementation.py      # Stage 3 implementation (e.g., code2wav)
    └── ... maybe other stage implementations

vllm_omni/model_executor/stage_input_processors/
└── your_model_name.py            # Stage transition processors

vllm_omni/model_executor/stage_configs/
└── your_model_name.yaml          # Stage configuration file
```

## Step-by-Step Implementation

### Step 1: Create the Model Directory

Create a new directory under `vllm_omni/model_executor/models/`

### Step 2: Implement Stage Components

For Qwen3-Omni, we have three stage components:

#### 2.1 Thinker Stage (`qwen3_omni_moe_thinker.py`)

The thinker stage handles multimodal understanding. Key features:

- Inherits from base Qwen3 MoE model in vLLM, using vLLM fused ops & page attn to accelerate
- Implements multimodal processing interfaces
- Handles audio, video, and image inputs
- Generates text outputs

```python
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM

class Qwen3OmniMoeThinkerForConditionalGeneration(
    Qwen3MoeForCausalLM,
    SupportsMultiModal,
    SupportsPP
):
    """Thinker stage: multimodal understanding → text generation."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Initialize base model
        # Set up multimodal processors
        # Configure audio/video/image encoders
        pass
```

#### 2.2 Talker Stage (`qwen3_omni_moe_talker.py`)

The talker stage converts text embeddings to codec codes:

```python
class Qwen3OmniMoeTalkerForConditionalGeneration(
    Qwen3MoeForCausalLM,
    SupportsPP
):
    """Talker stage: text embeddings → RVQ codec codes."""

    def __init__(self, vllm_config, talker_config, prefix):
        # Initialize base model
        # Replace LM head with codec head
        # Set up text projection from thinker
        pass
```

#### 2.3 Code2Wav Stage (`qwen3_omni_code2wav.py`)

The code2wav stage generates audio waveforms:

```python
class Qwen3OmniMoeCode2Wav(nn.Module):
    """Code2Wav stage: RVQ codes → audio waveform."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Initialize audio decoder
        # Set up codec processing
        pass
```

### Step 3: Implement the Unified Model Class

The main model class (`qwen3_omni.py`) orchestrates all stages:

```python
@MULTIMODAL_REGISTRY.register_processor(
    Qwen3OmniMoeThinkerMultiModalProcessor,
    info=Qwen3OmniMoeThinkerProcessingInfo,
    dummy_inputs=Qwen3OmniMoeThinkerDummyInputsBuilder,
)
class Qwen3OmniMoeForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, Qwen3OmniMoeConditionalGenerationMixin
):
    """
    Unified Qwen3 Omni MoE model combining thinker, talker, and code2wav.

    Architecture:
    - Thinker: Multimodal understanding (text + audio + video) → text generation
    - Talker: Text embeddings → RVQ codec codes
    - Code2Wav: RVQ codes → audio waveform

    Usage:
        Set `model_stage` in vllm_config to one of: "thinker", "talker", "code2wav"
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.have_multimodal_outputs = True
        config: Qwen3OmniMoeConfig = vllm_config.model_config.hf_config

        # Determine which stage to initialize
        self.model_stage = vllm_config.model_config.model_stage

        if self.model_stage == "thinker":
            # Initialize thinker model
            thinker_vllm_config = vllm_config.with_hf_config(
                config.thinker_config,
                architectures=["Qwen3OmniMoeThinkerForConditionalGeneration"]
            )
            self.thinker = init_vllm_registered_model(
                vllm_config=thinker_vllm_config,
                prefix=maybe_prefix(prefix, "thinker"),
                hf_config=config.thinker_config,
                architectures=["Qwen3OmniMoeThinkerForConditionalGeneration"],
            )
            self.model = self.thinker

        elif self.model_stage == "talker":
            # Initialize talker model
            talker_vllm_config = vllm_config.with_hf_config(
                config.talker_config,
                architectures=["Qwen3OmniMoeTalkerForConditionalGeneration"]
            )
            self.talker = init_vllm_registered_model(
                vllm_config=talker_vllm_config,
                prefix=maybe_prefix(prefix, "talker"),
                hf_config=config.talker_config,
                architectures=["Qwen3OmniMoeTalkerForConditionalGeneration"],
            )
            self.model = self.talker

        elif self.model_stage == "code2wav":
            # Initialize code2wav model
            code2wav_vllm_config = vllm_config.with_hf_config(
                config.code2wav_config,
                architectures=["Qwen3OmniMoeCode2Wav"]
            )
            self.code2wav = init_vllm_registered_model(
                vllm_config=code2wav_vllm_config,
                prefix=maybe_prefix(prefix, "code2wav"),
                hf_config=config.code2wav_config,
                architectures=["Qwen3OmniMoeCode2Wav"],
            )
            self.model = self.code2wav
        else:
            raise ValueError(
                f"Invalid model_stage: {self.model_stage}. "
                f"Must be one of: 'thinker', 'talker', 'code2wav'"
            )
```

#### Key Methods to Implement

1. **`forward()`**: Handles the forward pass for each stage
2. **`embed_input_ids()`**: Embeds input token IDs
3. **`embed_multimodal()`**: Processes multimodal inputs (if applicable)
4. **`compute_logits()`**: Computes logits from hidden states
5. **`load_weights()`**: Loads model weights with proper prefixing of different stages

### Step 4: Create `__init__.py`

Export the main model class:

```python
# vllm_omni/model_executor/models/qwen3_omni/__init__.py
from .qwen3_omni import Qwen3OmniMoeForConditionalGeneration

__all__ = ["Qwen3OmniMoeForConditionalGeneration"]
```

## Key Components

### 1. Model Interfaces

Your model should implement the appropriate interfaces:

- **`SupportsMultiModal`**: For models that process multimodal inputs
- **`SupportsPP`**: For models that support pipeline parallelism
- **`SupportsMRoPE`**: For models using multi-dimensional RoPE (if applicable)

### 2. Multimodal Registration

If your model processes multimodal inputs, register it with the multimodal registry:

```python
@MULTIMODAL_REGISTRY.register_processor(
    YourMultiModalProcessor,
    info=YourProcessingInfo,
    dummy_inputs=YourDummyInputsBuilder,
)
class YourModel(nn.Module, SupportsMultiModal):
    pass
```

### 3. Weight Loading

Implement `load_weights()` to handle weight loading with proper prefixing:

```python
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    """Load weights for all components of the omni model."""
    loaded_weights = set()
    thinker_weights = []
    talker_weights = []
    code2wav_weights = []

    # Separate weights by component
    for k, v in weights:
        if k.startswith("thinker."):
            thinker_weights.append((k, v))
        elif k.startswith("talker."):
            talker_weights.append((k, v))
        elif k.startswith("code2wav."):
            code2wav_weights.append((k, v))

    # Load each component's weights
    if self.thinker and thinker_weights:
        thinker_loaded = self.thinker.load_weights(thinker_weights)
        thinker_loaded = add_prefix_to_loaded_weights(thinker_loaded, "thinker")
        loaded_weights.update(thinker_loaded)

    # Similar for talker and code2wav...

    return loaded_weights
```

### 4. Output Format

Use `OmniOutput` for stage outputs:

```python
from vllm_omni.model_executor.models.output_templates import OmniOutput

# In forward method
return OmniOutput(
    text_hidden_states=hidden_states,
    multimodal_outputs={"additional_data": data},
    next_token_id=next_token_id,
)
```

## Model Registration

Register your model in `vllm_omni/model_executor/models/registry.py`:

```python
_OMNI_MODELS = {
    # ... existing models ...

    # Your new model
    "YourModelForConditionalGeneration": (
        "your_model_name",        # Module folder name
        "your_model",             # Module file name (without .py)
        "YourModelForConditionalGeneration",  # Class name
    ),
    "YourModelThinkerForConditionalGeneration": (
        "your_model_name",
        "your_model_thinker",
        "YourModelThinkerForConditionalGeneration",
    ),
    # ... other stages ...
}
```

The registry uses lazy loading, so the model class is imported only when needed.

## Stage Configuration

Create a YAML configuration file in `vllm_omni/model_executor/stage_configs/`. For a complete example, see the [Qwen3-Omni configuration file](gh-file:vllm_omni/model_executor/stage_configs/qwen3_omni_moe.yaml).

### Key Configuration Fields

- **`model_stage`**: Which stage to run ("thinker", "talker", "code2wav", etc.)
- **`model_arch`**: The model architecture name (must match registry)
- **`engine_input_source`**: List of stage IDs that provide input to this stage
- **`custom_process_input_func`**: Function to process inputs from previous stages
- **`final_output`**: Whether this stage produces the final output (True/False)
- **`final_output_type`**: Type of final output ("text", "audio", "image", etc.)

## Stage Input Processors

Stage transitions are the mechanism by which outputs from one stage are converted into inputs for the next stage. This section explains where and how stage transitions occur.

### Where Stage Transitions Are Called

Stage transitions happen automatically in the orchestrator (`OmniLLM` class) during the generation loop. Here's the detailed flow:

1. **Location**: `vllm_omni/entrypoints/omni_llm.py` in the `_run_generation()` method
2. **Trigger**: When a stage completes processing and produces outputs
3. **Execution Flow**:
   ```python
   # In omni_llm.py, _run_generation() method (around line 345-460)

   # Main orchestrator loop polls each stage for completed requests
   for stage_id, stage in enumerate(self.stage_list):
       result = stage.try_collect()  # Get completed request
       if result is None:
           continue

       # Store outputs from this stage
       engine_outputs = _load(result, obj_key="engine_outputs", shm_key="engine_outputs_shm")
       stage.set_engine_outputs(engine_outputs)

       # Check if there's a next stage to forward to
       next_stage_id = stage_id + 1
       if next_stage_id < len(self.stage_list):
           next_stage: OmniStage = self.stage_list[next_stage_id]

           # THIS IS WHERE STAGE TRANSITION HAPPENS
           next_inputs = next_stage.process_engine_inputs(
               self.stage_list,
               [request_id_to_prompt[req_id]]
           )

           # Submit to next stage
           task = {
               "type": OmniStageTaskType.GENERATE,
               "request_id": req_id,
               "engine_inputs": next_inputs[0],
               "sampling_params": sampling_params_list[next_stage_id],
           }
           next_stage.submit(task)
   ```

### How Stage Transitions Work

The stage transition process follows these steps:

1. **Stage Completion**: When a stage finishes processing a request, it stores outputs via `stage.set_engine_outputs(engine_outputs)`

2. **Transition Detection**: The orchestrator checks if there's a next stage and calls `process_engine_inputs()` on it

3. **Input Processing**: The `process_engine_inputs()` method in `OmniStage` (`omni_stage.py`) handles the transition:
   ```python
   def process_engine_inputs(
       self, stage_list: list[Any], prompt: OmniTokensPrompt | TextPrompt = None
   ) -> list[OmniTokensPrompt | TextPrompt]:
       """Process engine inputs for this stage from upstream stage outputs."""

       if self.custom_process_input_func is None:
           # Default behavior: pass token IDs directly
           # Extract outputs from source stage
           source_stage_id = self.engine_input_source[0]
           source_outputs = stage_list[source_stage_id].engine_outputs
           # ... create OmniTokensPrompt from token_ids ...
       else:
           # Custom transition function (YOUR CODE HERE)
           return self.custom_process_input_func(
               stage_list,
               self.engine_input_source,
               prompt,
               self.requires_multimodal_data
           )
   ```
   - If `custom_process_input_func` is configured, it calls that function
   - Otherwise, it uses default behavior (passing token IDs directly)

4. **Custom Function Execution**: Your custom function receives:
   - `stage_list`: List of all stage objects (to access upstream stage outputs)
   - `engine_input_source`: List of source stage IDs (e.g., `[0]` for stage 0)
   - `prompt`: Original prompt data (for preserving multimodal data)
   - `requires_multimodal_data`: Whether multimodal data is required

5. **Output Format**: The function must return a list of `OmniTokensPrompt` objects ready for the next stage

### Data Structures in Stage Transitions

Understanding the data structures is crucial for implementing stage transitions:

**Input to your function:**
- `stage_list[source_stage_id].engine_outputs`: List of `EngineCoreOutput` objects
  - Each contains `outputs`: List of `RequestOutput` objects
  - Each `RequestOutput` has:
    - `token_ids`: Generated token IDs
    - `multimodal_output`: Dict with keys like `"code_predictor_codes"`, etc.
      - These are the hidden states or intermediate outputs from the model's forward pass
    - `prompt_token_ids`: Original prompt token IDs

**Output from your function:**
- Must return `list[OmniTokensPrompt]` where each `OmniTokensPrompt` contains:
  - `prompt_token_ids`: List[int] - Token IDs for the next stage
  - `additional_information`: Dict[str, Any] - Optional metadata (e.g., embeddings, hidden states)
  - `multi_modal_data`: Optional multimodal data if needed

### How Model Outputs Are Stored

The model's `forward()` method returns an `OmniOutput` object that contains:
- `text_hidden_states`: Final hidden states for text generation
- `multimodal_outputs`: Dict containing intermediate outputs

These outputs are captured during the forward pass and stored in `multimodal_output` with specific keys:

```python
# In your model's forward() method (e.g., qwen3_omni.py)
def forward(self, ...):
    # ... processing ...

    # For thinker stage: capture embeddings and hidden states
    multimodal_outputs = {
        "0": captured_embeddings,      # Layer 0 embeddings
        "24": captured_hidden_states,  # Layer 24 hidden states
        "tts_bos_embed": tts_bos_embed,
        "tts_eos_embed": tts_eos_embed,
        # ... other intermediate outputs ...
    }

    return OmniOutput(
        text_hidden_states=hidden_states,
        multimodal_outputs=multimodal_outputs,
    )
```

These keys are then accessible in your stage transition function:
```python
# In stage_input_processors/qwen3_omni.py
thinker_embeddings = output.multimodal_output["0"]  # Access by key
thinker_hidden_states = output.multimodal_output["24"]
```

### Key Points

1. **Accessing Upstream Outputs**: Use `stage_list[source_stage_id].engine_outputs` to get outputs from the source stage
2. **Extracting Data**: Access `output.multimodal_output[key]` to get specific hidden states or intermediate results
   - Keys are defined by your model's `forward()` method when it creates `multimodal_outputs`
3. **Device Management**: Move tensors to appropriate devices (CPU for serialization, GPU for processing)
4. **Shape Transformations**: Reshape tensors as needed for the next stage (e.g., flattening codec codes)
5. **Batch Handling**: Process each request in the batch separately and return a list

### Complete Flow Diagram

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/vllm-omni-dataflow-between-stages.png">
    <img alt="Data Flow between stages" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/vllm-omni-dataflow-between-stages.png" width=55%>
  </picture>
</p>

### Implementation Example

Create stage transition processors in `vllm_omni/model_executor/stage_input_processors/your_model_name.py`:

```python
# qwen3_omni.py

def thinker2talker(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """
    Process thinker outputs to create talker inputs.

    Args:
        stage_list: List of stage objects
        engine_input_source: Source stage IDs (typically [0] for thinker)
        prompt: Original prompt data

    Returns:
        List of OmniTokensPrompt for talker stage
    """
    source_stage_id = engine_input_source[0]
    thinker_outputs = stage_list[source_stage_id].engine_outputs
    talker_inputs = []

    for thinker_output in thinker_outputs:
        output = thinker_output.outputs[0]
        # Extract thinker embeddings and hidden states
        thinker_embeddings = output.multimodal_output["0"].float().clone().detach().cuda()
        thinker_hidden_states = output.multimodal_output["24"].float().clone().detach().cuda()

        info = {
            "thinker_embeddings": thinker_embeddings,
            "thinker_hidden_states": thinker_hidden_states,
            "thinker_sequences": thinker_output.prompt_token_ids + output.token_ids,
            "thinker_input_ids": thinker_output.prompt_token_ids,
        }

        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * computed_length,
                additional_information=info,
                multi_modal_data=None,
            )
        )

    return talker_inputs


def talker2code2wav(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """
    Process talker outputs to create code2wav inputs.
    """
    source_stage_id = engine_input_source[0]
    talker_outputs = stage_list[source_stage_id].engine_outputs
    code2wav_inputs = []

    for talker_output in talker_outputs:
        output = talker_output.outputs[0]
        # Extract codec codes
        codec_codes = (
            output.multimodal_output["code_predictor_codes"]
            .to(torch.long)
            .transpose(0, 1)
            .cpu()
            .to(torch.long)
            .reshape(-1)
            .tolist()
        )

        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes,
                multi_modal_data=None,
            )
        )

    return code2wav_inputs
```

## Testing

For comprehensive testing guidelines, please refer to the [Test File Structure and Style Guide](../tests/tests_style.md).

## Adding a Model Recipe

After implementing and testing your model, please add a model recipe to the [vllm-project/recipes](https://github.com/vllm-project/recipes) repository. This helps other users understand how to use your model with vLLM-Omni.

### What to Include

Your recipe should include:

1. **Model Overview**: Brief description of the model and its capabilities
2. **Installation Instructions**: Step-by-step setup instructions including:
   - Installing vllm-omni and dependencies
   - Installing any additional required packages (e.g., xformers, diffusers)
   - Any version requirements
3. **Usage Examples**: Command-line examples demonstrating how to run the model
4. **Configuration Details**: Important configuration parameters and their meanings

### Example

For reference, see the [LongCat recipe example](https://github.com/vllm-project/recipes/pull/179) which demonstrates the expected format and structure.

### Recipe Location

Create your recipe file in the appropriate directory structure:
- For organization-specific models: `OrganizationName/ModelName.md`
- For general models: `ModelName.md`

The recipe should be a Markdown file that provides clear, reproducible instructions for users to get started with your model.

## Summary

Adding a new model to vLLM-Omni involves:

1. **Create model directory structure** with stage implementations
2. **Implement unified model class** that orchestrates stages
3. **Register model** in `registry.py`
4. **Create stage configuration** YAML file
5. **Implement stage input processors** for stage transitions
6. **Write tests** to verify functionality
7. **Add model recipe** to the [vllm-project/recipes](https://github.com/vllm-project/recipes) repository (see [Adding a Model Recipe](#adding-a-model-recipe) section)

### Qwen3-Omni Reference Files

For a complete reference implementation, see:

- **Main model**: `vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py`
- **Thinker**: `vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py`
- **Talker**: `vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_talker.py`
- **Code2Wav**: `vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_code2wav.py`
- **Stage config**: `vllm_omni/model_executor/stage_configs/qwen3_omni_moe.yaml`
- **Input processors**: `vllm_omni/model_executor/stage_input_processors/qwen3_omni.py`
- **Registry**: `vllm_omni/model_executor/models/registry.py`
- **Testing**: `vllm_omni/tests/e2e/offline_inference/test_qwen3_omni.py`

For more information, see:
- [Architecture Overview](../../design/architecture_overview.md)
- [Supported Models](../../models/supported_models.md)
- [Stage Configuration Guide](../../configuration/stage_configs.md)
