# PR #13 - Action Items for Authors

## Critical (Must Fix Before Merge)

### 1. Code Quality
- [ ] Fix import ordering across all modified files (use isort/ruff)
  - Files: `config/__init__.py`, `model_executor/stage_input_processors/qwen2_5_omni.py`
  - Apply PEP8 order: stdlib → third-party → local
  
- [ ] Add trailing newlines to all files
  - Files: `engine/arg_utils.py`, `inputs/data.py`, `outputs.py`, `entrypoints/utils.py`, etc.

- [ ] Replace assertions with proper validation
  ```python
  # In omni_llm.py - generate()
  - assert len(sampling_params_list) == len(self.stage_list), "..."
  + if len(sampling_params_list) != len(self.stage_list):
  +     raise ValueError(f"Expected {len(self.stage_list)} sampling params, got {len(sampling_params_list)}")
  ```

### 2. Error Handling & Validation
- [ ] Add input validation in `stage.py::process_engine_inputs()`
  ```python
  if stage_id > 0 and len(self.engine_input_source) == 0:
      raise ValueError(f"Non-initial stage {stage_id} requires input_source")
  ```

- [ ] Add bounds checking in `qwen2_5_omni.py::thinker2talker()`
  ```python
  if not engine_input_source:
      raise ValueError("engine_input_source cannot be empty")
  source_stage_id = engine_input_source[0]
  if source_stage_id >= len(stage_list):
      raise IndexError(f"Invalid stage_id: {source_stage_id}")
  if stage_list[source_stage_id].engine_outputs is None:
      raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")
  ```

### 3. Naming & Conventions
- [ ] Rename worker files for consistency
  - `vllm_omni/worker/AR_gpu_worker.py` → `ar_gpu_worker.py`
  - `vllm_omni/worker/gpu_diffusion_worker.py` (verify consistency)
  - Update YAML config references accordingly

### 4. Configuration
- [ ] Resolve TODO comments in `qwen2_5_omni.yaml`
  ```yaml
  - enforce_eager: true  # need to discuss
  - engine_output_type: latent  # change the param name,such as pooling_output
  ```
  Either remove comments and commit to values, or create follow-up issues

- [ ] Remove extra blank line in `qwen2_5_omni.yaml` (line 27, reviewer comment)

### 5. Testing (Blocking)
- [ ] Add minimum test coverage:
  - `tests/entrypoints/test_omni_llm.py`:
    ```python
    def test_load_stage_configs_from_model()
    def test_stage_initialization()
    def test_generate_single_stage()
    def test_generate_multi_stage()
    ```
  - `tests/entrypoints/test_stage.py`:
    ```python
    def test_stage_input_processing()
    def test_stage_with_custom_processor()
    ```
  - `tests/config/test_omni_model_config.py`:
    ```python
    def test_omni_model_config_creation()
    def test_engine_args_inheritance()
    ```

### 6. Documentation
- [ ] Add docstrings with parameter descriptions
  - All public methods in `OmniLLM`, `StageLLM`, `Stage`
  - Example:
    ```python
    def process_engine_inputs(self, stage_list: List[Stage], 
                             prompt: Optional[Union[OmniTokensPrompt, TextPrompt]] = None
                             ) -> List[Union[OmniTokensPrompt, TextPrompt]]:
        """
        Process inputs for this stage from previous stage outputs.
        
        Args:
            stage_list: List of all stages in the pipeline
            prompt: Original user prompts (used for multi-modal data)
            
        Returns:
            List of processed inputs ready for this stage's engine
            
        Raises:
            ValueError: If stage requires input but input_source is empty
        """
    ```

- [ ] Add usage example in docstring or separate file
  - Show how to use `OmniLLM` with Qwen2.5-Omni
  - Example in issue #10 is good - add to README or docs/

---

## Important (Should Fix Soon)

### 7. Architecture Clarifications
- [ ] Consider renaming for clarity (discuss with team):
  - `OmniLLM` → `OmniPipeline` or `MultiStageLLM` (since it doesn't inherit from LLM)
  - Document relationship: `OmniLLM` contains multiple `StageLLM` instances
  - Add class-level docstring explaining architecture

- [ ] Add type hints for stage processor protocol
  ```python
  # In new file: vllm_omni/entrypoints/protocols.py
  from typing import Protocol, List, Union
  
  class StageInputProcessor(Protocol):
      def __call__(self, 
                   stage_list: List[Any],
                   engine_input_source: List[int],
                   prompt: Optional[Any] = None) -> List[Any]:
          ...
  ```

### 8. Security
- [ ] Add validation for dynamic imports in `stage.py`
  ```python
  ALLOWED_PROCESSOR_MODULES = [
      'vllm_omni.model_executor.stage_input_processors',
  ]
  
  def _validate_processor_module(module_path: str) -> None:
      if not any(module_path.startswith(allowed) for allowed in ALLOWED_PROCESSOR_MODULES):
          raise ValueError(f"Processor module {module_path} not in allowed list")
  ```

### 9. Performance
- [ ] Add defensive `.to(device)` instead of hard-coded `.cuda()`
  ```python
  # In qwen2_5_omni.py
  - thinker_hidden_states = output.multimodal_output["latent"].clone().detach().cuda()
  + device = thinker_hidden_states.device  # or get from config
  + thinker_hidden_states = output.multimodal_output["latent"].clone().detach().to(device)
  ```

### 10. Code Organization
- [ ] Set up pre-commit hooks
  ```yaml
  # .pre-commit-config.yaml
  repos:
    - repo: https://github.com/astral-sh/ruff-pre-commit
      rev: v0.1.0
      hooks:
        - id: ruff
          args: [--fix, --exit-non-zero-on-fix]
    - repo: https://github.com/pycqa/isort
      rev: 5.12.0
      hooks:
        - id: isort
    - repo: https://github.com/pre-commit/pre-commit-hooks
      rev: v4.5.0
      hooks:
        - id: end-of-file-fixer
        - id: trailing-whitespace
  ```

---

## Nice-to-Have (Future)

### 11. Enhancements
- [ ] Add config caching for repeated model initialization
- [ ] Add logging for stage transitions
- [ ] Consider memory-efficient tensor passing (reduce clones)
- [ ] Add architecture diagram to documentation
- [ ] Add type validation for model_stage enum
  ```python
  from typing import Literal
  model_stage: Literal["thinker", "talker", "code2wav"]
  ```

### 12. Follow-up Issues to Create
- [ ] Comprehensive integration tests (Phase 2)
- [ ] Performance benchmarking framework
- [ ] Parallel stage execution exploration
- [ ] Better error messages with recovery suggestions

---

## Estimated Effort

- Critical items (1-9): **6-8 hours**
  - Tests: 3-4 hours
  - Code quality fixes: 1-2 hours
  - Error handling: 1 hour
  - Documentation: 1-2 hours

- Important items (10-11): **2-3 hours**

**Total to merge readiness: ~1 working day**

---

## Review Comments Status

✅ Addressed:
- Import order issue (comment by hsliuustc0106)
- Worker class naming (comment by hsliuustc0106)
- Blank line removal (comment by hsliuustc0106)
- Stage naming clarification (comment by Gaohan123)

⏳ Partially Addressed:
- Relationship between OmniLLM/StageLLM/LLM (comment by hsliuustc0106)
  - Author explained but consider renaming for clarity

🔴 Not Yet Addressed:
- Add docstring for @config decorator (comment by gemini-code-assist)
- Fix default value in --model-stage help (comment by gemini-code-assist)
- Validation in generate() (comment by gemini-code-assist)
- Empty engine_input_source handling (comment by gemini-code-assist)
- Test coverage (no comments but critical omission)

---

## Checklist for PR Author

Before requesting re-review:
- [ ] All P0 (Critical) items completed
- [ ] Tests added and passing
- [ ] Pre-commit hooks configured
- [ ] Documentation updated
- [ ] YAML TODOs resolved
- [ ] All review comments addressed or responded to
- [ ] Self-review completed using this checklist

