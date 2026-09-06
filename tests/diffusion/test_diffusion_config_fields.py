# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ensure diffusion stage YAML configs only use valid OmniDiffusionConfig fields.

Regression test for https://github.com/vllm-project/vllm-omni/issues/2563
"""

from dataclasses import fields
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

try:
    from vllm_omni.diffusion.data import OmniDiffusionConfig
except Exception:
    OmniDiffusionConfig = None


@pytest.mark.skipif(
    OmniDiffusionConfig is None,
    reason="OmniDiffusionConfig could not be imported (missing torch?)",
)
def test_diffusion_stage_configs_only_contain_valid_fields():
    """Diffusion stage engine_args must only contain OmniDiffusionConfig fields.

    Regression test for https://github.com/vllm-project/vllm-omni/issues/2563
    """
    # Scan both main configs and test configs
    repo_root = Path(__file__).parent.parent.parent
    config_dirs = [
        repo_root / "vllm_omni" / "model_executor" / "stage_configs",
    ]
    # Also scan test directories recursively
    test_dir = repo_root / "tests"

    yaml_paths: list[Path] = []
    for config_dir in config_dirs:
        yaml_paths.extend(sorted(config_dir.glob("*.yaml")))
    yaml_paths.extend(sorted(test_dir.rglob("*.yaml")))

    valid_fields = {f.name for f in fields(OmniDiffusionConfig)}
    # model_stage is consumed by the stage init layer, not OmniDiffusionConfig
    valid_fields.add("model_stage")
    # model_arch is consumed by the stage init layer for diffusion model class resolution
    valid_fields.add("model_arch")
    # "quantization" is mapped to "quantization_config" by from_kwargs() backwards-compat
    valid_fields.add("quantization")

    invalid_entries: list[tuple[str, set[str]]] = []
    for yaml_path in yaml_paths:
        with open(yaml_path) as fh:
            config = yaml.safe_load(fh)

        stages = config.get("stage_args", config.get("stages", []))
        for stage in stages:
            if stage.get("stage_type") != "diffusion":
                continue
            engine_args = stage.get("engine_args", {})
            invalid = set(engine_args.keys()) - valid_fields
            if invalid:
                invalid_entries.append((yaml_path.relative_to(repo_root), invalid))

    assert not invalid_entries, "Diffusion stage configs contain fields not in OmniDiffusionConfig:\n" + "\n".join(
        f"  {name}: {sorted(bad)}" for name, bad in invalid_entries
    )


@pytest.mark.skipif(
    OmniDiffusionConfig is None,
    reason="OmniDiffusionConfig could not be imported (missing torch?)",
)
class TestAttnChunkingValidation:
    """Chunking knobs must fail fast when they cannot take effect.

    Built via ``__new__`` with only the attributes _validate_attn_chunking
    reads: the full __post_init__ (port probing, HF metadata) is unrelated
    to this validation.
    """

    @staticmethod
    def _bare(**overrides):
        config = OmniDiffusionConfig.__new__(OmniDiffusionConfig)
        config.diffusion_kv_cache_dtype = None
        config.diffusion_attn_q_chunk = 1
        config.diffusion_attn_head_chunk = 0
        config.diffusion_attn_head_chunk_min_kv = 50000
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    def test_defaults_pass(self):
        self._bare()._validate_attn_chunking()  # no raise

    def test_chunking_with_fp8_passes(self):
        self._bare(
            diffusion_kv_cache_dtype="fp8", diffusion_attn_q_chunk=8, diffusion_attn_head_chunk=2
        )._validate_attn_chunking()  # no raise

    def test_chunking_without_fp8_is_rejected(self):
        with pytest.raises(ValueError, match="require.*diffusion_kv_cache_dtype='fp8'"):
            self._bare(diffusion_attn_q_chunk=8)._validate_attn_chunking()

        with pytest.raises(ValueError, match="require.*diffusion_kv_cache_dtype='fp8'"):
            self._bare(diffusion_attn_head_chunk=2)._validate_attn_chunking()

        with pytest.raises(ValueError, match="require.*diffusion_kv_cache_dtype='fp8'"):
            self._bare(diffusion_attn_head_chunk_min_kv=60000)._validate_attn_chunking()

    @pytest.mark.parametrize(
        "field,value",
        [
            ("diffusion_attn_q_chunk", 0),
            ("diffusion_attn_head_chunk", -1),
            ("diffusion_attn_head_chunk_min_kv", -1),
        ],
    )
    def test_out_of_range_values_are_rejected(self, field: str, value: int):
        with pytest.raises(ValueError, match=field.replace("diffusion_attn_", "")):
            self._bare(diffusion_kv_cache_dtype="fp8", **{field: value})._validate_attn_chunking()
