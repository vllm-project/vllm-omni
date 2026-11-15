# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Offline inference regression tests: explicit CLI parallel-knob forwarding.

Covers the argparse defaults -> ``build_parallel_knob_kwargs`` -> ``main()``'s
``omni_kwargs`` assembly in examples/offline_inference/text_to_image/text_to_image.py.
Three invariants:

1. The shipped argparse must default the parallel knobs to ``None`` so unset
   flags do not enter ``omni_kwargs`` and a deploy YAML value stays in effect.
2. Explicitly supplied values equal to the old argparse defaults (e.g.
   ``--ulysses-degree 1``, ``--ulysses-mode strict``) must still be forwarded
   so they can override the deploy YAML, instead of being silently dropped
   (review feedback on PR #7293).
3. The AR grid ``max_tokens`` budget must stay owned by the shared
   ``ar_grid_max_tokens`` helper used by both this example and the online
   serving path (review feedback on PR #7293).

Both invariants are asserted against the real script (its ``parse_args()`` and
its ``build_parallel_knob_kwargs``), not a re-implemented copy. Invariant 3 is
asserted directly against ``ar_grid_max_tokens`` itself.
"""

import shlex
import sys
from pathlib import Path

import pytest

_script_dir = str(Path(__file__).parent.parent.parent.parent / "examples" / "offline_inference" / "text_to_image")
sys.path.insert(0, _script_dir)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_KNOBS = (
    "ulysses_degree",
    "ulysses_mode",
    "ring_degree",
    "cfg_parallel_size",
    "vae_patch_parallel_size",
)


def _parse_real_args(cli_flags: list[str]):
    """Run the example's real argparse against a synthetic argv."""
    import text_to_image as t2i

    saved = sys.argv
    sys.argv = ["text_to_image.py", *cli_flags]
    try:
        return t2i.parse_args()
    finally:
        sys.argv = saved


class TestParallelKnobForwarding:
    """Drive the example's real argparse + real forwarding helper."""

    def test_unset_knobs_forward_nothing(self):
        """No CLI flags -> no keys, so the deploy YAML values stay in effect."""
        import text_to_image as t2i

        args = _parse_real_args([])
        assert t2i.build_parallel_knob_kwargs(args) == {}

    def test_explicit_default_equivalent_values_are_forwarded(self):
        """--ulysses-degree 1 / --ulysses-mode strict must override the YAML."""
        import text_to_image as t2i

        args = _parse_real_args(["--ulysses-degree", "1", "--ulysses-mode", "strict"])
        assert t2i.build_parallel_knob_kwargs(args) == {
            "ulysses_degree": 1,
            "ulysses_mode": "strict",
        }

        args = _parse_real_args(["--ring-degree", "1", "--cfg-parallel-size", "1", "--vae-patch-parallel-size", "1"])
        assert t2i.build_parallel_knob_kwargs(args) == {
            "ring_degree": 1,
            "cfg_parallel_size": 1,
            "vae_patch_parallel_size": 1,
        }

    def test_non_default_values_are_forwarded(self):
        import text_to_image as t2i

        args = _parse_real_args(
            [
                "--ulysses-degree",
                "2",
                "--ulysses-mode",
                "advanced_uaa",
                "--ring-degree",
                "2",
                "--cfg-parallel-size",
                "2",
                "--vae-patch-parallel-size",
                "2",
            ]
        )
        assert t2i.build_parallel_knob_kwargs(args) == {
            "ulysses_degree": 2,
            "ulysses_mode": "advanced_uaa",
            "ring_degree": 2,
            "cfg_parallel_size": 2,
            "vae_patch_parallel_size": 2,
        }

    def test_script_defaults_are_none(self):
        """The shipped argparse must default to None so unset == defer to YAML."""
        import text_to_image as t2i

        parser = t2i.build_parser()
        defaults = {key: parser.get_default(key) for key in _KNOBS}
        assert defaults == {key: None for key in _KNOBS}


class TestARGridMaxTokens:
    """The AR grid budget formula must stay shared between serving and offline.

    ``ar_grid_max_tokens`` (vllm_omni.diffusion.utils.param_utils) is the
    single owner of the ``ar_height * (ar_width + 1) + 1`` contract used by
    both the online serving path (serving_chat.py) and this example script,
    so a grid-contract change cannot desynchronize one copy (review feedback
    on PR #7293).
    """

    def test_grid_budget_matches_formula(self):
        from vllm_omni.diffusion.utils.param_utils import ar_grid_max_tokens

        assert ar_grid_max_tokens(64, 64) == 64 * 65 + 1
        assert ar_grid_max_tokens(32, 48) == 48 * 33 + 1

    def test_grid_budget_none_for_absent_or_invalid_grid(self):
        from vllm_omni.diffusion.utils.param_utils import ar_grid_max_tokens

        # Callers treat None as "keep the existing max_tokens default".
        assert ar_grid_max_tokens(0, 0) is None
        assert ar_grid_max_tokens(0, 64) is None
        assert ar_grid_max_tokens(64, 0) is None


def test_documented_qwen21_attention_deploy_config_reaches_omni(monkeypatch, tmp_path):
    """Parse the documented command and load its YAML without loading a model."""
    import text_to_image as t2i

    from vllm_omni.config.config_factory import StageConfigFactory

    repo_root = Path(__file__).resolve().parents[3]
    doc = (repo_root / "docs/user_guide/quantization/quantized_kvcache.md").read_text()
    section = doc.split("### Qwen-Image-2.1 on Ascend", 1)[1].split("## Validation and Notes", 1)[0]
    yaml_text = section.split("```yaml\n", 1)[1].split("```", 1)[0]
    command = section.split("```bash\n", 1)[1].split("```", 1)[0]
    deploy_path = tmp_path / "qwen21_mxfp8.yaml"
    deploy_path.write_text(yaml_text)
    monkeypatch.chdir(tmp_path)
    argv = shlex.split(command.replace("\\\n", " "))
    monkeypatch.setattr(sys, "argv", argv[1:])
    generator = t2i.torch.Generator
    monkeypatch.setattr(t2i.torch, "Generator", lambda **kwargs: generator(device="cpu"))

    class ReachedOmniError(Exception):
        pass

    def capture_omni(**kwargs):
        assert Path(kwargs["deploy_config"]).resolve() == deploy_path
        runtime_kwargs = {key: value for key, value in kwargs.items() if key not in ("model", "deploy_config")}
        runtime_kwargs["model_class_name"] = "QwenImage21Pipeline"
        config = (
            StageConfigFactory.create_typed_default_diffusion(
                kwargs["model"], runtime_kwargs, deploy_config_path=kwargs["deploy_config"]
            )
            .stage_by_id(0)
            .diffusion_config
        )
        assert config.diffusion_attention_config.default.backend == "FLASH_ATTN"
        assert config.diffusion_kv_cache_dtype == "mxfp8"
        assert config.diffusion_kv_cache_skip_step_indices == {0}
        raise ReachedOmniError

    monkeypatch.setattr(t2i, "Omni", capture_omni)
    with pytest.raises(ReachedOmniError):
        t2i.main()
