"""
Offline inference regression tests: explicit CLI parallel-knob forwarding.

Covers the argparse defaults -> ``build_parallel_knob_kwargs`` -> ``main()``'s
``omni_kwargs`` assembly in examples/offline_inference/text_to_image/text_to_image.py.
Two invariants:

1. The shipped argparse must default the parallel knobs to ``None`` so unset
   flags do not enter ``omni_kwargs`` and a deploy YAML value stays in effect.
2. Explicitly supplied values equal to the old argparse defaults (e.g.
   ``--ulysses-degree 1``, ``--ulysses-mode strict``) must still be forwarded
   so they can override the deploy YAML, instead of being silently dropped
   (review feedback on PR #7293).

Both invariants are asserted against the real script (its ``parse_args()`` and
its ``build_parallel_knob_kwargs``), not a re-implemented copy.
"""

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

        args = _parse_real_args(
            ["--ring-degree", "1", "--cfg-parallel-size", "1", "--vae-patch-parallel-size", "1"]
        )
        assert t2i.build_parallel_knob_kwargs(args) == {
            "ring_degree": 1,
            "cfg_parallel_size": 1,
            "vae_patch_parallel_size": 1,
        }

    def test_non_default_values_are_forwarded(self):
        import text_to_image as t2i

        args = _parse_real_args(
            [
                "--ulysses-degree", "2",
                "--ulysses-mode", "advanced_uaa",
                "--ring-degree", "2",
                "--cfg-parallel-size", "2",
                "--vae-patch-parallel-size", "2",
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
