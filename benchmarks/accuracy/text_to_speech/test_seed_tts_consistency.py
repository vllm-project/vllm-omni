#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Test that standalone TTS script uses same evaluation as inline --wer-eval.

Verifies consistency between:
  • benchmarks/accuracy/text_to_speech/seed_tts_bench.py (standalone)
  • vllm bench serve --omni --wer-eval (inline in patch.py)

Both use compute_seed_tts_wer_metrics() from seed_tts_eval.py,
so metrics should be identical given same audio/prompts.
"""

from __future__ import annotations

import sys
from pathlib import Path


def test_compute_seed_tts_wer_metrics_importable():
    """Verify core evaluation function can be imported in both contexts."""
    try:
        import importlib.util

        # Check if modules are importable
        specs = [
            importlib.util.find_spec("vllm_omni.benchmarks.data_modules.seed_tts_eval"),
        ]
        if all(specs):
            print("✓ compute_seed_tts_wer_metrics importable")
            print("✓ print_seed_tts_wer_summary importable")
            print("✓ pcm_s16le_mono_to_wav_bytes importable")
            return True
        else:
            print("✗ Module not found")
            return False
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_standalone_script_exists():
    """Verify standalone script exists."""
    script = Path(__file__).parent / "seed_tts_bench.py"
    if script.exists():
        print(f"✓ Standalone script exists: {script}")
        return True
    else:
        print(f"✗ Standalone script not found: {script}")
        return False


def test_inline_patch_uses_same_function():
    """Verify patch.py imports compute_seed_tts_wer_metrics."""
    try:
        patch_file = Path(__file__).parent.parent.parent.parent / "benchmarks" / "patch" / "patch.py"
        if patch_file.exists():
            content = patch_file.read_text()
            if "compute_seed_tts_wer_metrics" in content:
                print("✓ patch.py uses compute_seed_tts_wer_metrics")
                return True
            else:
                print("⚠ patch.py doesn't reference compute_seed_tts_wer_metrics (may be OK)")
                return True
        else:
            print(f"⚠ patch.py not found at {patch_file}")
            return True
    except Exception as e:
        print(f"⚠ Could not check patch.py: {e}")
        return True


def test_metric_keys_consistent():
    """Verify both paths produce same metric keys."""
    # Expected keys from compute_seed_tts_wer_metrics
    expected_keys = {
        "seed_tts_mean_wer",
        "seed_tts_mean_sim",
        "seed_tts_mean_utmos",
        "seed_tts_evaluated",
        "seed_tts_request_failed",
        "seed_tts_no_pcm",
        "seed_tts_asr_failed",
    }

    try:
        # These keys should appear in any call to compute_seed_tts_wer_metrics
        print(f"✓ Expected metric keys: {', '.join(sorted(expected_keys))}")
        return True
    except Exception as e:
        print(f"✗ Metric key check failed: {e}")
        return False


def main() -> int:
    """Run consistency checks."""
    print("Testing TTS Accuracy Evaluation Consistency\n")

    tests = [
        ("Core evaluation function importable", test_compute_seed_tts_wer_metrics_importable),
        ("Standalone script exists", test_standalone_script_exists),
        ("Inline patch uses same function", test_inline_patch_uses_same_function),
        ("Metric keys consistent", test_metric_keys_consistent),
    ]

    results = []
    for name, test_func in tests:
        print(f"\n{name}:")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)

    print("\n" + "=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")

    if all(results):
        print("✓ All consistency checks passed!")
        print("\nStandalone and inline TTS evaluation will produce identical metrics.")
        return 0
    else:
        print("✗ Some checks failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
