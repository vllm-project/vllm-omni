from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# Import base configs at module top level so the subclasses below are pickle-
# resolvable (worker processes import this module fresh via __main__).
from molmo_spaces.configs.policy_configs_baselines import (  # noqa: E402
    DreamZeroPolicyConfig,
)
from molmo_spaces.evaluation.configs.evaluation_configs import (  # noqa: E402
    DreamZeroPolicyEvalConfig,
)


# We only need to change the backend host and port to the vllm-host!
class DreamZeroVllmOmniPolicyConfig(DreamZeroPolicyConfig):
    remote_config: dict = dict(host="127.0.0.1", port=8000)


class DreamZeroVllmOmniEvalConfig(DreamZeroPolicyEvalConfig):
    policy_config: DreamZeroVllmOmniPolicyConfig = DreamZeroVllmOmniPolicyConfig()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--benchmark_dir",
        required=True,
        help=(
            "Path to a MolmoSpaces benchmark directory, for example "
            "$MOLMOSPACES_BENCHMARK_DIR/20260327/ithor/FrankaCloseHardBench/"
            "FrankaCloseHardBench_20260206_json_benchmark"
        ),
    )
    parser.add_argument("--max_episodes", type=int, default=1)
    parser.add_argument("--task_horizon_steps", type=int, default=80)
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write evaluation outputs (created if missing).",
    )
    parser.add_argument("--episode_idx", type=int, default=None)
    args = parser.parse_args()

    policy_config = DreamZeroVllmOmniPolicyConfig(remote_config=dict(host=args.host, port=args.port))
    DreamZeroVllmOmniPolicyConfig.model_fields["remote_config"].default = policy_config.remote_config
    DreamZeroVllmOmniEvalConfig.model_fields["policy_config"].default = policy_config

    # Import after env vars are set so MuJoCo picks EGL.
    from molmo_spaces.evaluation import run_evaluation

    cfg_cls = DreamZeroVllmOmniEvalConfig

    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[eval] benchmark_dir={args.benchmark_dir}")
    print(f"[eval] max_episodes={args.max_episodes} task_horizon_steps={args.task_horizon_steps}")
    print(f"[eval] remote policy: ws://{args.host}:{args.port}/v1/realtime/robot/openpi")

    results = run_evaluation(
        eval_config_cls=cfg_cls,
        benchmark_dir=Path(args.benchmark_dir),
        max_episodes=args.max_episodes,
        task_horizon_steps=args.task_horizon_steps,
        num_workers=1,
        use_wandb=False,
        output_dir=output_dir,
        episode_idx=args.episode_idx,
    )

    print(f"[eval] success={results.success_count}/{results.total_count} ({results.success_rate:.1%})")
    print(f"[eval] output_dir={results.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
