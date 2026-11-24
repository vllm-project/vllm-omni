"""Serve synchronous LIBERO actions with a vLLM-Omni-owned HTTP frontend.

Both this frontend and the GPU worker run from the same vLLM-Omni checkout.
No training framework, config loader or checkpoint loader is imported here.
"""

import argparse
import logging
from pathlib import Path

from .action_policy_common import DEFAULT_LIBERO_STATS
from .libero_remote_policy import LiberoRemotePolicyService, RemotePolicyConfig
from .remote_action_stream import RemoteActionPolicyHTTPServer
from .vllm_omni_action import VllmOmniActionTransport


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--vllm-base-url", default="http://127.0.0.1:8100")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8101)
    parser.add_argument("--callback-base-url")
    parser.add_argument("--model")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flow-shift", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--poll-interval", type=float, default=0.05)
    parser.add_argument("--action-stats-path", type=Path, default=DEFAULT_LIBERO_STATS)
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--dump-every", type=int, default=1)
    parser.add_argument("--disable-streaming", action="store_true")
    parser.add_argument("--action-time-schedule", choices=("has_euler", "constant_euler"), default="constant_euler")
    parser.add_argument("--early-stop-actions", type=int, help="Stop after the first K actions are ready (1..16)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    callback_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    callback_url = args.callback_base_url or f"http://{callback_host}:{args.port}"
    transport = VllmOmniActionTransport(
        args.vllm_base_url,
        model=args.model,
        seed=args.seed,
        flow_shift=args.flow_shift,
        timeout=args.timeout,
        poll_interval=args.poll_interval,
        action_time_schedule=args.action_time_schedule,
        early_stop_actions=args.early_stop_actions,
    )
    service = LiberoRemotePolicyService(
        transport,
        RemotePolicyConfig(
            action_stats_path=args.action_stats_path,
            dump_dir=args.dump_dir,
            dump_every=args.dump_every,
            streaming_enabled=not args.disable_streaming,
            callback_base_url=callback_url,
        ),
    )
    try:
        # Fail before opening the policy socket if the model API is unavailable.
        info = service.get_info()
        logging.info("Cosmos3 synchronous action policy: %s", info)
        server = RemoteActionPolicyHTTPServer((args.host, args.port), service)
        try:
            server.serve_forever()
        finally:
            server.server_close()
    finally:
        service.close()


if __name__ == "__main__":
    main()
