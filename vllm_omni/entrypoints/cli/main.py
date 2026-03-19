"""
CLI entry point for vLLM-Omni that intercepts vLLM commands.
"""

import importlib.metadata
import sys


def main():
    """Main CLI entry point that intercepts vLLM commands."""
    # Check if --omni flag is present
    if "--omni" not in sys.argv:
        from vllm.entrypoints.cli.main import main as vllm_main

        vllm_main()
        return
    else:
        from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG, cli_env_setup
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        import vllm_omni.entrypoints.cli.benchmark.main
        import vllm_omni.entrypoints.cli.serve

        CMD_MODULES = [
            vllm_omni.entrypoints.cli.serve,
            vllm_omni.entrypoints.cli.benchmark.main,
        ]

        cli_env_setup()

        parser = FlexibleArgumentParser(
            description="vLLM OMNI CLI",
            epilog=VLLM_SUBCMD_PARSER_EPILOG.format(subcmd="[subcommand]"),
        )
        parser.add_argument(
            "-v",
            "--version",
            action="version",
            version=importlib.metadata.version("vllm_omni"),
        )
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        cmds = {}
        for cmd_module in CMD_MODULES:
            new_cmds = cmd_module.cmd_init()
            for cmd in new_cmds:
                cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
                cmds[cmd.name] = cmd
        args = parser.parse_args()
        if args.subparser in cmds:
            cmds[args.subparser].validate(args)

        if hasattr(args, "dispatch_function"):
            args.dispatch_function(args)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
