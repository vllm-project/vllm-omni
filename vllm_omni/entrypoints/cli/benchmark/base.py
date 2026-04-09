import argparse

from vllm.entrypoints.cli.types import CLISubcommand


class OmniBenchmarkSubcommandBase(CLISubcommand):
    """The base class of subcommands for vllm bench."""

    help: str

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add the CLI arguments to the parser."""
        raise NotImplementedError

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        """Run the benchmark.

        Args:
            args: The arguments to the command.
        """
        raise NotImplementedError
