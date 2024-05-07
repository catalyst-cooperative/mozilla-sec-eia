"""Implements CLI for SEC to EIA linkage development.

CLI is structured with nested sub-commands to make it easy
to add new scripts which can be accessed through one top-level
interface.
"""

import argparse
import logging
import sys

import coloredlogs

from mozilla_sec_eia.basic_10k import extract
from mozilla_sec_eia.utils import GCSArchive

# This is the module-level logger, for any logs
logger = logging.getLogger(__name__)


def parse_command_line(argv: list[str]) -> argparse.Namespace:
    """Parse command line arguments. See the -h option for details.

    Args:
        argv (str): Command line arguments, including caller filename.

    Returns:
        dict: Dictionary of command line arguments and their parsed values.

    """

    def formatter(prog) -> argparse.HelpFormatter:
        """This is a hack to create HelpFormatter with a particular width."""
        return argparse.HelpFormatter(prog, width=88)

    # Use the module-level docstring as the script's description in the help message.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=formatter)
    subparsers = parser.add_subparsers(required=True)

    # Add command to validate filing archive contents
    validate_parser = subparsers.add_parser("validate")
    validate_parser.set_defaults(func=lambda: GCSArchive().validate_archive())

    # Add command to extract basic 10k data
    validate_parser = subparsers.add_parser("extract")
    validate_parser.set_defaults(func=extract)

    arguments = parser.parse_args(argv[1:])
    return arguments


def main() -> int:
    """Demonstrate a really basic command line interface (CLI) that takes arguments."""
    logger = logging.getLogger("catalystcoop")
    log_format = "%(asctime)s [%(levelname)8s] %(name)s:%(lineno)s %(message)s"
    coloredlogs.install(fmt=log_format, logger=logger)

    args = parse_command_line(sys.argv)
    return args.func(**{key: val for key, val in vars(args).items() if key != "func"})


if __name__ == "__main__":
    sys.exit(main())
