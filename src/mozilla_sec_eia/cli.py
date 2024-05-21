"""Implements CLI for SEC to EIA linkage development.

CLI is structured with nested sub-commands to make it easy
to add new scripts which can be accessed through one top-level
interface.
"""

import argparse
import logging
import sys

from mozilla_sec_eia.ex_21.extractor import train_model
from mozilla_sec_eia.rename_labeled_filings import rename_filings
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
    validate_parser = subparsers.add_parser("validate_archive")
    validate_parser.set_defaults(func=lambda: GCSArchive().validate_archive())

    # Add command to fine-tune ex21 extractor
    validate_parser = subparsers.add_parser("finetune_ex21")
    validate_parser.add_argument("--model-output-dir", default="layoutlm_trainer")
    validate_parser.add_argument("--test-size", default=0.2)
    validate_parser.set_defaults(func=train_model)

    # Add command to rename labeled filings on GCS
    validate_parser = subparsers.add_parser("rename_filings")
    validate_parser.set_defaults(func=rename_filings)

    arguments = parser.parse_args(argv[1:])
    return arguments


def main() -> int:
    """Demonstrate a really basic command line interface (CLI) that takes arguments."""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)8s] %(name)s:%(lineno)s %(message)s",
        level=logging.INFO,
    )

    args = parse_command_line(sys.argv)
    return args.func(**{key: val for key, val in vars(args).items() if key != "func"})


if __name__ == "__main__":
    sys.exit(main())
