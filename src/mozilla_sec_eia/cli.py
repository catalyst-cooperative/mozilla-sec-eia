"""Implements CLI for SEC to EIA linkage development.

CLI is structured with nested sub-commands to make it easy
to add new scripts which can be accessed through one top-level
interface.
"""

import argparse
import logging
import sys
from pathlib import Path

import coloredlogs

from mozilla_sec_eia.ex_21.create_labeled_dataset import (
    create_inputs_for_label_studio,
)
from mozilla_sec_eia.ex_21.rename_labeled_filings import rename_filings
from mozilla_sec_eia.ex_21.train_extractor import train_model
from mozilla_sec_eia.extract import extract_filings, validate_extraction
from mozilla_sec_eia.utils import GCSArchive

# This is the module-level logger, for any logs
logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).parent.parent.parent.resolve()


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
    validate_parser.add_argument("--labeled-json-path")
    validate_parser.add_argument("--gcs-training-data-dir", default="labeled/")
    validate_parser.add_argument("--model-output-dir", default="layoutlm_trainer")
    validate_parser.add_argument("--test-size", default=0.2)
    validate_parser.set_defaults(func=train_model)

    # Add command to rename labeled filings on GCS
    validate_parser = subparsers.add_parser("rename_filings")
    validate_parser.set_defaults(func=rename_filings)

    # Add command to extract basic 10k data
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--dataset", nargs=1, default="basic_10k")
    extract_parser.add_argument("--continue-run", action="store_true", default=False)
    extract_parser.add_argument("--num-filings", default=-1, nargs="?", type=int)
    extract_parser.set_defaults(func=extract_filings)

    validate_extract_parser = subparsers.add_parser("validate")
    validate_extract_parser.add_argument("--dataset", nargs=1, default="basic_10k")
    validate_extract_parser.add_argument("--dataset", nargs=1, default="ex21")
    validate_extract_parser.set_defaults(func=validate_extraction)

    # Add command to create Label Studio inputs from cached Ex. 21 images and PDFs
    validate_parser = subparsers.add_parser("create_ls_inputs")
    validate_parser.add_argument("--pdfs-dir", default=ROOT_DIR / "sec10k_filings/pdfs")
    validate_parser.add_argument("--cache-dir", default=ROOT_DIR / "sec10k_filings")
    validate_parser.set_defaults(func=create_inputs_for_label_studio)

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
