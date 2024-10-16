"""Implement useful generic io-managers."""

import pickle

import pandas as pd
from dagster import InputContext, OutputContext, UPathIOManager
from upath import UPath


class PandasParquetIOManager(UPathIOManager):
    """Read and write pandas dataframes as parquet files on local or remote filesystem."""

    extension: str = ".parquet"

    def dump_to_path(self, context: OutputContext, obj: pd.DataFrame, path: UPath):
        """Write parquet."""
        with path.open("wb") as file:
            obj.to_parquet(file)

    def load_from_path(self, context: InputContext, path: UPath) -> pd.DataFrame:
        """Read parquet."""
        with path.open("rb") as file:
            return pd.read_parquet(file)


class PickleUPathIOManager(UPathIOManager):
    """Read and write pandas dataframes as parquet files on local or remote filesystem."""

    extension: str = ".pickle"

    def dump_to_path(self, context: OutputContext, obj: pd.DataFrame, path: UPath):
        """Write parquet."""
        with path.open("wb") as file:
            pickle.dump(obj, file)

    def load_from_path(self, context: InputContext, path: UPath) -> pd.DataFrame:
        """Read parquet."""
        with path.open("rb") as file:
            return pickle.load(file)  # noqa: S301
