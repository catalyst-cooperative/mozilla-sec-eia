"""Create an EIA input utilities table that's ready for record linkage with the SEC 10K companies."""

import numpy as np
import pandas as pd
from dagster import AssetOut, multi_asset

from mozilla_sec_eia.library.record_linkage_utils import (
    fill_street_address_nulls,
    transform_company_name,
)
from mozilla_sec_eia.models.sec_eia_record_linkage.sec_eia_splink_config import STR_COLS

EIA_COL_MAP = {
    "utility_name_eia": "company_name",  # TODO: should be linking to owner or operator name?
    "address_2": "street_address_2",
}


# TODO: make Dagster inputs instead of reading from AWS?
def harvest_eia861_utilities():
    """Get the utilities contained in EIA Form 861.

    TODO: In PUDL we should eventually implement an actual thorough
    harvesting of utilities from all EIA Form 861 tables, but this is
    good enough for now.
    """
    raw_eia861_df = pd.read_parquet(
        "s3://pudl.catalyst.coop/stable/core_eia861__assn_utility.parquet"
    )
    harvested_df = pd.concat(
        [
            pd.read_parquet(
                "s3://pudl.catalyst.coop/stable/core_eia861__yearly_utility_data_misc.parquet"
            )[["report_date", "utility_id_eia", "utility_name_eia"]],
            pd.read_parquet(
                "s3://pudl.catalyst.coop/stable/core_eia861__yearly_operational_data_misc.parquet"
            )[["report_date", "utility_id_eia", "utility_name_eia"]],
            pd.read_parquet(
                "s3://pudl.catalyst.coop/stable/core_eia861__yearly_demand_side_management_misc.parquet"
            )[["report_date", "utility_id_eia", "utility_name_eia"]],
            pd.read_parquet(
                "s3://pudl.catalyst.coop/stable/core_eia861__yearly_energy_efficiency.parquet"
            )[["report_date", "utility_id_eia", "utility_name_eia"]],
        ]
    )
    eia861_df = raw_eia861_df.merge(
        harvested_df, on=["report_date", "utility_id_eia"], how="left"
    ).drop_duplicates(subset=["report_date", "utility_id_eia"])
    mergers_df = pd.read_parquet(
        "s3://pudl.catalyst.coop/stable/core_eia861__yearly_mergers.parquet"
    )
    mergers_df = mergers_df[mergers_df["new_parent"].notna()]
    eia861_df = eia861_df.merge(
        mergers_df[
            ["report_date", "new_parent", "merge_address", "merge_city", "merge_state"]
        ],
        how="left",
        left_on=["report_date", "utility_name_eia"],
        right_on=["report_date", "new_parent"],
    )
    eia861_df = eia861_df.rename(
        columns={"merge_address": "street_address", "merge_city": "city"}
    )
    eia861_df = (
        eia861_df.groupby(["report_date", "utility_id_eia"]).first().reset_index()
    )

    eia861_df["state"] = eia861_df["state"].where(
        eia861_df["merge_state"].isnull(), eia861_df["merge_state"]
    )
    eia861_df = eia861_df.drop(columns=["new_parent", "merge_state"])
    return eia861_df


@multi_asset(
    outs={
        "core_eia__parents_and_subsidiaries": AssetOut(
            io_manager_key="pandas_parquet_io_manager"
        )
        # TODO: allow year partitions?
    }
)
# TODO: add Dagster asset inputs for PUDL inputs instead of reading from AWS?
def eia_rl_input_table():
    """Create a table of EIA Form 860 and 861 utilities."""
    raw_eia_df = pd.read_parquet(
        "s3://pudl.catalyst.coop/stable/out_eia__yearly_utilities.parquet"
    )
    eia861_df = harvest_eia861_utilities()
    eia_df = pd.concat([raw_eia_df, eia861_df])
    eia_df = eia_df.drop_duplicates(
        subset=["utility_id_eia", "report_date"], keep="first"
    ).dropna(subset="utility_name_eia")
    eia_df = eia_df.rename(columns=EIA_COL_MAP)
    eia_df["report_date"] = eia_df["report_date"].astype("datetime64[ns]")
    eia_df.loc[:, "report_year"] = eia_df["report_date"].dt.year
    eia_df = transform_company_name(eia_df)
    eia_df.loc[:, "zip_code"] = eia_df["zip_code"].str[:5]
    eia_df = fill_street_address_nulls(eia_df)
    eia_df[STR_COLS] = eia_df[STR_COLS].apply(lambda x: x.str.strip().str.lower())
    eia_df = eia_df.fillna(np.nan)
    eia_df = eia_df.reset_index(drop=True).reset_index(names="record_id")

    return eia_df


production_assets = [eia_rl_input_table]
