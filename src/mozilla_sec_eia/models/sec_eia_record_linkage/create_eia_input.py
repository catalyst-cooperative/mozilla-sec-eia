"""Create an EIA input utilities table that's ready for record linkage with the SEC 10K companies."""

import pandas as pd


# TODO: make Dagster inputs instead of reading from AWS?
def get_eia861_utilities_table():
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


# TODO: add Dagster asset inputs for PUDL inputs instead of reading from AWS?
def get_eia_utilities_table():
    """Create a table of EIA Form 860 and 861 utilities."""
    raw_eia_df = pd.read_parquet(
        "s3://pudl.catalyst.coop/stable/out_eia__yearly_utilities.parquet"
    )
    eia861_df = get_eia861_utilities_table()
    eia_df = pd.concat([raw_eia_df, eia861_df])
    eia_df = eia_df.drop_duplicates(
        subset=["utility_id_eia", "report_date"], keep="first"
    )
    eia_df["report_date"] = eia_df["report_date"].astype("datetime64[ns]")
    # there are nulls from non harvested 861 utilities
    eia_df = eia_df.dropna(subset="utility_name_eia")
    return eia_df
