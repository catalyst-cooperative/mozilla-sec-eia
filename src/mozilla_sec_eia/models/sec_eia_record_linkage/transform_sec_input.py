"""Module for creating an SEC 10K output table with filing companies and subsidiary companies."""

import logging
import re
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
from dagster import AssetIn, asset, file_relative_path
from dagstermill import define_dagstermill_asset

from mozilla_sec_eia.library.record_linkage_utils import (
    expand_street_name_abbreviations,
    fill_street_address_nulls,
    flatten_companies_across_time,
    transform_company_name,
)
from mozilla_sec_eia.models.sec10k.utils.cloud import (
    convert_ex21_id_to_filename,
)
from mozilla_sec_eia.models.sec_eia_record_linkage.sec_eia_splink_config import STR_COLS

logger = logging.getLogger(f"catalystcoop.{__name__}")


EX21_COL_MAP = {"subsidiary": "company_name", "loc": "location_of_inc"}
SEC_COL_MAP = {
    "company_conformed_name": "company_name",
    "street_1": "street_address",
    "street_2": "street_address_2",
    "zip": "zip_code",
    "business_phone": "phone_number",
}

INVALID_NAMES = [
    "llc",
    "limited liability company",
    "limited",
    "ltd",
    "iiii",
    "inc",
    "incorporated",
    "partnership",
    "i",
    "name",
    "company",
    "&",
    "",
]


def _remove_weird_sec_cols(sec_df) -> pd.DataFrame:
    weird_cols = ["]fiscal_year_end", "]irs_number", "]state_of_incorporation"]
    for weird_col in weird_cols:
        if weird_col not in sec_df:
            continue
        normal_col = weird_col[1:]
        sec_df.loc[:, normal_col] = sec_df[normal_col].where(
            sec_df[weird_col].isnull(), sec_df[weird_col]
        )
        sec_df = sec_df.drop(columns=[weird_col])
    return sec_df


def _add_report_year_to_sec(sec_df: pd.DataFrame, md: pd.DataFrame) -> pd.DataFrame:
    """Merge metadata on to get a report year for extracted SEC data.

    Expects filename to be the index of the SEC dataframe.
    """
    sec_df = sec_df.merge(md[["filename", "date_filed"]], how="left", on=["filename"])
    sec_df = sec_df.rename(columns={"date_filed": "report_date"})
    sec_df.loc[:, "report_year"] = (
        sec_df["report_date"].astype("datetime64[ns]").dt.year
    )
    return sec_df


def get_sec_state_code_dict() -> dict[str, str]:
    """Create a dictionary mapping state codes to their names.

    Table found at https://www.sec.gov/submit-filings/filer-support-resources/edgar-state-country-codes
    Published by SEC and reports valid state codes
    for filers of Form D. Used to standardize the state codes
    in the SEC 10K filings. The expanded names of the state codes
    are comments in the XML file, so we have to read the XML in as
    text and parse it.
    """
    # TODO: make a check to see if SEC has published a new version of this table
    xml_filepath = (
        resources.files("mozilla_sec_eia.package_data") / "formDStateCodes.xsd.xml"
    )
    with Path.open(xml_filepath) as file:
        xml_text = file.read()

    pattern = r'<xs:enumeration value="(.*?)"/>.*?<!--\s*(.*?)\s*-->'
    state_code_dict = {
        code.lower(): name.lower()
        for code, name in re.findall(pattern, xml_text, re.DOTALL)
    }
    return state_code_dict


def clean_location_of_inc(df) -> pd.DataFrame:
    """Clean location of incorporation column in SEC basic 10K or Ex. 21 dataframe.

    Arguments:
        df: Ex. 21 or SEC 10K basic info dataframe with location_of_inc
            column.
    """
    if "state_of_incorporation" in df:
        df.loc[:, "location_of_inc"] = df["state_of_incorporation"]
    state_code_to_name = get_sec_state_code_dict()
    df.loc[:, "location_of_inc"] = (
        df["location_of_inc"]
        .replace(state_code_to_name)
        .fillna(pd.NA)
        .str.strip()
        .str.lower()
        .replace("", pd.NA)
    )
    return df


def _add_parent_company_cik(ex21_df: pd.DataFrame, md: pd.DataFrame) -> pd.DataFrame:
    """Add the CIK of the parent company to Ex. 21 subsidiaries."""
    ex21_df = ex21_df.merge(md[["filename", "cik"]], how="left", on="filename").rename(
        columns={"cik": "parent_company_cik"}
    )
    ex21_df.loc[:, "parent_company_cik"] = (
        ex21_df["parent_company_cik"].astype(str).str.zfill(10)
    )
    return ex21_df


def match_ex21_subsidiaries_to_filer_company(
    basic10k_df: pd.DataFrame, ex21_df: pd.DataFrame
) -> pd.DataFrame:
    """Match Ex. 21 subsidiaries to filer companies.

    We want to assign CIKs to Ex. 21 subsidiaries if they in turn
    file a 10k. To do this, we merge the Ex. 21 subsidiaries to 10k
    filers on comapny name. If there are multiple matches with the same
    company name we take the company with the most overlap in location of
    incorporation and nearest report years. Then we merge the CIK back onto
    the Ex. 21 df.

    Returns:
        A dataframe of the Ex. 21 subsidiaries with a column for the
        subsidiaries CIK (null if the subsidiary doesn't file).
    """
    basic10k_df = basic10k_df.drop_duplicates(
        subset=[
            "central_index_key",
            "company_name",
            "location_of_inc",
            "report_year",
        ]
    )
    merged_df = basic10k_df.merge(
        ex21_df, how="inner", on="company_name", suffixes=("_sec", "_ex21")
    )
    # split up the location of incorporation on whitespace, creating a column
    # with lists of word tokens
    merged_df.loc[:, "loc_tokens_sec"] = (
        merged_df["location_of_inc_sec"].fillna("").str.lower().str.split()
    )
    merged_df.loc[:, "loc_tokens_ex21"] = (
        merged_df["location_of_inc_ex21"].fillna("").str.lower().str.split()
    )
    # get the number of words overlapping between location of incorporation tokens
    merged_df["loc_overlap"] = merged_df.apply(
        lambda row: len(set(row["loc_tokens_sec"]) & set(row["loc_tokens_ex21"])),
        axis=1,
    )
    merged_df = merged_df.fillna(
        {
            "report_year_sec": 0,
            "report_year_ex21": 0,
        }
    )
    # get the difference in report years
    merged_df["report_year_diff"] = merged_df.apply(
        lambda row: abs(int(row["report_year_sec"]) - int(row["report_year_ex21"])),
        axis=1,
    )
    merged_df = merged_df.sort_values(
        by=[
            "company_name",
            "location_of_inc_ex21",
            "loc_overlap",
            "report_year_diff",
        ],
        ascending=[True, True, False, True],
    )
    # Select the row with the highest loc overlap and nearest report years
    # for each company name, location, and parent company record
    closest_match_df = merged_df.groupby(
        ["company_name", "location_of_inc_ex21", "parent_company_cik"], as_index=False
    ).first()
    ex21_with_cik_df = ex21_df.merge(
        closest_match_df[
            [
                "company_name",
                "parent_company_cik",
                "location_of_inc_ex21",
                "central_index_key",
            ]
        ].rename(columns={"location_of_inc_ex21": "location_of_inc"}),
        how="left",
        on=["company_name", "location_of_inc", "parent_company_cik"],
    ).rename(columns={"central_index_key": "subsidiary_cik"})
    # if a subsidiary doesn't have a CIK and has a null location
    # but its company name was assigned a CIK (with a different location)
    # then assign that CIK to the subsidiary
    ex21_with_cik_df = ex21_with_cik_df.merge(
        closest_match_df[["company_name", "central_index_key"]],
        how="left",
        on="company_name",
    ).rename(columns={"central_index_key": "company_name_merge_cik"})
    ex21_with_cik_df["subsidiary_cik"] = ex21_with_cik_df["subsidiary_cik"].where(
        ~(ex21_with_cik_df.subsidiary_cik.isnull())
        | ~(ex21_with_cik_df.location_of_inc.isnull()),
        ex21_with_cik_df["company_name_merge_cik"],
    )
    ex21_with_cik_df = ex21_with_cik_df.drop(columns="company_name_merge_cik")
    ex21_with_cik_df = ex21_with_cik_df.rename(
        columns={"subsidiary_cik": "central_index_key"}
    )
    ex21_with_cik_df = ex21_with_cik_df.drop_duplicates()

    return ex21_with_cik_df


def create_sec_company_id_for_ex21_subs(ex21_df: pd.DataFrame) -> pd.DataFrame:
    """Create an sec_company_id for Ex. 21 subsidiaries.

    This is a unique identifier string for Ex. 21 subsidiaries.
    This ID is necessary for tracking subsidiaries who aren't ultimately
    matched to a 10K filer company.
    """
    ex21_df.loc[:, "sec_company_id"] = (
        ex21_df["parent_company_cik"]
        + "_"
        + ex21_df["company_name"]
        + "_"
        + ex21_df["location_of_inc"]
    )
    return ex21_df


@asset(
    ins={
        "ex21_dfs": AssetIn("ex21_company_ownership_info"),
        "sec10k_filing_metadata_dfs": AssetIn("sec10k_filing_metadata"),
    },
)
def transformed_ex21_subsidiary_table(
    ex21_dfs: dict[str, pd.DataFrame],
    sec10k_filing_metadata_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Transform Ex. 21 table of subsidiaries before combining with basic 10k table."""
    ex21_df = pd.concat(ex21_dfs.values())
    sec10k_filing_metadata = pd.concat(sec10k_filing_metadata_dfs.values())

    ex21_df.loc[:, "filename"] = convert_ex21_id_to_filename(ex21_df)
    ex21_df = ex21_df.drop(columns=["id"])
    ex21_df = _add_report_year_to_sec(ex21_df, sec10k_filing_metadata)
    ex21_df = ex21_df.rename(columns=EX21_COL_MAP)
    ex21_df = clean_location_of_inc(ex21_df)
    ex21_df = transform_company_name(ex21_df)
    ex21_df = _add_parent_company_cik(ex21_df, sec10k_filing_metadata)
    # add an sec_company_id, ultimately this ID become the subsidiary's CIK
    # if the subsidiary is matched to an SEC filer
    ex21_df = create_sec_company_id_for_ex21_subs(ex21_df=ex21_df)
    ex21_df = flatten_companies_across_time(
        df=ex21_df, key_cols=["sec_company_id"], date_col="report_year"
    )
    ex21_df = ex21_df.fillna(np.nan)

    return ex21_df


def transform_basic10k_table(
    basic_10k_df: pd.DataFrame, sec10k_filing_metadata: pd.DataFrame
) -> pd.DataFrame:
    """Transformations on SEC basic 10K filer table to prepare for record linkage."""
    basic_10k_df = basic_10k_df.reset_index().pivot_table(
        values="value", index="filename", columns="key", aggfunc="first"
    )
    basic_10k_df.columns.name = None
    basic_10k_df = (
        basic_10k_df.reset_index()
        .pipe(_remove_weird_sec_cols)
        .pipe(_add_report_year_to_sec, sec10k_filing_metadata)
        .rename(columns=SEC_COL_MAP)
        .pipe(clean_location_of_inc)
        .pipe(transform_company_name)
        .assign(
            zip_code=lambda df: df["zip_code"].str[:5],
            files_10k=True,
            sec_company_id=lambda df: df["central_index_key"],
        )
        .pipe(fill_street_address_nulls)
    )
    basic_10k_df[STR_COLS] = basic_10k_df[STR_COLS].apply(
        lambda x: x.str.strip().str.lower()
    )
    basic_10k_df["street_address"] = expand_street_name_abbreviations(
        basic_10k_df["street_address"]
    )
    # flatten across time on unique company name and address pair
    basic_10k_df = flatten_companies_across_time(
        df=basic_10k_df, key_cols=["company_name", "street_address"]
    )

    return basic_10k_df


@asset(
    ins={
        "basic_10k_dfs": AssetIn("basic_10k_company_info"),
        "sec10k_filing_metadata_dfs": AssetIn("sec10k_filing_metadata"),
    },
)
def transformed_basic_10k(
    basic_10k_dfs: dict[str, pd.DataFrame],
    sec10k_filing_metadata_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Asset for creating a cleaned basic 10k table with EIA utility matched.

    Flatten the table across time to only keep the most recent record
    for each unique company name and address pair. Clean table and link filers
    to EIA utilities.
    """
    basic_10k_df = pd.concat(basic_10k_dfs.values())
    sec10k_filing_metadata = pd.concat(sec10k_filing_metadata_dfs.values())
    basic_10k_df = transform_basic10k_table(basic_10k_df, sec10k_filing_metadata)
    out_df = basic_10k_df.fillna(np.nan).reset_index(names="record_id")

    return out_df


core_sec_10k__filers = define_dagstermill_asset(
    "core_sec_10k__filers",
    notebook_path=file_relative_path(__file__, "./notebooks/splink-sec-eia.ipynb"),
    ins={
        "clean_eia_df": AssetIn("core_eia__parents_and_subsidiaries"),
        "clean_basic_10k_df": AssetIn("transformed_basic_10k"),
    },
    save_notebook_on_failure=True,
)


@asset(
    ins={
        "clean_ex21_df": AssetIn("transformed_ex21_subsidiary_table"),
        "clean_eia_df": AssetIn("core_eia__parents_and_subsidiaries"),
    },
    deps=["core_sec_10k__filers"],
    io_manager_key="pandas_parquet_io_manager",
)
def out_sec_10k__parents_and_subsidiaries(
    clean_ex21_df: pd.DataFrame,
    clean_eia_df: pd.DataFrame,
) -> pd.DataFrame:
    """Asset for creating an SEC 10K output table.

    Add in Ex. 21 subsidiaries and link them to already present
    filing companies. Create an sec_company_id for subsidiaries
    that aren't linked to a CIK.
    """
    sec_10k_filers_matched_df = pd.read_parquet(
        "gs://sec10k-outputs/v2/core_sec_10k__filers.parquet"
    )
    ex21_df_with_cik = match_ex21_subsidiaries_to_filer_company(
        basic10k_df=sec_10k_filers_matched_df, ex21_df=clean_ex21_df
    )
    sec_10k_filers_matched_df = sec_10k_filers_matched_df.merge(
        ex21_df_with_cik[["central_index_key", "parent_company_cik", "own_per"]],
        how="left",
        on="central_index_key",
    )
    # get the subsidiary companies that weren't matched to a 10K filing company
    ex21_non_filing_subs_df = ex21_df_with_cik[
        ex21_df_with_cik["central_index_key"].isnull()
    ]
    ex21_non_filing_subs_df.loc[:, "files_10k"] = False
    # the last step is to take the EIA utilities that haven't been matched
    # to a filer company, and merge them by company name onto the Ex. 21 subs
    unmatched_eia_df = clean_eia_df[
        ~clean_eia_df["utility_id_eia"].isin(
            sec_10k_filers_matched_df.utility_id_eia.unique()
        )
    ].drop_duplicates(subset="company_name")
    ex21_non_filing_subs_df = ex21_non_filing_subs_df.merge(
        unmatched_eia_df[["utility_id_eia", "company_name"]],
        how="left",
        on="company_name",
    ).drop_duplicates(subset="sec_company_id")
    logger.info(
        f"Ex. 21 subsidiary names matched to an EIA utility name: {len(ex21_non_filing_subs_df["utility_id_eia"].unique())}"
    )
    out_df = pd.concat([sec_10k_filers_matched_df, ex21_non_filing_subs_df])
    return out_df


production_assets = [
    transformed_basic_10k,
    transformed_ex21_subsidiary_table,
    core_sec_10k__filers,
    out_sec_10k__parents_and_subsidiaries,
]
