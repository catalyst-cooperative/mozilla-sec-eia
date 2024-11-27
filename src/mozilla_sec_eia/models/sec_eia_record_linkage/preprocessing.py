"""Preprocessing for EIA and SEC input data before record linkage."""

import re
from importlib import resources
from pathlib import Path

import jellyfish
import numpy as np
import pandas as pd

from mozilla_sec_eia.models.sec10k.utils.cloud import GCSArchive
from pudl.analysis.record_linkage import name_cleaner

EIA_COL_MAP = {
    "utility_name_eia": "company_name",  # TODO: should be linking to owner or operator name?
    "address_2": "street_address_2",
}

EX21_COL_MAP = {"subsidiary": "company_name", "loc": "loc_of_incorporation"}

SEC_COL_MAP = {
    "company_conformed_name": "company_name",
    "street_1": "street_address",
    "street_2": "street_address_2",
    "zip": "zip_code",
    "business_phone": "phone_number",
    "date_filed": "report_date",
}

SHARED_COLS = [
    "report_date",
    "report_year",
    "company_name",
    "street_address",
    "street_address_2",
    "city",
    "state",  # could use state of incorporation from SEC
    "zip_code",
    "phone_number",
]

STR_COLS = [
    "company_name",
    "street_address",
    "street_address_2",
    "city",
    "state",
    "zip_code",
]

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


company_name_cleaner = name_cleaner.CompanyNameCleaner(
    cleaning_rules_list=[
        "remove_word_the_from_the_end",
        "remove_word_the_from_the_beginning",
        "replace_ampersand_by_AND",
        "replace_hyphen_by_space",
        "replace_underscore_by_space",
        "remove_text_punctuation",
        "remove_parentheses",
        "remove_brackets",
        "remove_curly_brackets",
        "enforce_single_space_between_words",
    ]
)

legal_term_remover = name_cleaner.CompanyNameCleaner(
    cleaning_rules_list=[], handle_legal_terms=2
)


# TODO: remove
def get_sec_state_code_dict():
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


# TODO: moved to output table module, take out
def _add_report_year_to_sec(sec_df):
    """Merge metadata on to get a report year for extracted SEC data.

    Expects filename to be the index of the SEC dataframe.
    """
    archive = GCSArchive()
    md = archive.get_metadata()
    sec_df = sec_df.merge(
        md[["date_filed"]], how="left", left_index=True, right_index=True
    )
    sec_df.loc[:, "report_year"] = (
        sec_df["report_date"].astype("datetime64[ns]").dt.year
    )
    return sec_df


# TODO: this is in PUDL, pull out into helper function
def _get_metaphone(row, col_name):
    if pd.isnull(row[col_name]):
        return None
    return jellyfish.metaphone(row[col_name])


# TODO: deduplicate this with what's already been done
def _clean_company_name(df):
    df.loc[:, "company_name_clean"] = company_name_cleaner.apply_name_cleaning(
        df[["company_name"]]
    ).str.strip()
    df = df[df["company_name_clean"] != ""]
    df = df.rename(columns={"company_name": "company_name_raw"}).rename(
        columns={"company_name_clean": "company_name"}
    )
    df.loc[:, "company_name_no_legal"] = legal_term_remover.apply_name_cleaning(
        df[["company_name"]]
    )
    return df


# TODO: deduplicate this with what's already been done
def clean_sec_df(df):
    """Shared cleaning for SEC 10K and Ex. 21 dataframes.

    Arguments:
        df: Ex. 21 or SEC 10K basic info dataframe with columns
        company_name, loc_of_incorporation, and report_year.
    """
    df[["company_name", "loc_of_incorporation"]] = (
        df[["company_name", "loc_of_incorporation"]]
        .fillna(pd.NA)
        .apply(lambda x: x.str.strip().str.lower())
        .replace("", pd.NA)
    )
    df = _clean_company_name(df)
    df.loc[:, "company_name_mphone"] = df.apply(
        _get_metaphone, axis=1, args=("company_name_no_legal",)
    )
    df = df[
        (~df["company_name"].isin(INVALID_NAMES))
        & (~df["company_name_raw"].isin(INVALID_NAMES))
    ]
    df = df.fillna(np.nan)

    return df


# TODO: moved to output table module, take out
def _remove_weird_sec_cols(sec_df):
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


# TODO: for now split these into separate cleaning functions
# later unite them into one cleaning function
def prepare_sec10k_basic_info_df(sec_df):
    """Preprocess SEC 10k basic information dataframe for record linkage."""
    # sec_df = _add_report_year_to_sec(sec_df)
    sec_df = sec_df.rename(columns=SEC_COL_MAP).reset_index()
    # state_code_to_name = get_sec_state_code_dict()
    # sec_df.loc[:, "loc_of_incorporation"] = sec_df["state_of_incorporation"].replace(
    #     state_code_to_name
    # )
    # TODO: maybe shouldn't expand the state names and comparison should
    # just be an exact match or nothing?
    # sec_df.loc[:, "state"] = sec_df["state"].replace(state_code_to_name)
    # TODO: needs a record_id_sec column?
    # sec_df = sec_df.rename(columns={"record_id_sec": "record_id"})
    # sec_df = _remove_weird_sec_cols(sec_df)
    sec_df = clean_sec_df(sec_df)
    sec_df[STR_COLS] = sec_df[STR_COLS].apply(lambda x: x.str.strip().str.lower())
    # TODO: cluster/mark these duplicates so they can be assigned
    # IDs post matching
    sec_df = sec_df.drop_duplicates(
        subset=[
            "central_index_key",
            "report_year",
            "company_name",
            "standard_industrial_classification",
            "city",
            "state",
            "street_address",
            "zip_code",
        ]
    )
    sec_df.loc[:, "sec_company_id"] = sec_df["central_index_key"]
    return sec_df


def prepare_ex21_df(ex21_df):
    """Preprocess Ex. 21 extracted dataframe for record linkage."""
    ex21_df = ex21_df.rename(columns=EX21_COL_MAP)
    # TODO: move this to general preprocessing function?
    state_code_to_name = get_sec_state_code_dict()
    ex21_df.loc[:, "loc_of_incorporation"] = ex21_df["loc_of_incorporation"].replace(
        state_code_to_name
    )
    name_to_state_code = {v: k for k, v in state_code_to_name.items()}
    # need this?
    ex21_df.loc[:, "state_of_incorporation"] = ex21_df["loc_of_incorporation"].replace(
        name_to_state_code
    )
    ex21_df = clean_sec_df(ex21_df)
    ex21_df = ex21_df.drop_duplicates(
        subset=["company_name", "loc_of_incorporation", "report_year"]
    )
    # ex21_df = ex21_df.reset_index(drop=True).reset_index(names="record_id")
    return ex21_df


def prepare_eia_df(eia_df):
    """Preprocess EIA utility dataframe for record linkage."""
    eia_df = eia_df.rename(columns=EIA_COL_MAP)
    eia_df.loc[:, "report_year"] = (
        eia_df["report_date"].astype("datetime64[ns]").dt.year
    )
    eia_df = eia_df.fillna(np.nan)
    eia_df[STR_COLS] = eia_df[STR_COLS].apply(lambda x: x.str.strip().str.lower())
    eia_df = _clean_company_name(eia_df)
    eia_df.loc[:, "company_name_mphone"] = eia_df.apply(
        _get_metaphone, axis=1, args=("company_name_no_legal",)
    )
    eia_df = eia_df.reset_index(drop=True).reset_index(names="record_id")
    return eia_df


def add_sec_company_id_to_subsidiaries(ex21_df: pd.DataFrame):
    """Add sec_company_id onto SEC Ex. 21 subsidiaries.

    At this point, the passed in Ex. 21 dataframe should have been
    matched to SEC 10K filers with record linkage and assigned a CIK
    where applicable (if the subsidiary files with the SEC). Take the
    subsidiaries that don't have a CIK and create an sec_company_id
    for those companies.

    Arguments:
        ex21_df: A dataframe of subsidiaries from SEC Ex. 21 filings with
        columns subsidiary_cik, company_name (of the subsidiary),
        and loc_of_incorporation.
    """
    ex21_df = ex21_df.sort_values(by="parent_cik")
    ex21_df = ex21_df.drop_duplicates(subset=["company_name", "loc_of_incorporation"])
    ex21_df.loc[:, "sec_company_id"] = (
        ex21_df["parent_cik"]
        + "_"
        + (ex21_df.groupby("parent_cik").cumcount() + 1).astype(str)
    )
    # override sec_company_id with CIK where a subsidiary has an assigned CIK
    ex21_df.loc[:, "sec_company_id"] = ex21_df["sec_company_id"].where(
        ex21_df["subsidiary_cik"].isnull(), ex21_df["subsidiary_cik"]
    )
    return ex21_df
