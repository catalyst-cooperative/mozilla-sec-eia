"""Preprocessing for EIA and SEC input data before record linkage."""

import jellyfish
import numpy as np
import pandas as pd

from pudl.analysis.record_linkage import name_cleaner

EIA_COL_MAP = {
    "utility_name_eia": "company_name",  # TODO: should be linking to owner or operator name?
    "address_2": "street_address_2",
}

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


# TODO: this is in PUDL, pull out into helper function
def _get_metaphone(row, col_name):
    if pd.isnull(row[col_name]):
        return None
    return jellyfish.metaphone(row[col_name])


# TODO: delete
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


# TODO: delete
def clean_sec_df(df):
    """Shared cleaning for SEC 10K and Ex. 21 dataframes.

    Arguments:
        df: Ex. 21 or SEC 10K basic info dataframe with columns
        company_name, loc_of_incorporation, and report_year.
    """
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


# TODO: delete
def prepare_sec10k_basic_info_df(sec_df):
    """Preprocess SEC 10k basic information dataframe for record linkage."""
    sec_df = sec_df.rename(columns=SEC_COL_MAP).reset_index()
    sec_df = clean_sec_df(sec_df)
    sec_df[STR_COLS] = sec_df[STR_COLS].apply(lambda x: x.str.strip().str.lower())
    # TODO: does this actually drop anything?
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
    return sec_df


# TODO: delete
def prepare_ex21_df(ex21_df):
    """Preprocess Ex. 21 extracted dataframe for record linkage."""
    ex21_df = clean_sec_df(ex21_df)
    return ex21_df


# TODO: delete
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
