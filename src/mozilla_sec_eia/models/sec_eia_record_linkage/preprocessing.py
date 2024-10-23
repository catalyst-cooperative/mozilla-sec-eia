"""Preprocessing for EIA and SEC input data before record linkage."""

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

state_code_dict = {
    # https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States#States.
    "AK": "Alaska",
    "AL": "Alabama",
    "AR": "Arkansas",
    "AZ": "Arizona",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "IA": "Iowa",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "MA": "Massachusetts",
    "MD": "Maryland",
    "ME": "Maine",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MO": "Missouri",
    "MS": "Mississippi",
    "MT": "Montana",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "NE": "Nebraska",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NV": "Nevada",
    "NY": "New York",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VA": "Virginia",
    "VT": "Vermont",
    "WA": "Washington",
    "WI": "Wisconsin",
    "WV": "West Virginia",
    "WY": "Wyoming",
    # https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States#Federal_district.
    "DC": "District of Columbia",
    # https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States#Inhabited_territories.
    "AS": "American Samoa",
    "GU": "Guam GU",
    "MP": "Northern Mariana Islands",
    "PR": "Puerto Rico PR",
    "VI": "U.S. Virgin Islands",
}
state_code_to_name = {k.lower(): v.lower() for k, v in state_code_dict.items()}

company_name_cleaner = name_cleaner.CompanyNameCleaner(
    cleaning_rules_list=[
        "remove_word_the_from_the_end",
        "remove_word_the_from_the_beginning",
        "replace_amperstand_between_space_by_AND",
        "replace_hyphen_by_space",
        "replace_hyphen_between_spaces_by_single_space",
        "replace_underscore_by_space",
        "replace_underscore_between_spaces_by_single_space",
        # "remove_all_punctuation",
        # "remove_numbers",
        # "remove_math_symbols",
        "remove_words_in_parentheses",
        "remove_parentheses",
        "remove_brackets",
        "remove_curly_brackets",
        "enforce_single_space_between_words",
    ]
)


def _add_report_year_to_sec(sec_df):
    """Merge metadata on to get a report year for extracted SEC data.

    Expects filename to be the index of the SEC dataframe.
    """
    archive = GCSArchive()
    md = archive.get_metadata()
    return sec_df.merge(
        md[["date_filed"]], how="left", left_index=True, right_index=True
    )


# TODO: this is in PUDL, pull out into helper function
def _get_metaphone(row, col_name):
    if pd.isnull(row[col_name]):
        return None
    return jellyfish.metaphone(row[col_name])


def _clean_company_name(df):
    df.loc[:, "company_name_clean"] = company_name_cleaner.apply_name_cleaning(
        df[["company_name"]]
    ).str.strip()
    df = df[df["company_name_clean"] != ""]
    df = df.rename(columns={"company_name": "company_name_raw"}).rename(
        columns={"company_name_clean": "company_name"}
    )
    return df


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
    )
    df.loc[:, "company_name"] = df["company_name"].replace("", pd.NA)
    df.loc[:, "loc_of_incorporation"] = df["loc_of_incorporation"].replace("", pd.NA)
    df = _clean_company_name(df)
    df = df[
        (~df["company_name"].isin(INVALID_NAMES))
        & ~(df["company_name_raw"].isin(INVALID_NAMES))
    ]
    df = df.fillna(np.nan)
    df = df.drop_duplicates(
        subset=["company_name", "loc_of_incorporation", "report_year"]
    )
    return df


def _remove_weird_sec_cols(sec_df):
    for weird_col in ["]fiscal_year_end", "]irs_number", "]state_of_incorporation"]:
        if weird_col not in sec_df:
            continue
        normal_col = weird_col[1:]
        sec_df.loc[:, normal_col] = sec_df[normal_col].where(
            sec_df[weird_col].isnull(), sec_df[weird_col]
        )
    return sec_df


# TODO: for now split these into separate cleaning functions
# later unite them into one cleaning function
def prepare_sec10k_basic_info_df(sec_df):
    """Preprocess SEC 10k basic information dataframe for record linkage."""
    sec_df = _add_report_year_to_sec(sec_df)
    sec_df = sec_df.rename(columns=SEC_COL_MAP).reset_index()
    sec_df.loc[:, "report_year"] = (
        sec_df["report_date"].astype("datetime64[ns]").dt.year
    )
    sec_df.loc[:, "loc_of_incorporation"] = sec_df["state_of_incorporation"].replace(
        state_code_to_name
    )
    # TODO: maybe shouldn't expand the state names and comparison should
    # just be an exact match or nothing?
    # sec_df.loc[:, "state"] = sec_df["state"].replace(state_code_to_name)
    # TODO: needs a record_id_sec column?
    # sec_df = sec_df.rename(columns={"record_id_sec": "record_id"})
    sec_df = _remove_weird_sec_cols(sec_df)
    sec_df = clean_sec_df(sec_df)
    sec_df[STR_COLS] = sec_df[STR_COLS].apply(lambda x: x.str.strip().str.lower())
    sec_df.loc[:, "company_name_mphone"] = sec_df.apply(
        _get_metaphone, axis=1, args=("company_name",)
    )
    sec_df = sec_df.reset_index(names="record_id")
    return sec_df


def prepare_ex21_df(ex21_df):
    """Preprocess Ex. 21 extracted dataframe for record linkage."""
    ex21_df = ex21_df.rename(columns=EX21_COL_MAP)
    # TODO: move this to general preprocessing function?
    ex21_df.loc[:, "loc_of_incorporation"] = ex21_df["loc_of_incorporation"].replace(
        state_code_to_name
    )
    ex21_df = clean_sec_df(ex21_df)
    ex21_df.loc[:, "company_name_mphone"] = ex21_df.apply(
        _get_metaphone, axis=1, args=("company_name",)
    )
    ex21_df = ex21_df.reset_index(names="record_id")
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
        _get_metaphone, axis=1, args=("company_name",)
    )
    eia_df = eia_df.reset_index(names="record_id")
    return eia_df


"""
def preprocessing(eia_df, sec_df):
    # TODO: reorganize to be more similar to ferc to eia match structure
    eia_df = eia_df.rename(columns=EIA_COL_MAP)

    # TODO: fill out this prepare for matching function
    # eia_df = prepare_for_matching(eia_df)
    # sec_df = prepare_for_matching(sec_df)
    sec_df.loc[:, "loc_of_incorporation"] = sec_df["state_of_incorporation"].replace(
        state_code_to_name
    )
    sec_df.loc[:, "loc_of_incorporation"] = sec_df["loc_of_incorporation"].where(
        ~sec_df["loc_of_incorporation"].isnull(), sec_df["city"]
    )
    sec_df = sec_df.rename(columns={"record_id_sec": "record_id"})
    eia_df = eia_df.rename(columns={"record_id_eia": "record_id"})
"""
