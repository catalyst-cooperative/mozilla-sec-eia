"""Utility functions for cleaning strings during modeling preprocessing steps."""

import json
from enum import StrEnum
from importlib import resources

import jellyfish
import pandas as pd

from pudl.analysis.record_linkage import name_cleaner

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


def clean_company_name(
    df: pd.DataFrame, col_name: str = "company_name"
) -> pd.DataFrame:
    """Conduct cleaning on a company name column and add column without legal terms.

    Uses the PUDL name cleaner object to do basic cleaning on `col_name` column
    such as stripping punctuation, correcting case, normalizing legal
    terms etc. The clean column becomes the `col_name` column and the original
    `col_name` column is renamed to `{col_name}_raw`. Also adds a column called
    `{col_name}_no_legal` which has legal terms stripped from the clean strings.

    Arguments:
        df: The dataframe that is to be cleaned. Must contain `col_name` column.
        col_name: The name of the column with the company name strings.

    Returns:
        pd.DataFrame: The original dataframe with `col_name` now containing
            cleaned strings and an additional column with the raw strings
            and a column with the legal terms stripped from the company name.
    """
    df[col_name] = df[col_name].fillna(pd.NA).str.strip().str.lower().replace("", pd.NA)
    df.loc[:, f"{col_name}_clean"] = company_name_cleaner.apply_name_cleaning(
        df[[col_name]]
    ).str.strip()
    df = df[df[f"{col_name}_clean"] != ""]
    df = df.rename(columns={col_name: f"{col_name}_raw"}).rename(
        columns={f"{col_name}_clean": col_name}
    )
    df.loc[:, f"{col_name}_no_legal"] = legal_term_remover.apply_name_cleaning(
        df[[col_name]]
    )
    return df


def handle_invalid_names(
    df: pd.DataFrame, col_name: str = "company_name", drop_invalid: bool = True
) -> pd.DataFrame:
    """Drop rows that have invalid company names, like just 'llc', or 'partnership'.

    Either drop invalid company name values or fill with the empty string. Invalid
    values are contained in `INVALID_NAMES`.
    """
    if drop_invalid:
        return df[(~df[col_name].isin(INVALID_NAMES))]
    df[col_name] = df[col_name].where(~df[col_name].isin(INVALID_NAMES), "")
    return df


def flatten_companies_across_time(
    df: pd.DataFrame, key_cols: list[str], date_col: str = "report_date"
) -> pd.DataFrame:
    """Keep only the most recent record for each group of `key_cols`.

    Dataframe must have all of `key_cols` and `date_col`.
    """
    df = (
        df.sort_values(by=date_col, ascending=False).groupby(key_cols).first()
    ).reset_index()
    return df


# TODO: this is in PUDL, deduplicate
def get_metaphone_col(col: pd.Series) -> pd.Series:
    """Get the metaphones of the strings in a column."""
    return col.apply(jellyfish.metaphone)


class HandleNulls(StrEnum):
    """Enum for handling null values in company name transform."""

    DROP = "drop"
    FILL_EMPTY_STR = "fill_empty_str"


def transform_company_name(
    df: pd.DataFrame,
    col_name: str = "company_name",
    handle_nulls: HandleNulls = HandleNulls.DROP,
) -> pd.DataFrame:
    """Apply cleaning, get metaphone col, drop invalid rows."""
    df = clean_company_name(df, col_name=col_name)
    if handle_nulls == HandleNulls.DROP:
        df = handle_invalid_names(df, col_name, drop_invalid=True)
        df = df[~df[col_name].isnull()]
    elif handle_nulls == HandleNulls.FILL_EMPTY_STR:
        df = handle_invalid_names(df, col_name, drop_invalid=False)
        df = df.fillna({col_name: ""})
    df.loc[:, f"{col_name}_mphone"] = get_metaphone_col(df[f"{col_name}_no_legal"])

    return df


def fill_street_address_nulls(
    df: pd.DataFrame,
    address_col: str = "street_address",
    secondary_address_col: str = "street_address_2",
) -> pd.DataFrame:
    """Fill null street address with value from secondary address column."""
    df[address_col] = df[address_col].where(
        (~df[address_col].isnull()) | (df[secondary_address_col].isnull()),
        df[secondary_address_col],
    )
    return df


def expand_street_name_abbreviations(col: pd.Series) -> pd.Series:
    """Standardize street address suffixes, like street to st.

    Expects lower case strings in column.
    """
    # remove punctuation from column first
    col = col.str.replace(r"[^\w\s]", "", regex=True)

    json_source = (
        resources.files("mozilla_sec_eia.package_data")
        / "street_suffix_abbreviations.json"
    )
    with json_source.open() as f:
        address_expansions = json.load(f)
    for standard_abbr, suffix_list in address_expansions.items():
        pattern = r"\b(" + "|".join(suffix_list) + r")\b"
        col = col.str.replace(pattern, standard_abbr, regex=True)
    return col
