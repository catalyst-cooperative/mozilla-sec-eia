"""Define table structure SEC10k extraction."""

import pandera as pa
from dagster_pandera import pandera_schema_to_dagster_type
from pandera.typing import Index, Series


class Ex21CompanyOwnership(pa.DataFrameModel):
    """Define table structure for extracted EX 21 data."""

    _id: Series[str] = pa.Field(alias="id", description="ID of extracted filing.")
    subsidiary: Series[str] = pa.Field(description="Name of subsidiary company.")
    loc: Series[str] = pa.Field(
        description="Location of subsidiary company.", nullable=True
    )
    #: Use str to avoid conversion errors
    own_per: Series[str] = pa.Field(
        description="Percent ownership of subsidiary company.",
        nullable=True,
        coerce=True,
    )


class Basic10kCompanyInfo(pa.DataFrameModel):
    """Define table structure for extracted basic 10k data."""

    filename: Index[str] = pa.Field(description="Name of extracted filing.")
    filer_count: Index[str] = pa.Field(
        description="Some filings have multiple blocks of company data."
    )
    block: Index[str] = pa.Field(description="Block of company data.")
    block_count: Index[str] = pa.Field(description="Some blocks occur multiple times.")
    key: Index[str] = pa.Field(description="Key within block.")
    value: Series[str] = pa.Field(description="Company info fact.")

    class Config:
        """Provide multi index options in the config."""

        multiindex_name = "time"
        multiindex_strict = True
        multiindex_coerce = True


class Sec10kExtractionMetadata(pa.DataFrameModel):
    """Define table structure extraction metadata."""

    filename: Index[str] = pa.Field(description="Name of extracted filing.")
    success: Series[bool] = pa.Field(
        description="Indicates whether filing was successfully extracted.", coerce=True
    )
    notes: Series[str] = pa.Field(
        description="Optional notes about extraction.", nullable=True
    )


class Ex21Layout(pa.DataFrameModel):
    """Define table structure for ex21 layout classification."""

    filename: Index[str] = pa.Field(description="Name of extracted filing.")
    paragraph: Series[bool] = pa.Field(
        description="Indicates whether ex21 is formatted as a paragraph or not.",
        coerce=True,
    )


ex21_extract_type = pandera_schema_to_dagster_type(Ex21CompanyOwnership)
basic_10k_extract_type = pandera_schema_to_dagster_type(Basic10kCompanyInfo)
sec10k_extract_metadata_type = pandera_schema_to_dagster_type(Sec10kExtractionMetadata)
ex21_layout_type = pandera_schema_to_dagster_type(Ex21Layout)
