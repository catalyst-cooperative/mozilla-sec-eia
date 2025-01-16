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
    filer_count: Index[int] = pa.Field(
        description="Some filings have multiple blocks of company data."
    )
    block: Index[str] = pa.Field(description="Block of company data.")
    block_count: Index[int] = pa.Field(description="Some blocks occur multiple times.")
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


class Sec10kCoreTable(pa.DataFrameModel):
    """Define table structure for core SEC companies table."""

    sec_company_id: Series[str] = pa.Field(
        description="Assigned identifier for the company."
    )
    filename: Series[str] = pa.Field(description="Name of extracted filing.")
    central_index_key: Series[str] = pa.Field(
        description="Identifier of the company in SEC database."
    )
    report_date: Series[pa.DateTime] = pa.Field(
        description="Report date of the record."
    )
    company_name: Series[str] = pa.Field(
        description="Cleaned name of the company with legal terms expanded."
    )
    utility_id_eia: Series[int] = pa.Field(
        description="EIA utility identifier for the company. Matched via record linkage model.",
        nullable=True,
    )
    street_address: Series[str] = pa.Field(
        description="Street address of the company.", nullable=True
    )
    street_address_2: Series[str] = pa.Field(
        description="Secondary street address of the company.", nullable=True
    )
    phone_number: Series[str] = pa.Field(
        description="Phone number of company.", nullable=True
    )
    city: Series[str] = pa.Field(
        description="The city where the company is located.", nullable=True
    )
    state: Series[str] = pa.Field(
        description="Two letter state code where the company is located.", nullable=True
    )
    state_of_incorporation: Series[str] = pa.Field(
        description="Two letter state code where the company is located.", nullable=True
    )
    zip_code: Series[str] = pa.Field(
        description="5 digit zip code where the company is located.", nullable=True
    )
    company_name_raw: Series[str] = pa.Field(
        description="The raw company name.", nullable=True
    )
    date_of_name_change: Series[pa.DateTime] = pa.Field(
        description="Date of last name change of the company.", nullable=True
    )
    location_of_inc: Series[str] = pa.Field(
        description="Cleaned location of incorporation of the company.", nullable=True
    )
    company_name_no_legal: Series[str] = pa.Field(
        description="Company name with legal terms stripped, e.g. LLC", nullable=True
    )
    company_name_mphone: Series[str] = pa.Field(
        description="Metaphone of the company name, could used for record linkage."
    )
    files_10k: Series[bool] = pa.Field(
        description="Indicates whether the company files a 10-K."
    )


class Sec10kOutputTable(pa.DataFrameModel):
    """Define table structure for output parents and subsidiaries table."""

    parent_company_cik: Series[str] = pa.Field(
        description="CIK of the company's parent company.", nullable=True
    )
    own_per: Series[float] = pa.Field(
        description="Parent company's ownership percentage of the company.",
        nullable=True,
    )


class EiaCompanies(pa.DataFrameModel):
    """Define table structure for EIA owner and operator companies table."""

    company_name: Series[str] = pa.Field(
        description="Cleaned name of the owner or operator company with legal terms expanded."
    )
    street_address: Series[str] = pa.Field(
        description="Street address of the company.", nullable=True
    )
    street_address_2: Series[str] = pa.Field(
        description="Secondary street address of the company.", nullable=True
    )
    utility_id_eia: Series[int] = pa.Field(
        description="EIA utility identifier for the company.", coerce=True
    )
    company_name_raw: Series[str] = pa.Field(description="The raw company name.")
    # TODO: What type for type expression?
    report_date: Series[pa.DateTime] = pa.Field(
        description="Report date of the record."
    )
    report_year: Series[int] = pa.Field(
        description="Report year of the record.", coerce=True
    )
    city: Series[str] = pa.Field(
        description="The city where the company is located.", nullable=True
    )
    state: Series[str] = pa.Field(
        description="Two letter state code where the company is located.", nullable=True
    )
    zip_code: Series[str] = pa.Field(
        description="5 digit zip code where the company is located.", nullable=True
    )
    phone_number: Series[str] = pa.Field(
        description="Phone number of company.", nullable=True
    )
    company_name_no_legal: Series[str] = pa.Field(
        description="Company name with legal terms stripped, e.g. LLC", nullable=True
    )
    company_name_mphone: Series[str] = pa.Field(
        description="Metaphone of the company name, could used for record linkage."
    )


ex21_extract_type = pandera_schema_to_dagster_type(Ex21CompanyOwnership)
basic_10k_extract_type = pandera_schema_to_dagster_type(Basic10kCompanyInfo)
sec10k_extract_metadata_type = pandera_schema_to_dagster_type(Sec10kExtractionMetadata)
ex21_layout_type = pandera_schema_to_dagster_type(Ex21Layout)
eia_layout_type = pandera_schema_to_dagster_type(EiaCompanies)
sec10k_output_layout_type = pandera_schema_to_dagster_type(Sec10kOutputTable)
sec10k_core_layout_type = pandera_schema_to_dagster_type(Sec10kCoreTable)
