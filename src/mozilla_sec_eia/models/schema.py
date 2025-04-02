"""Define schema for PUDL models."""

import pyarrow as pa

PUDL_MODELS_SCHEMA = {
    "core_sec10k__filings": pa.schema(
        [
            pa.field(
                "filename",
                pa.string(),
                metadata={
                    "description": "Name of filing as provided by SEC data portal."
                },
            ),
            pa.field(
                "cik",
                pa.string(),
                metadata={
                    "description": "SEC Central Index Key, which uniquely identifies corporations."
                },
            ),
            pa.field(
                "company_name",
                pa.string(),
                metadata={"description": "Name of company submitting filing."},
            ),
            pa.field(
                "form_type",
                pa.string(),
                metadata={"description": "Specific version of SEC 10k filed."},
            ),
            pa.field(
                "date_filed",
                pa.date64(),
                metadata={"description": "Date filing was submitted."},
            ),
            pa.field(
                "exhibit_21_version",
                pa.string(),
                metadata={
                    "description": "Version of exhibit 21 submitted (if applicable)."
                },
            ),
            pa.field(
                "year_quarter",
                pa.string(),
                metadata={"description": "Year quarter filing applies to."},
            ),
        ],
        metadata={"description": "Metadata describing all submitted SEC 10k filings."},
    ),
    "core_sec10k__exhibit_21_company_ownership": pa.schema(
        [
            pa.field(
                "filename",
                pa.string(),
                metadata={
                    "description": "Name of filing as provided by SEC data portal."
                },
            ),
            pa.field(
                "subsidiary",
                pa.string(),
                metadata={"description": "Name of subsidiary company."},
            ),
            pa.field(
                "location",
                pa.string(),
                metadata={"description": "Location of subsidiary company."},
            ),
            pa.field(
                "ownership_percentage",
                pa.string(),
                metadata={
                    "description": "Percentage of subsidiary company owned by parent."
                },
            ),
            pa.field(
                "year_quarter",
                pa.string(),
                metadata={"description": "Year quarter filing applies to."},
            ),
        ],
        metadata={
            "description": "Company ownership data extracted from Exhibit 21 attachments to SEC 10k filings."
        },
    ),
    "core_sec10k__company_information": pa.schema(
        [
            pa.field(
                "filename",
                pa.string(),
                metadata={
                    "description": "Name of filing as provided by SEC data portal."
                },
            ),
            pa.field(
                "filer_count",
                pa.int64(),
                metadata={
                    "description": "Index company information as some filings contain information for multiple companies."
                },
            ),
            pa.field(
                "block",
                pa.string(),
                metadata={"description": "Title of block of data."},
            ),
            pa.field(
                "block_count",
                pa.int64(),
                metadata={
                    "description": "Some blocks are repeated, `block_count` defines the index of the data block."
                },
            ),
            pa.field(
                "key",
                pa.string(),
                metadata={"description": "Key within block."},
            ),
            pa.field(
                "value",
                pa.string(),
                metadata={"description": "String value of data point."},
            ),
            pa.field(
                "year_quarter",
                pa.string(),
                metadata={"description": "Year quarter filing applies to."},
            ),
        ],
        metadata={"description": "Company information extracted from SEC 10k filings."},
    ),
}
