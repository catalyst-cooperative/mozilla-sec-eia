"""Module for creating the SEC company output table which connects to EIA company data."""


# the input to this method is "core_sec_10k__parents_and_subsidiaries"
def sec_output_table():
    """Connect SEC to EIA and format an output table."""
    # run record linkage to connect SEC to EIA?
    # add a utility_id_eia column onto the core table
    # drop the following columns: company_name_no_legal, company_name_mphone, any other intermediate columns
    pass
