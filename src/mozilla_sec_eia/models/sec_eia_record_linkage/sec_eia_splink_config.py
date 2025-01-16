"""Configuration file for the splink SEC to EIA record linkage model."""

import splink.comparison_library as cl
from splink import block_on

STR_COLS = [
    "company_name",
    "street_address",
    "street_address_2",
    "city",
    "state",
    "zip_code",
]

SHARED_COLS = [
    "record_id",
    "report_date",
    "report_year",
    "company_name",
    "company_name_no_legal",
    "company_name_mphone",
    "street_address",
    "street_address_2",
    "city",
    "state",  # could use state of incorporation from SEC
    "zip_code",
    "phone_number",
]

MATCH_COLS = ["company_name", "state", "city", "street_address"]

BLOCKING_RULES = [
    "substr(l.company_name_mphone,1,4) = substr(r.company_name_mphone,1,4)",
    "l.street_address = r.street_address",
    "substr(l.company_name_mphone,1,2) = substr(r.company_name_mphone,1,2) and l.city = r.city",
    # "substr(l.company_name_mphone,1,2) = substr(r.company_name_mphone,1,2) and array_length(list_intersect(l.street_address_list, r.street_address_list)) >= 2",
]

company_name_comparison = cl.NameComparison(
    "company_name_no_legal",
    jaro_winkler_thresholds=[0.95],
)

address_comparison = cl.LevenshteinAtThresholds(
    "street_address", distance_threshold_or_thresholds=[1]
).configure(term_frequency_adjustments=True)

state_comparison = cl.ExactMatch("state").configure(term_frequency_adjustments=True)
city_comparison = cl.NameComparison("city", jaro_winkler_thresholds=[0.9])

# blocking rules for estimating probability two random records match
deterministic_blocking_rules = [
    block_on("company_name_mphone", "company_name_mphone"),
    "jaro_winkler_similarity(r.company_name, l.company_name) >= .95 and l.city = r.city",
    "substr(l.company_name_mphone,1,4) = substr(r.company_name_mphone,1,4) and l.city = r.city and l.street_address = r.street_address",
]
