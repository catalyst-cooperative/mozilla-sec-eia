from importlib import resources

import pandas as pd

labels_filename = "ex21_labels_with_paragraph.csv"
labels_df = pd.read_csv(
    resources.files("mozilla_sec_eia.package_data.validation_data") / labels_filename,
    comment="#",
)
hist_filename = "ex21_layout_histogram.csv"
layout_hist_df = pd.read_csv(
    resources.files("mozilla_sec_eia.package_data.validation_data") / hist_filename,
    comment="#",
)
paragraph_filenames = layout_hist_df[layout_hist_df["Layout Type"] == "Paragraph"][
    "Filename"
].unique()

non_paragraph_df = labels_df[~labels_df["Filename"].isin(paragraph_filenames)]

labels_filename_new = "ex21_labels.csv"
non_paragraph_df.to_csv(
    resources.files("mozilla_sec_eia.package_data.validation_data")
    / labels_filename_new,
    index=False,
)
