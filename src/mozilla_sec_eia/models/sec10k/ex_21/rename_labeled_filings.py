"""Rename labeled filings in GCS after importing from Label Studio."""

import json
import logging

import pandas as pd

from ..utils.cloud import GCSArchive

logger = logging.getLogger(f"catalystcoop.{__name__}")


def rename_filings():
    """Rename labeled filings in GCS after importing from Label Studio.

    After importing labeled documents from Label Studio into GCS the
    labeled items in the bucket do not share the same filename of the
    unlabeled version of that document, instead the items are just
    named with the task number from Label Studio. This script looks
    inside the labeled JSON and gets the filename of the unlabeled
    document. It then labels the labeled item in GCS with that
    filename.
    """
    archive = GCSArchive()
    bucket = archive._labels_bucket

    labeled_bucket_name = "labeled/"

    for blob in bucket.list_blobs(prefix=labeled_bucket_name):
        if blob.name != labeled_bucket_name:
            logger.info(blob.name)
            file_dict = json.loads(blob.download_as_text())
            archive_name = file_dict["task"]["data"]["ocr"].split("/")[-1].split(".")[0]
            archive_filepath = f"{labeled_bucket_name}/{archive_name}"
            logger.info(archive_filepath)
            bucket.rename_blob(blob, archive_filepath)


def copy_labeled_jsons_to_new_version_folder(
    source_folder: str, dest_folder: str, tracking_df: pd.DataFrame
):
    """Copy a labeled filing JSON from one GCS blob to another.

    When conducting a new round of labeling there may be filings
    that don't have any changes in the new version. Copy labeled filings
    from the source directory in GCS (older labeling version) to the
    destination directory in GCS (latest labeling version). Only
    copy filings that are in the labeled_data_tracking CSV but not in
    the destination directory.

    Arguments:
        source_folder: The source folder name in the labels bucket
        dest_folder: The destination folder name in the labels bucket
        tracking_df: Dataframe of the labeled data tracking CSV # TODO: make package data
    """
    archive = GCSArchive()
    bucket = archive._labels_bucket

    tracked_ciks = [str(cik) for cik in tracking_df["CIK"].to_list()]
    if source_folder[-1] != "/":
        source_folder = source_folder + "/"
    if dest_folder[-1] != "/":
        dest_folder = dest_folder + "/"

    dest_blobs = bucket.list_blobs(prefix=dest_folder)
    dest_filenames = {blob.name.split("/")[-1] for blob in dest_blobs}

    for blob in bucket.list_blobs(prefix=source_folder):
        if blob.name == source_folder:
            continue
        filename = blob.name.split("/")[-1]
        if filename in dest_filenames:
            logger.info(f"{filename} is already in the destination folder. Skipping.")
            continue
        file_cik = filename.split("-")[0]
        if file_cik not in tracked_ciks:
            logger.info(f"{filename} is not currently being tracked. Skipping.")
            continue
        destination_blob_name = dest_folder + filename
        logger.info(f"Copying blob to {destination_blob_name}.")
        bucket.copy_blob(blob, bucket, destination_blob_name)
