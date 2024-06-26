"""Rename labeled filings in GCS after importing from Label Studio."""

import json
import logging

from mozilla_sec_eia.utils.cloud import GCSArchive

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
    bucket = archive._filings_bucket

    labeled_bucket_name = "labeled"

    for blob in bucket.list_blobs(prefix=labeled_bucket_name):
        if blob.name != labeled_bucket_name + "/":
            logger.info(blob.name)
            file_dict = json.loads(blob.download_as_text())
            archive_name = file_dict["task"]["data"]["ocr"].split("/")[-1].split(".")[0]
            archive_filepath = f"{labeled_bucket_name}/{archive_name}"
            logger.info(archive_filepath)
            bucket.rename_blob(blob, archive_filepath)
