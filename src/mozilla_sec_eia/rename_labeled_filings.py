from mozilla_sec_eia.utils import GCSArchive
import json

archive = GCSArchive(GCS_BUCKET_NAME="labeled-ex21-filings")

labeled_bucket_name = "labeled"

for blob in archive._bucket.list_blobs(prefix=labeled_bucket_name):
    if blob.name != labeled_bucket_name + "/":
        print(blob.name)
        file_dict = json.loads(blob.download_as_text())
        archive_name = file_dict["task"]["data"]["ocr"].split("/")[-1].split(".")[0]
        archive_filepath = f"{labeled_bucket_name}/{archive_name}"
        print(archive_filepath)
        archive._bucket.rename_blob(blob, archive_filepath)
