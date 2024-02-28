terraform {
  backend "gcs" {
    bucket = "f3441e415e6e5e7d-bucket-tfstate"
    prefix = "terraform/state"
  }
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "4.51.0"
    }
  }
}

variable "project_id" {
  type    = string
  default = "catalyst-cooperative-mozilla"
}

variable "catalyst_people" {
  type = list(string)
  default = [
    "trenton.bush@catalyst.coop",
    "katherine.lamb@catalyst.coop",
    "zach.schira@catalyst.coop",
    "dazhong.xia@catalyst.coop",
    "bennett.norman@catalyst.coop"
  ]
}

provider "google" {
  project = var.project_id
  region  = "us-east1"
  zone    = "us-east1-c"
}

resource "google_service_account" "mozilla_dev_sa" {
  account_id   = "mozilla-dev-sa"
  display_name = "Mozilla dev"
}

resource "random_id" "bucket_prefix" {
  byte_length = 8
}

resource "google_storage_bucket" "sec_10ks" {
  name          = "${random_id.bucket_prefix.hex}-bucket-sec-10ks"
  location      = "US"
  storage_class = "STANDARD"
  versioning {
    enabled = true
  }
}

resource "google_storage_bucket" "tfstate" {
  name          = "${random_id.bucket_prefix.hex}-tfstate"
  location      = "US"
  storage_class = "STANDARD"
  versioning {
    enabled = true
  }
}

resource "google_storage_bucket_iam_binding" "catalyst_gcs_access" {
  bucket  = google_storage_bucket.sec_10ks.name
  role    = "roles/storage.admin"
  members = ["serviceAccount:${google_service_account.mozilla_dev_sa.email}"]
}

resource "google_project_iam_binding" "catalyst_cloudsql_instance_user" {
  project = var.project_id
  role    = "roles/cloudsql.instanceUser"
  members = ["serviceAccount:${google_service_account.mozilla_dev_sa.email}"]
}

resource "google_project_iam_binding" "catalyst_cloudsql_client" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  members = ["serviceAccount:${google_service_account.mozilla_dev_sa.email}"]
}

resource "google_project_iam_binding" "catalyst_people_editors" {
  project = var.project_id
  role    = "roles/editor"
  members = [for email in var.catalyst_people : "user:${email}"]
}

resource "google_project_iam_binding" "catalyst_iam_act_as_sa" {
  project = var.project_id
  role    = "roles/iam.serviceAccountUser"
  members = [for email in var.catalyst_people : "user:${email}"]
}

# cloud sql instance for usage
resource "google_sql_database_instance" "mozilla" {
  name             = "pg-mozilla"
  database_version = "POSTGRES_14"
  region           = "us-central1"

  settings {
    tier              = "db-f1-micro"
    activation_policy = "ALWAYS"
    database_flags {
      name  = "cloudsql.iam_authentication"
      value = "on"
    }
  }
}

resource "google_sql_database" "database" {
  name     = "mozilla"
  instance = google_sql_database_instance.mozilla.name
}

resource "google_sql_user" "default_user" {
  name     = "${google_service_account.mozilla_dev_sa.account_id}@${var.project_id}.iam"
  instance = google_sql_database_instance.mozilla.name
  type     = "CLOUD_IAM_SERVICE_ACCOUNT"
}

resource "google_sql_user" "catalyst_users" {
  for_each = toset(var.catalyst_people)
  name     = each.value
  instance = google_sql_database_instance.mozilla.name
  type     = "CLOUD_IAM_USER"
}

resource "google_secret_manager_secret" "postgres_pass" {
  secret_id = "mozilla-postgres-pass"

  replication {
    automatic = true
  }
}

# DB permissions can't be granted with vanilla TF, so we have to do that manually:

# 1. gcloud sql connect <instance> --user=postgres (using password stored in Secret Manager)
# 2. CREATE ROLE mozillareadwrite;
# 3. GRANT ALL ON DATABASE <db name> TO mozillareadwrite;
# 4. GRANT mozillareadwrite to "firstname.lastname@catalyst.coop", ...;
