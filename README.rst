mozilla-sec-eia: Developing a linkage between SEC and EIA
=======================================================================================

.. readme-intro

.. image:: https://github.com/catalyst-cooperative/mozilla-sec-eia/workflows/tox-pytest/badge.svg
   :target: https://github.com/catalyst-cooperative/mozilla-sec-eia/actions?query=workflow%3Atox-pytest
   :alt: Tox-PyTest Status

.. image:: https://img.shields.io/codecov/c/github/catalyst-cooperative/mozilla-sec-eia?style=flat&logo=codecov
   :target: https://codecov.io/gh/catalyst-cooperative/mozilla-sec-eia
   :alt: Codecov Test Coverage

.. image:: https://img.shields.io/readthedocs/catalystcoop-mozilla-sec-eia?style=flat&logo=readthedocs
   :target: https://catalystcoop-mozilla-sec-eia.readthedocs.io/en/latest/
   :alt: Read the Docs Build Status

.. image:: https://img.shields.io/pypi/v/catalystcoop.mozilla-sec-eia?style=flat&logo=python
   :target: https://pypi.org/project/catalystcoop.mozilla-sec-eia/
   :alt: PyPI Latest Version

.. image:: https://img.shields.io/conda/vn/conda-forge/catalystcoop.mozilla-sec-eia?style=flat&logo=condaforge
   :target: https://anaconda.org/conda-forge/catalystcoop.mozilla-sec-eia
   :alt: conda-forge Version

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black>
   :alt: Any color you want, so long as it's black.

This repo contains exploratory development for an SEC-EIA linkage.

Usage
-----
Utility functions for accessing and working with 10k filings as well as their exhibit
21 attachments can be found in 'src/mozilla_sec_eia/utils.py'. The base class is the
``GCSArchive`` which provides an interface to archived filings on GCS. To instantiate
this class, the following environment variables need to be set, or defined in a ``.env``
file:

``GCS_BUCKET_NAME``
``GCS_METADATA_DB_INSTANCE_CONNECTION``
``GCS_IAM_USER``
``GCS_METADATA_DB_NAME``

This code sample shows how to use the class to fetch filings from the archive:

.. code-block:: python

   from mozilla_sec_eia.utils import GCSArchive
   archive = GCSArchive()

   # Get metadata from postgres instance
   metadata_df = archive.get_metadata()

   # Do some filtering to get filings of interest
   filings = metadata_df.loc[...  # Get rows from original df

   # This will download and cache filings locally for later use
   # Successive calls to get_filings will not re-download filings which are already cahced
   downloaded_filings = archive.get_filings(filings)

   # Get exhibit 21's and extract subsidiary data
   for filing in downloaded_filings:
           cool_extraction_model(filing.get_ex_21().as_image())

Labeling
--------
We are using `Label Studio <https://labelstud.io/>`_ to create training data
for fine-tuning the Ex. 21 extraction model. The very preliminary workflow
for labeling data is as follows:

* For each filing that you want to label, follow notebook 7 to create the
  inputs for Label Studio.
   * First, create a PDF of the filing. Then, extract the bounding boxes
   around each word and create a "task" JSON and image for each Ex. 21 table
   that will be used in Label Studio
* Upload these JSONs and images to the same bucket in GCS (the "unlabeled"
  bucket by default).
* `Install Label Studio <https://labelstud.io/guide/install>`
* Start Label Studio locally and create a project.
* Under Settings, set the template/config for the project with the config
  found in ``labeling-configs/labeling-config.xml``. This should create the
  correct entity labels and UI setup.
* Connect GCS to Label Studio:
  * Follow `these directions 
    <https://labelstud.io/guide/storage#Google-Cloud-Storage>`
    to connect the bucket to Label Studio.
  * Specific settings: Filter files for only JSONs (these are your tasks).
    Leave "Treat every bucket object as a source file" disabled.
    Add the service account authentication JSON for your bucket.
  * Additionally add a Target Storage bucket (the "labeled" bucket by
    default).
* Import data and label Ex. 21 tables.
* Sync with target storage.
* Update the ``labeled_data_tracking.csv`` with the new filings you've
  labeled.


About Catalyst Cooperative
=======================================================================================
`Catalyst Cooperative <https://catalyst.coop>`__ is a small group of data
wranglers and policy wonks organized as a worker-owned cooperative consultancy.
Our goal is a more just, livable, and sustainable world. We integrate public
data and perform custom analyses to inform public policy (`Hire us!
<https://catalyst.coop/hire-catalyst>`__). Our focus is primarily on mitigating
climate change and improving electric utility regulation in the United States.

Contact Us
----------
* For general support, questions, or other conversations around the project
  that might be of interest to others, check out the
  `GitHub Discussions <https://github.com/catalyst-cooperative/pudl/discussions>`__
* If you'd like to get occasional updates about our projects
  `sign up for our email list <https://catalyst.coop/updates/>`__.
* Want to schedule a time to chat with us one-on-one? Join us for
  `Office Hours <https://calend.ly/catalyst-cooperative/pudl-office-hours>`__
* Follow us on Twitter: `@CatalystCoop <https://twitter.com/CatalystCoop>`__
* More info on our website: https://catalyst.coop
* For private communication about the project or to hire us to provide customized data
  extraction and analysis, you can email the maintainers:
  `pudl@catalyst.coop <mailto:pudl@catalyst.coop>`__
