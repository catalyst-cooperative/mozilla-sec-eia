sec10k: Extracting company ownership data from sec10k documents
=======================================================================================

This repo contains exploratory development for an SEC-EIA linkage.

Models
------
Basic 10k
^^^^^^^^^
The extraction model for basic 10k company information is very simple and requires no
training. This model is implemented as a simple rules based parser that finds key-value
pairs containing company information, which is embedded in a header for all 10k filings.

Exhibit 21
^^^^^^^^^^
Exhibit 21 extraction is much more complicated and requires pretrained models that are
cached with our mlflow tracking server. Currently, there are 2 models which are
implemented in the ``notebooks/`` directory. These notebooks use
`Dagstermill <https://docs.dagster.io/integrations/dagstermill/using-notebooks-with-dagster>`_
so they can be run interactively like any normal Jupyter Notebook, or run in a Dagster
job.

Extraction
""""""""""
The primary extraction model is implemented in the ``notebooks/exhibit21_extractor.ipynb``.
This model is based on
`layoutlm <https://huggingface.co/microsoft/layoutlmv3-base>`_ with custom inference logic
to construct a table of ownership information from an exhibit 21 document. Both the
layoutlm model and the inference model are logged separately with mlflow. This
separation between the models allows for testing minor modifications to the inference
portion with the same pretrained layoutlm model.

There are currently two configuration parameters that used by the extraction model
notebook:

* ``layoutlm_training_run``: This should be an existing mlflow run name, which was used
  to train layoutlm, and has a logged model associated with it. If ``None`` layoutlm
  will be trained when the notebook is run, and the new training run will be used for
  inference and validation.
* ``training_data_version``: This should point to a GCS folder containing training
  data to use with layoutlm. If ``layoutlm_training_run`` is set, then this parameter
  doesn't matter, as layoutlm will not be re-trained when the notebook is executed.

The notebook also depends on several upstream dagster assets, which produce training and
validation datasets. Using upstream assets allows these datasets, which are relatively
expensive to produce, to be easily cached and reused while interating on the model.
These upstream assets need to be produced before the notebook can be run. They should
also be re-materialized if you want to modify the training or validation data, otherwise
the notebook can be re-run as many times as desired with existing data.

Layout Classification
"""""""""""""""""""""
The second model is a classifier, which labels filings as either having a 'paragraph'
layout or not. This is done because the extraction model performs poorly on documents
formatted as paragraphs rather than tables. For now we will likely just filter out these
results, but we could also develop a separate extraction model which handles these
documents better.

This model is located in ``notebooks/exhibit21_layout_classifier.ipynb``, and it also
depends on upstream assets to produce training data, which will need
to be produced before running the notebook.

Training the Models
"""""""""""""""""""
The models are trained by running the notebooks. This can be done either interactively
like a normal notebook or through dagster directly.

Whether running interactively or with dagster, you will first need to produce the
upstream data assets:

1. Launch dagster from the repo root with the ``dagster dev`` command
2. Locate the training Job in question using the webui
3. Select the upstream assets by holding down the shift key and clicking on each
   asset excluding the notebook asset
4. Click ``Materialize all`` in the UI

Once this is complete, you can simply launch ``Jupyter`` and run the notebooks
interactively as you would any other notebook. The first cell loads the upstream
assets and sets configuration. You can modify the configuration directly in the
notebook as normal.

To run the notebook in dagster, you simply execute it like any other normal asset.
You can first set configuration in the dagster launchpad if desired, and when it
completes executing, you can click on the asset to view the fully rendered notebook.

Usage
-----

Helper Tools
^^^^^^^^^^^^
Utility functions for accessing and working with 10k filings as well as their exhibit
21 attachments can be found in 'src/mozilla_sec_eia/utils/cloud.py'. The base class is
the ``GCSArchive`` which provides an interface to archived filings on GCS. To
instantiate this class, the following environment variables need to be set, or defined
in a ``.env`` file:

``GCS_BUCKET_NAME``
``GCS_METADATA_DB_INSTANCE_CONNECTION``
``GCS_IAM_USER``
``GCS_METADATA_DB_NAME``
``GCS_PROJECT``
``MLFLOW_TRACKING_URI``

This code sample shows how to use the class to fetch filings from the archive:

.. code-block:: python

   from mozilla_sec_eia.utils.cloud import GCSArchive
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
  inputs for Label Studio. This notebook first creates a PDF of the filing.
  Then, it extracts the bounding boxes around each word and create a "task"
  JSON and image for each Ex. 21 table that will be used in Label Studio.
* Upload these JSONs and images to the same bucket in GCS (the "unlabeled"
  bucket by default).
* `Install Label Studio <https://labelstud.io/guide/install>`_
* Start Label Studio locally and create a project.
* Under Settings, set the template/config for the project with the config
  found in ``labeling-configs/labeling-config.xml``. This should create the
  correct entity labels and UI setup.
* Connect GCS to Label Studio by following `these directions
  <https://labelstud.io/guide/storage#Google-Cloud-Storage>`_
* Specific Label Studio settings: Filter files for only JSONs
  (these are your tasks). Leave "Treat every bucket object as a source file"
  disabled. Add the service account authentication JSON for your bucket.
* Additionally add a Target Storage bucket (the "labeled" bucket by
  default).
* Import data and label Ex. 21 tables.
* Sync with target storage.
* Update the ``labeled_data_tracking.csv`` with the new filings you've
  labeled.
* Run the ``rename_labeled_filings.py`` script to update labeled file
  names in the GCS bucket with their SEC filename.


About Catalyst Cooperative
---------------------------------------------------------------------------------------
`Catalyst Cooperative <https://catalyst.coop>`__ is a small group of data
wranglers and policy wonks organized as a worker-owned cooperative consultancy.
Our goal is a more just, livable, and sustainable world. We integrate public
data and perform custom analyses to inform public policy (`Hire us!
<https://catalyst.coop/hire-catalyst>`__). Our focus is primarily on mitigating
climate change and improving electric utility regulation in the United States.

Contact Us
^^^^^^^^^^
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
