pudl-models: ML models developed for PUDL
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

About
-----
The `PUDL <https://github.com/catalyst-cooperative/pudl>`__ project makes US energy data free and open
for all. For more information, see the PUDL repo and `website <https://catalyst.coop/pudl/>`__.

This repo implements machine learning models which support PUDL. The types of
modelling performed here include record linkage between datasets, and extracting
structured data from unstructured documents. The outputs of these models then feed
into PUDL tables, and are distributed in the PUDL data warehouse.

Project Structure
-----------------
This repo is split into two main sections, with shared tooling being implemented in
``src/mozilla_sec_eia/library`` and actual models implemented in
``src/mozilla_sec_eia/models``.

Models
^^^^^^
Each model is contained in its own Dagster
`code location <https://docs.dagster.io/concepts/code-locations>`__. This keeps models
isolated from each other, allowing finetuned dependency management, and provides useful
organization in the Dagster UI. To add a new model, you must create a new python module
in the ``src/mozilla_sec_eia/models/`` directory. This module should define a single
Dagster ``Definitions`` object which can be imported from the top-level of the module.
For reference on how to structure a code location, see
``src/mozilla_sec_eia/models/sec10k/`` for an example. After creating a new model,
it must be added to
`workspace.yaml <https://docs.dagster.io/concepts/code-locations/workspace-files>`__.

There are three types of dagster `jobs <https://docs.dagster.io/concepts/assets/asset-jobs>`__
expected in a model code location:

* **Production Jobs**: Production jobs define a pipeline to execute a model and produce
  outputs which typicall feed into PUDL.
* **Validation Jobs**: Validation jobs are used to test/validate models. They will be
  run in a single process with an
  `mlflow <https://mlflow.org/docs/latest/tracking.html>`__ run backing
  them to allow logging results to a tracking server.
* **Training Jobs**: Training jobs are meant to train models and log results with
  mlflow for use in production jobs.

There are helper functions in ``src/mozilla_sec_eia/library/model_jobs.py`` for
constructing each of these jobs. These functions help to ensure each job will
use the appropriate executor and supply the job with necessary resources.

Library
^^^^^^^
There's generic shared tooling for ``pudl-models`` defined in
``src/mozilla_sec_eia/library/``. This includes the helper fucntions for
constructing dagster jobs discussed above, as well as useful methods for computing
validation metrics, and an interface to our mlflow tracking server integrated with
our tracking server.

MlFlow
""""""
We use a remote `mlflow tracking <https://mlflow.org/docs/latest/tracking.html>`__ to aide in the
development and management of ``pudl-models``. In the ``mlflow`` module, there are
several dagster resources and IO-managers that can be used in any models to allow simple
seamless interface to the server.

.. TODO: Add mlflow resource/io-manager examples


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
