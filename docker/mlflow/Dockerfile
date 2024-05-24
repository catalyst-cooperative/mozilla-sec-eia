ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim
ARG MLFLOW_VERSION=2.12.1
RUN pip --no-cache-dir install \
    mlflow==${MLFLOW_VERSION} \
    google-cloud-storage \
    psycopg2-binary \
    pg8000
