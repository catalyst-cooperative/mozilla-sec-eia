ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim
RUN pip --no-cache-dir install mlflow==2.12.1 && \
    pip --no-cache-dir google-cloud-storage==2.17.0  && \
    pip --no-cache-dir psycopg2-binary=2.9.9  && \
    pip --no-cache-dir pg8000==1.31.2
