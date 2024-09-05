FROM continuumio/miniconda3:24.7.1-0

WORKDIR /opt/dagster/app

# Build environment
COPY . .
RUN conda env create -f environment.yml && conda activate mozilla-sec-eia

EXPOSE 3000
ENTRYPOINT ["dagster", "dev"]
