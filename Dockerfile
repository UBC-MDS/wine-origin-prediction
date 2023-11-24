FROM quay.io/jupyter/minimal-notebook:2023-11-19

COPY environment.yaml .
RUN conda env update --file environment.yaml