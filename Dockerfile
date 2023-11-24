FROM quay.io/jupyter/minimal-notebook:2023-11-19

RUN conda env update --file environment.yaml
