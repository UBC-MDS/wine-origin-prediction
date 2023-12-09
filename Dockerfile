#Author Sean McKay
FROM quay.io/jupyter/minimal-notebook:2023-11-19

#Use root to install make
USER root

COPY environment.yaml .
RUN apt-get update && apt-get install make

USER jovyan
RUN conda env create -f environment.yaml && \
	source activate wine-origin-prediction && \
	python -m ipykernel install --user --name=wine-origin-prediction && \
	conda deactivate



