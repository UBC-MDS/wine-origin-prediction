#Author Sean McKay
FROM quay.io/jupyter/minimal-notebook:2023-11-19

COPY environment.yaml .
RUN conda env create -f environment.yaml && \
	source activate wine-origin-prediction && \
	python -m ipykernel install --user --name=wine-origin-prediction && \
	conda deactivate
