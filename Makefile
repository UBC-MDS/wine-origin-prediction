all: report/_build/html/index.html

# Fetch data from the web, save, and split
data/processed/train.csv data/processed/test.csv : scripts/fetch_split_data.py
	python scripts/fetch_split_data.py \
	--output-path='data/processed/'

# Preprocess data and save preprocessor object
data/processed/scaled_wine_train.csv data/processed/scaled_wine_test.csv results/models/preprocessor_model.pickle : scripts/preprocessing.py \
data/processed/train.csv \
data/processed/test.csv \
data/processed/variables.csv
	python scripts/preprocessing.py \
	--train-data ./data/processed/train.csv \
	--test-data ./data/processed/test.csv \
	--variable-data ./data/processed/variables.csv \
	--output-file-path ./data/processed/scaled_wine_train.csv ./data/processed/scaled_wine_test.csv \
	--output-preprocessor ./results/models/preprocessor_model \
	--output-metadata-path ./data/processed/preprocessor_model

# Perform EDA and save plots
results/figures/densities_plot_by_class.png : scripts/eda.py \
data/processed/scaled_wine_train.csv
	python scripts/eda.py \
	--input_path='data/processed/scaled_wine_train.csv' \
	--output_figure_path='results/figures/' \
	--plot_width=150 \
	--plot_height=100

# Fit model and optimize hyperparameters
results/models/wine_pipeline.pickle results/figures/wine_cv_C.png : scripts/fit_wine_classifier.py \
data/processed/train.csv \
results/models/preprocessor_model.pickle
	python scripts/fit_wine_classifier.py \
	--training-data='data/processed/train.csv' \
	--preprocessor='results/models/preprocessor_model.pickle' \
	--pipeline-to='results/models/' \
	--plot-to='results/figures/' \
	--seed=123

# Evaluate model on full train/test set
results/tables/test_results.csv : scripts/evaluation_test.py \
data/processed/test.csv \
results/models/wine_pipeline.pickle
	python scripts/evaluation_test.py \
	--input-test-path='data/processed/test.csv' \
	--pipeline-from='results/models/wine_pipeline.pickle' \
	--target-col='class' \
	--results-to='results/tables/'

# Build report
report/_build/html/index.html : report/wine_classification_report.ipynb \
report/_toc.yml \
report/_config.yml \
results/tables/test_results.csv \
results/figures/densities_plot_by_class.png \
results/figures/wine_cv_C.png
	jupyter-book build report

clean:
	rm -f data/processed/train.csv data/processed/test.csv
	rm -f results/figures/densities_plot_by_class.png
	rm -f data/processed/scaled_wine_train.csv data/processed/scaled_wine_test.csv
	rm -f data/processed/preprocessor_model.csv
	rm -f results/models/preprocessor_model.pickle
	rm -f data/processed/variables.csv
	rm -f results/models/wine_pipeline.pickle results/figures/wine_cv_C.png
	rm -r results/tables/test_results.csv
	rm -rf report/_build
	
