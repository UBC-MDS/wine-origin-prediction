# Wine Origin Prediction

Author: Hina Bandukwala, Julia Everitt, Sean McKay & Yimeng Xia

Students of UBC MDS cohort 8

## Project Summary

In this project, we attempt to build a classification model using Logistic Reggression to predict wine origin. Our final classifier performed fairly well on an unseen test data cases, with accuracy rate of 98.15%. Considering the applicability of wine origin prediction, our model can be implemented for business use, providing a faster and more accurate service in classifying wine origin compared to traditional methods that require experts with sufficient knowledge and experience.

This project employs a data set comprising 13 chemical information from 178 Italian wine samples of three distinct cultivars from the same region. Originating in 1991, the data set was collected and contributed by M. Forina and Stefan Aeberhard. This data set is accessible from the UC Irvine Machine Learning Repository and can be found [here](https://archive.ics.uci.edu/dataset/109/wine).

## Report

The final report can be found [here](https://ubc-mds.github.io/wine-origin-prediction/wine_classification_report.html).

## Environment Setup - Getting Started with Docker

Install and launch Docker on your device, then clone this Github repository. Navigate to the root of this project using the command line.

Run `docker compose up jupyter-lab`. This will output two urls once complete - select the one beginning with `http://127.0.0.1:8888/lab?` and paste it into your browser to launch jupyter lab.

## Data Analysis - All at Once
Navigate to the root of the project directory using the command line and run `make clean` to remove all files generated by the analysis previously. Then, run `make all` to re-produce the analysis and re-generate all files from scratch.


## Data Analysis - Step by Step

Once docker is set up, the following commands can be used to run the analysis. Copy paste these in the terminal at the project root to reproduce our analysis step by step:

```
# Fetch data from the web, save, and split
python scripts/fetch_split_data.py --output-path='data/processed/'

# Preprocess data and save preprocessor object
python scripts/preprocessing.py --train-data ./data/processed/train.csv --test-data ./data/processed/test.csv --variable-data ./data/processed/variables.csv --output-file-path ./data/processed/scaled_wine_train.csv ./data/processed/scaled_wine_test.csv --output-preprocessor ./results/models/preprocessor_model --output-metadata-path ./data/processed/preprocessor_model

# Perform EDA and save plots
python scripts/eda.py --input_path='data/processed/scaled_wine_train.csv' --output_figure_path='results/figures/' --plot_width=150 --plot_height=100

# Fit model and optimize hyperparameters
python scripts/fit_wine_classifier.py --training-data='data/processed/train.csv' --preprocessor='results/models/preprocessor_model.pickle' --pipeline-to='results/models/' --plot-to='results/figures/' --seed=123

# Evaluate model on full train/test set
python scripts/evaluation_test.py --input-test-path='data/processed/test.csv' --pipeline-from='results/models/wine_pipeline.pickle' --target-col='class' --results-to='results/tables/'
```

## Docker Clean Up
Shut down the container by clicking `Ctrl + C` on your keyboard in the terminal where you launched Docker. Run `docker compose rm` to finish cleaning up.

## Function Testing
Tests are located in the `tests` folder. To test a function, run `pytest <file-path>` in the terminal at the root of the project.


## Dependencies

Dependencies are managed using Docker and based on the jupter minimal-notebook image, with exact details available in the Dockerfile. Additionally, the environment details can be found in `environment.yaml`

## License:

Software licensed under the [MIT License](https://spdx.org/licenses/MIT.html), non-software content licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License](https://creativecommons.org/licenses/by-nc-sa/4.0/). See LICENSE.md for more information.
