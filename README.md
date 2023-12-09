# Wine Origin Prediction

author: Hina Bandukwala, Julia Everitt, Sean McKay & Yimeng Xia

Students of UBC MDS cohort 8

## Project Summary

In this project, we attempt to build a classification model using Logistic Reggression to predict wine origin. Our final classifier performed fairly well on an unseen test data cases, with accuracy rate of 98.15%. Considering the applicability of wine origin prediction, our model can be implemented for business use, providing a faster and more accurate service in classifying wine origin compared to traditional methods that require experts with sufficient knowledge and experience.

This project employs a data set comprising 13 chemical information from 178 Italian wine samples of three distinct cultivars from the same region. Originating in 1991, the data set was collected and contributed by M. Forina and Stefan Aeberhard. This data set is accessible from the UC Irvine Machine Learning Repository and can be found [here](https://archive.ics.uci.edu/dataset/109/wine).

## Report

The final report can be found [here](https://ubc-mds.github.io/wine-origin-prediction/wine_classification_report.html).

## Setup Option 1 - Running from Docker

Install Docker on your device and clone this Github repository. Navigate to the project directory.

Run the following to build and launch a jupyter notebook environment in local.

``` bash
docker build . --tag <tag_name>
docker run --rm -it -v /$(pwd):/home/jovyan/work -p 8888:8888 <tag_name>
```

Then navigate to http://127.0.0.1:8888/lab/ and launch report.ipynb under the src folder

OR

Run `docker compose up` (this will pull the latest tag from dockerhub)

Then navigate to http://127.0.0.1:8888/lab/ and launch report.ipynb under the src folder

## Setup Option 2 - Using conda

First time running this project, please run the following from the root of this repository to create the environment:

``` bash
conda env create --file environment.yaml
```

``` bash
conda activate wine-origin-prediction
```

Open the project folder in Visual Studio Code and click "Restart".

or

Open the project folder in Jupyter Lab and under the "Kernel" menu click "Restart Kernel and Run All Cells..."


# Data Analysis

Open the project in the Terminal and navigate to the scripts folder, then run the following commands to run the analysis:

```
# Fetch data from the web, save, and split
python scripts/fetch_split_data.py --output-raw-path='data/raw' --output-processed-path='data/processed/'

# Preprocess data and save preprocessor object
python scripts/preprocessing.py --train-data ./data/processed/train.csv --test-data ./data/processed/test.csv --variable-data ./data/raw/variables.csv --output-file-path ./data/processed/scaled_wine_train.csv ./data/processed/scaled_wine_test.csv --output-preprocessor ./results/models/preprocessor_model --output-metadata-path ./data/processed/preprocessor_model

# Perform EDA and save plots
python scripts/eda.py --input_path='data/processed/scaled_wine_train.csv' --output_figure_path='results/figures/' --plot_width=150 --plot_height=100

# Fit model and optimize hyperparameters
python scripts/fit_wine_classifier.py --training-data='data/processed/train.csv' --preprocessor='results/models/preprocessor_model.pickle' --pipeline-to='results/models/' --plot-to='results/figures/' --seed=123

# Evaluate model on full train/test set
python scripts/evaluation_test.py --input-test-path='data/processed/test.csv' --pipeline-from='results/models/wine_pipeline.pickle' --target-col='class' --results-to='results/tables/'
```

# Running the test suite

Open the project in terminal and navigate to the tests folder, then run the following command

```
pytest
```

[Contribution guidelines for this project](docs/CONTRIBUTING.md)

## Dependencies

Please ensure you have the following dependencies installed:

-   `conda` (version 23.9.0 or higher)

-   `nb_conda_kernels`

-   Python and packages listed in `environment.yaml` [here](https://github.com/UBC-MDS/wine-origin-prediction/blob/main/environment.yaml)

## License:

Software licensed under the [MIT License](https://spdx.org/licenses/MIT.html), non-software content licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License](https://creativecommons.org/licenses/by-nc-sa/4.0/). See LICENSE.md for more information.
