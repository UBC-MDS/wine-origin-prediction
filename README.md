# Wine Origin Prediction

author: Hina Bandukwala, Julia Everitt, Sean McKay & Yimeng Xia

Students of UBC MDS cohort 8

## Project Summary

In this project, we attempt to build a classification model using Logistic Reggression to predict wine origin. Our final classifier performed fairly well on an unseen test data cases, with accuracy rate of 98.15%. Considering the applicability of wine origin prediction, our model can be implemented for business use, providing a faster and more accurate service in classifying wine origin compared to traditional methods that require experts with sufficient knowledge and experience.

This project employs a data set comprising 13 chemical information from 178 Italian wine samples of three distinct cultivars from the same region. Originating in 1991, the data set was collected and contributed by M. Forina and Stefan Aeberhard. This data set is accessible from the UC Irvine Machine Learning Repository and can be found [here](https://archive.ics.uci.edu/dataset/109/wine).

## Report

The final report can be found [here](https://ubc-mds.github.io/wine-origin-prediction/docs/report.html).

## Data Analysis

Clone the repository and navigate to the project directory.

First time running this project, please run the following from the root of this repository:

``` bash
conda env create --file environment.yaml
```

``` bash
conda activate wine-origin-prediction
```

Open `src/report.ipynb` in Visual Studio Code and click "Restart".

or

Open `src/report.ipynb` in Jupyter Lab and under the "Kernel" menu click "Restart Kernel and Run All Cells..."

## Running from docker

Run the following to build and launch a jupyter notebook environment in local.

``` bash
docker build . --tag <tag_name>
docker run --rm -it -v /$(pwd):/home/jovyan/work -p 8888:8888 <tag_name>
```

Then navigate to http://127.0.0.1:8888/lab/ and launch report.ipynb under the src folder

## Running using docker compose

Run `docker compose up` (this will pull the latest tag from dockerhub)

Then navigate to http://127.0.0.1:8888/lab/ and launch report.ipynb under the src folder

## Dependencies

Please ensure you have the following dependencies installed:

-   `conda` (version 23.9.0 or higher)

-   `nb_conda_kernels`

-   Python and packages listed in `environment.yaml`

## License:

Software licensed under the [MIT License](https://spdx.org/licenses/MIT.html), non-software content licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License](https://creativecommons.org/licenses/by-nc-sa/4.0/). See LICENSE.md for more information.
