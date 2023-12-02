# fit_wine_classifier.py
# author: Julia Everitt
# date: 2023-11-28

import click
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

# Import the optimization and accuracy curve
# plotting functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.optimization import optimize_hyperparameter
from src.accuracy_curve import accuracy_curve


@click.command()
@click.option('--training-data', type=str, help="Path to training data")
@click.option('--preprocessor', type=str, help="Path to preprocessor object")
@click.option('--pipeline-to', type=str, help="Path to directory where the pipeline object will be written")
@click.option('--plot-to', type=str, help="Path to directory where the cv plot will be written")
@click.option('--seed', type=int, help="Random seed", default=123)

def main(training_data, preprocessor, pipeline_to, plot_to, seed):
    '''Fits a wine region predictor to the training data 
    and saves the pipeline object.'''
    np.random.seed(seed)

    # read in data & preprocessor, split data into X and y
    wine_train = pd.read_csv(training_data)
    wine_train_X = wine_train.drop(columns=['class'])
    wine_train_y = wine_train['class']
    wine_preprocessor = pickle.load(open(preprocessor, "rb"))

    # create pipeline
    pipe_lr = make_pipeline(
        wine_preprocessor, LogisticRegression(max_iter=2000)
    )

    # tune model
    param_grid = {
        "logisticregression__C": np.array([0.01, 0.1, 1, 10, 100, 1000])
    }
    
    lr_fit = optimize_hyperparameter(wine_train_X, 
                                        wine_train_y, 
                                        pipe_lr, 
                                        param_grid, 
                                        njobs=-1, 
                                        cv=10, 
                                        method=GridSearchCV)
    

    with open(os.path.join(pipeline_to, "wine_pipeline.pickle"), 'wb') as f:
        pickle.dump(lr_fit, f)

    train_scores = lr_fit.cv_results_["mean_train_score"]
    cv_scores = lr_fit.cv_results_["mean_test_score"]
    
    plot = accuracy_curve(train_scores, cv_scores, param_grid["logisticregression__C"])
    plt.savefig(os.path.join(plot_to, "wine_cv_C.png"))

if __name__ == '__main__':
    main()