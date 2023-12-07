import pandas as pd
import numpy as np
import pytest
import sys
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression

# Import the optimization function
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.optimization import optimize_hyperparameter

# Test data
param_grid = param_grid = {
    "logisticregression__C": np.array([0.01,0.1,1,10,100,1000])
}

param_grid_empty = {}

param_grid_invalid = {
    "logistic--regression__C": np.array([0.01,0.1,1,10,100,1000])
}

pipe_lr = make_pipeline(
    StandardScaler(), LogisticRegression(max_iter=2000)
)

X_train_empty = pd.DataFrame([])
y_train_empty = pd.DataFrame([])

X_train_3_feat_8_val_2_class = pd.DataFrame({
        'a': [1, 2, 3, 4, 5, 6, 7, 8],
        'b': [100, 200, 300, 400, 500, 600, 700, 800],
        'c': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    })
y_train_3_feat_8_val_2_class = pd.DataFrame({'Label': ['class1', 'class2', 'class1', 
                                  'class1', 'class2', 'class1',
                                  'class1', 'class2']})

X_train_2_feat_15_val_3_class = pd.DataFrame({
        'a': np.ones(15),
        'b': np.zeros(15)
    })
y_train_2_feat_15_val_3_class = pd.DataFrame({'Label': ['class1', 'class3', 'class2', 'class3',
                                                        'class3', 'class1', 'class2', 'class1',
                                                        'class1', 'class2', 'class3', 'class3',
                                                        'class1', 'class2', 'class2']})


X_train_small = pd.DataFrame({
        'a': [4, 5],
        'b': [700, 800]
    })

y_train_small = pd.DataFrame({'Label': ['class2', 'class1']})

# Test for correct output type
def test_optimization_returns_gridsearchcv():
    output = optimize_hyperparameter(X_train_3_feat_8_val_2_class, 
                                    y_train_3_feat_8_val_2_class, 
                                    pipe_lr, 
                                    param_grid)
    assert isinstance(output, GridSearchCV), "optimization should return a GridSearchCV object"

# Test hyperparameters have been optimized
def test_optimization_successful():
    output = optimize_hyperparameter(X_train_2_feat_15_val_3_class, 
                                    y_train_2_feat_15_val_3_class, 
                                    pipe_lr, 
                                    param_grid).best_params_
    assert isinstance(output, dict), "optimal hyperparameters should be accessible in a dictionary object"
    assert len(output) != 0, "optimal hyperparameter dict should not be empty"

# Test cross validation results are available and of expected type
def test_optimization_cv_available():
    output = optimize_hyperparameter(X_train_2_feat_15_val_3_class, 
                                    y_train_2_feat_15_val_3_class, 
                                    pipe_lr, 
                                    param_grid).cv_results_
    assert isinstance(output['mean_train_score'], np.ndarray), "cv results should be available as array with cv_results_ attribute"
    assert isinstance(output['mean_test_score'], np.ndarray), "cv results should be available as array with cv_results_ attribute"
    assert len(output['mean_train_score']) != 0, "cv train results should not be empty"
    assert len(output['mean_test_score']) != 0, "cv test results should not be empty"

# Test exception is thrown for empty paramgrid
def test_empty_paramgrid():
   with pytest.raises(ValueError):
      optimize_hyperparameter(X_train_3_feat_8_val_2_class, 
                              y_train_3_feat_8_val_2_class, 
                              pipe_lr, 
                              param_grid_empty)

# Test exception is thrown for empty train/test df
def test_empty_train_data():
    with pytest.raises(ValueError):
      optimize_hyperparameter(X_train_empty, 
                              y_train_empty, 
                              pipe_lr, 
                              param_grid)
      
# Test exception is thrown for train/test df with too few data
# points for cross validation
def test_empty_train_data():
    with pytest.raises(ValueError):
      optimize_hyperparameter(X_train_small, 
                              y_train_small, 
                              pipe_lr, 
                              param_grid)