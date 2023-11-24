import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytest
import inspect
import os
import sys

# the code line below taken from
# https://github.com/ttimbers/demo-tests-ds-analysis-python/blob/main/tests/test-count_classes.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.accuracy_curve import accuracy_curve

# helper data
np.random.seed(42)

# expected data (array, list or pd.Series)
# randomly chose one of the datatypes for each arg
param_grid = np.array([0.01,0.1,1,10,100,1000]).tolist()
train_scores = pd.Series(np.random.uniform(0.995, 1.005, 6))
cv_scores = np.random.uniform(0.97, 0.995, 6)

# incorrect input length: unequal
param_grid_5 = np.array([0.01,0.1,1,10,100])
train_scores_4 = np.random.uniform(0.995, 1.005, 4)
cv_scores_3 = np.random.uniform(0.97, 0.995, 3)

# incorrect input type: empty
param_grid_empty = np.empty(shape=(0,))
train_scores_empty = np.empty(shape=(0,))
cv_scores_empty = np.empty(shape=(0,))
    
# Test for the correct return of fig and ax objects
def test_correct_return_figax():
    fig, ax = accuracy_curve(train_scores, cv_scores, param_grid)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert len(ax.lines) == 2, "plot should have 2 trendlines, one for train and cv sets each"

# Test for the correct number of inputs
# number: (should be 3)
def test_correct_num_input():
    args = inspect.getfullargspec(accuracy_curve).args
    assert len(args) == 3, "Exactly 3 arguments must be provided"
      
# Test for correct error handing when arrays are
# of unequal length
def test_equal_input_len():
    with pytest.raises(ValueError):
        accuracy_curve(train_scores_4, cv_scores_3, param_grid_5)

# Test for correct error handing when arrays are
# empty
def test_empty_arrays():
    with pytest.raises(ValueError):
        accuracy_curve(train_scores_empty, cv_scores, param_grid)
        accuracy_curve(train_scores, cv_scores, param_grid_empty)
