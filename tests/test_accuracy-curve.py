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

# expected data (numpy array, length: 6)
param_grid = np.array([0.01, 0.1, 1, 10, 100, 1000])
train_scores = np.random.uniform(0.995, 1.005, 6)
cv_scores = np.random.uniform(0.995, 1.005, 6)

# expected data (numpy array, length: 4)
param_grid_4 = np.array([1, 1, 1, 1])
train_scores_4 = np.array([1, 1.5, 2, 2.5])
cv_scores_4 = np.array([0, 1, 0, 1])

# expected data (numpy array, length: 2)
param_grid_2 = np.array([1, 1])
train_scores_2 = np.array([1, 1.5])
cv_scores_2 = np.array([0, 1])

# expected data (list, length: 6)
param_grid_list = param_grid.tolist()
train_scores_list = train_scores.tolist()
cv_scores_list = cv_scores.tolist()

# expected data (Series, length 6)
param_grid_series = pd.Series(param_grid)
train_scores_series = pd.Series(train_scores)
cv_scores_series = pd.Series(cv_scores)

# incorrect input type: length 1
param_grid_1 = np.array([1])
train_scores_1 = np.array([1])
cv_scores_1 = np.array([1])

# incorrect input type: empty
param_grid_empty = np.empty(shape=(0,))
train_scores_empty = np.empty(shape=(0,))
cv_scores_empty = np.empty(shape=(0,))


# Test for the correct return of fig and ax objects
# with correct inputs
# Expected inputs: numpy arrays of length 6
def test_correct_figax_np6():
    fig, ax = accuracy_curve(train_scores, cv_scores, param_grid)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert len(ax.lines) == 2, "plot should have 2 trendlines"

# Expected inputs: numpy arrays of length 4
def test_correct_figax_np4():
    fig, ax = accuracy_curve(train_scores_4, cv_scores_4, param_grid_4)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert len(ax.lines) == 2

# Expected inputs: numpy arrays of length 2
def test_correct_figax_np2():
    fig, ax = accuracy_curve(train_scores_2, cv_scores_2, param_grid_2)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert len(ax.lines) == 2

# Expected inputs: lists of length 6
def test_correct_figax_lst6():
    fig, ax = accuracy_curve(train_scores_list, cv_scores_list, param_grid_list)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert len(ax.lines) == 2

# Expected inputs: lists of length 4
def test_correct_figax_lst4():
    fig, ax = accuracy_curve(train_scores_list[:4], cv_scores_list[:4], param_grid_list[:4])
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert len(ax.lines) == 2

# Expected inputs: pd.Series of length 6
def test_correct_figax_srs6():
    fig, ax = accuracy_curve(train_scores_series, cv_scores_series, param_grid_series)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert len(ax.lines) == 2

# Test for the correct number of inputs
# number: (should be 3)
def test_correct_num_input():
    args = inspect.getfullargspec(accuracy_curve).args
    assert len(args) == 3, "Exactly 3 arguments must be provided"
      
# Test for correct error handing when arrays are
# of unequal length
def test_unequal_input_len():
    with pytest.raises(ValueError):
        accuracy_curve(train_scores, cv_scores_4, param_grid_4)

# Test for correct error handing when length of arrays is
# NOT greater than 1
def test_one_empty_array():
    with pytest.raises(ValueError):
        accuracy_curve(train_scores_empty, cv_scores, param_grid)
        accuracy_curve(train_scores, cv_scores, param_grid_empty)

def test_all_empty_arrays():
    with pytest.raises(ValueError):
        accuracy_curve(train_scores_empty, cv_scores_empty, param_grid_empty)

def test_all_arrays_len1():
    with pytest.raises(ValueError):
        accuracy_curve(train_scores_1, cv_scores_1, param_grid_1)
