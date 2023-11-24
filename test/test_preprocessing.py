import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer, make_column_selector
import pytest
import sys
import os

# Import the preprocessing function
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.function_preprocessing import preprocessing

# Test data
train_data = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [400, 500, 600],
        'c': [0.007, 0.008, 0.009],
        'Label': ['class1', 'class2', 'class1']
    })

test_data = pd.DataFrame({
        'a': [4, 5],
        'b': [700, 800],
        'c': [0.001, 0.002],
        'Label': ['class2', 'class1']
    })
    
variables = pd.DataFrame({
        'name': ['a', 'b', 'c', 'Label'],
        'role': ['Feature', 'Feature', 'Feature', 'Target'],
        'type': ['Integer', 'Integer', 'Continuous', 'Categorical']
    })

output_train_path = "./scaled_wine_train.csv"
output_test_path = "./scaled_wine_test.csv"

# Test for correct return type
def test_preprocessing_returns_df():
    output = preprocessing(train_data, test_data, output_train_path, output_test_path, variables)
    assert isinstance(output, pd.DataFrame), "preprocessing should return a Pandas DataFrame"

# Test for expected columns
def test_preprocessing_expected_cols():
    output = preprocessing(train_data, test_data, output_train_path, output_test_path, variables)
    expected_cols = ['a', 'b', 'c']
    assert all(col in scaled_train_data.columns for col in expected_columns)

# Test for all values in the DataFrame are numeric
def test_preprocessing_all_values_numeric():
    output = preprocessing(train_data, test_data, output_train_path, output_test_path, variables)
    assert output.applymap(lambda x: isinstance(x, (int, float))).all().all(), "All values in the DataFrame should be numeric"

# Test for output files are generated
def test_preprocessing_files_generated():
    output = preprocessing(train_data, test_data, output_train_path, output_test_path, variables)
    assert output_train_path.is_file()
    assert output_test_path.is_file()
