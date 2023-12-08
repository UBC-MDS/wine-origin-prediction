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

numerical_cols = variables[variables['type'].isin(['Integer', 'Continuous'])]['name'].tolist()
categorical_cols = ['Label']
all_cols = numerical_cols + categorical_cols

output_train_path = "./scaled_wine_train.csv"
output_test_path = "./scaled_wine_test.csv"

# Test for correct return type
def test_preprocessing_returns_df():
    preprocessor = preprocessing(train_data, test_data, output_train_path, output_test_path, numerical_cols)
    output_train = pd.DataFrame(preprocessor.transform(train_data), columns=all_cols)
    output_test = pd.DataFrame(preprocessor.transform(test_data), columns=all_cols)
    assert isinstance(output_train, pd.DataFrame), "preprocessing should return a Pandas DataFrame"
    assert isinstance(output_test, pd.DataFrame), "preprocessing should return a Pandas DataFrame"

# Test for expected columns
def test_preprocessing_expected_cols():
    preprocessor = preprocessing(train_data, test_data, output_train_path, output_test_path, numerical_cols)
    output_train = pd.DataFrame(preprocessor.transform(train_data), columns=all_cols)
    output_test = pd.DataFrame(preprocessor.transform(test_data), columns=all_cols)
    expected_cols = ['a', 'b', 'c']
    assert all(col in output_train.columns for col in expected_cols)
    assert all(col in output_test.columns for col in expected_cols)

# Test for output files are generated
def test_preprocessing_files_generated():
    preprocessor = preprocessing(train_data, test_data, output_train_path, output_test_path, numerical_cols)
    output_train = pd.DataFrame(preprocessor.transform(train_data), columns=all_cols)
    output_test = pd.DataFrame(preprocessor.transform(test_data), columns=all_cols)
    assert os.path.exists(output_train_path)
    assert os.path.exists(output_test_path)