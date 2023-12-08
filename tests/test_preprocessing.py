import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer, make_column_selector
import pytest
import sys
import os

# Import the preprocessing function
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.function_preprocessing import preprocessing

# Test data scenario 1
train_data = pd.DataFrame({
        'a': [1, 5, 9],
        'b': [200, 600, 100],
        'c': [0.009, 0.008, 0.008],
        'Label': ['class1', 'class2', 'class1']
    })

test_data = pd.DataFrame({
        'a': [3, 7],
        'b': [400, 800],
        'c': [0.002, 0.006],
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

# Test data scenario 2
extra_train_data = pd.DataFrame({
        'x': [10, 20, 28],
        'y': ['high', 'medium', 'low'],
        'z': [0.02, 0.025, 0.03],
        'Target': ['alpha', 'beta', 'gamma']
    })

extra_test_data = pd.DataFrame({
        'x': [12, 25],
        'y': ['medium', 'high'],
        'z': [0.015, 0.018],
        'Target': ['beta', 'alpha']
    })

extra_variables = pd.DataFrame({
        'name': ['x', 'y', 'z', 'Target'],
        'role': ['Feature', 'Feature', 'Feature', 'Target'],
        'type': ['Integer', 'Categorical', 'Continuous', 'Categorical']
    })

extra_numerical_cols = extra_variables[extra_variables['type'].isin(['Integer', 'Continuous'])]['name'].tolist()
extra_categorical_cols = ['y', 'Target']
extra_all_cols = extra_numerical_cols + extra_categorical_cols

extra_output_train_path = "./scaled_data_train_extra.csv"
extra_output_test_path = "./scaled_data_test_extra.csv"

# Test for correct return type
def test_preprocessing_returns_df():
    preprocessor = preprocessing(extra_train_data, extra_test_data, extra_output_train_path, extra_output_test_path, extra_numerical_cols)
    output_train = pd.DataFrame(preprocessor.transform(extra_train_data), columns=extra_all_cols)
    output_test = pd.DataFrame(preprocessor.transform(extra_test_data), columns=extra_all_cols)
    assert isinstance(output_train, pd.DataFrame)
    assert isinstance(output_test, pd.DataFrame)

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

# Test for handling edge case with all categorical features
def test_preprocessing_all_categorical():
    all_categorical_data = train_data.copy()
    all_categorical_data[['a', 'b', 'c']] = all_categorical_data['a'].astype('str')
    all_categorical_cols = ['a', 'b', 'c']

    preprocessor = preprocessing(all_categorical_data, test_data, output_train_path, output_test_path, all_categorical_cols)
    output_train = pd.read_csv(output_train_path)
    output_test = pd.read_csv(output_test_path)

    assert isinstance(output_train, pd.DataFrame)
    assert isinstance(output_test, pd.DataFrame)
    assert 'a' in output_train.columns
    assert 'a' in output_test.columns
    assert 'b' in output_train.columns
    assert 'b' in output_test.columns
    assert 'c' in output_train.columns
    assert 'c' in output_test.columns
    assert 'Label' in output_train.columns
    assert 'Label' in output_test.columns

# Test for invalid input types
def test_preprocessing_input_types():
    with pytest.raises(ValueError):
        preprocessing(train_data, test_data, output_train_path, output_test_path, 'a')

# Test for nonexistent output paths
def test_preprocessing_output_paths_exist(tmpdir):
    with pytest.raises(OSError):
        nonexistent_output_train_path = str(tmpdir.join("nonexistent_folder/scaled_wine_train.csv"))
        preprocessing(train_data, test_data, nonexistent_output_train_path, output_test_path, numerical_cols)

# Test for missing cols
def test_preprocessing_missing_columns():
    with pytest.raises(ValueError):
        missing_train_data = train_data.drop(columns='c')
        missing_test_data = test_data.drop(columns='c')
        preprocessing(missing_train_data, missing_test_data, output_train_path, output_test_path, numerical_cols)
