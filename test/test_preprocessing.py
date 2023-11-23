#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer, make_column_selector
import pytest
import sys
import os


# In[2]:


# Import the preprocessing function
current_directory = os.getcwd()
sys.path.append(os.path.join(current_directory, '..'))
from src.function_preprocessing import preprocessing


# In[3]:


# Test data
train_data = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [400, 500, 600],
        'c': [0.007, 0.008, 0.009],
        'Label': ['A', 'B', 'A']
    })

test_data = pd.DataFrame({
        'a': [4, 5],
        'b': [700, 800],
        'c': [0.001, 0.002],
        'Label': ['B', 'A']
    })
    
variables = pd.DataFrame({
        'name': ['a', 'b', 'c', 'Label'],
        'role': ['Feature', 'Feature', 'Feature', 'Target'],
        'type': ['Integer', 'Integer', 'Continuous', 'Categorical']
    })

output_train_path = "./test/scaled_wine_train.csv"
output_test_path = "./test/scaled_wine_test.csv"


# In[4]:


# Test for correct return type
def test_preprocessing_returns_df():
    output = preprocessing(train_data, test_data, output_train_path, output_test_path, variables)
    assert isinstance(output, pd.DataFrame), "preprocessing should return a Pandas DataFrame"


# In[5]:


# Test for expected columns
def test_preprocessing_expected_cols():
    output = preprocessing(train_data, test_data, output_train_path, output_test_path, variables)
    expected_cols = ['a', 'b', 'c']
    assert all(col in scaled_train_data.columns for col in expected_columns)


# In[6]:


# Test for if all values in the DataFrame are numeric
def test_preprocessing_all_values_numeric():
    output = preprocessing(train_data, test_data, output_train_path, output_test_path, variables)
    assert output.applymap(lambda x: isinstance(x, (int, float))).all().all(), "All values in the DataFrame should be numeric"


# In[7]:


# Test for output files are generated
def test_preprocessing_files_generated():
    output = preprocessing(train_data, test_data, output_train_path, output_test_path, variables)
    assert output_train_path.is_file()
    assert output_test_path.is_file()

