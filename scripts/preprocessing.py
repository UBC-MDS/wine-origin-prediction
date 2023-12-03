# preprocessing.py
# author: Yimeng Xia
# date: 2023-12-01

import click
import pickle
import sys
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer

# Import the preprocessing function
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.function_preprocessing import preprocessing

@click.command()
@click.option('--train-data', type=click.Path(exists=True), help='Path to the training data CSV file')
@click.option('--test-data', type=click.Path(exists=True), help='Path to the test data CSV file')
@click.option('--variable-data', type=click.Path(exists=True), help='Path to the CSV file with information on the input variables')
@click.option('--output-file-path', type=click.Path(), nargs=2, help='Paths to save the preprocessed data CSV files for training and test data')
@click.option('--output-metadata-path', type=click.Path(), help='Paths to save CSV with metadata about the preprocessing done')
@click.option('--output-preprocessor', type=click.Path(), help='Path to save the preprocessor model')

def main(train_data, test_data, variable_data, output_file_path, output_metadata_path, output_preprocessor):
    """
    Main function to run the preprocessing script.

    Parameters:
    ----------
        train_data : str
            Path to the training data CSV file.
        test_data : str
            Path to the test data CSV file.
        output_file_path : str
            Path to save the preprocessed data CSV files for training and test data.
        output_preprocessor : str
            Path to save the preprocessor model.
    """

    # Load data
    train_data = pd.read_csv(train_data)
    test_data = pd.read_csv(test_data)
    variable_data = pd.read_csv(variable_data)

    cols_to_scale = variable_data.query('role == "Feature" and type in ["Continuous", "Integer"]')["name"].to_list()

    preprocessor = preprocessing(train_data, test_data, output_file_path[0], output_file_path[1], cols_to_scale)

    # Save metadata about the preprocessing done
    preprocessor_df = pd.DataFrame.from_dict({
        'columns': cols_to_scale,
        'mean': preprocessor.named_transformers_['standardscaler'].mean_,
        'scale': preprocessor.named_transformers_['standardscaler'].scale_
    })
    preprocessor_df.to_csv(output_metadata_path + '.csv', index=False)

    # Save preprocessor model using pickle
    with open(output_preprocessor + '.pickle', 'wb') as preprocessor_file:
        pickle.dump(preprocessor, preprocessor_file)

if __name__ == '__main__':
    main()