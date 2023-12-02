# preprocessing.py
# author: Yimeng Xia
# date: 2023-12-01

import click
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer

@click.command()
@click.option('--train-data', type=click.Path(exists=True), help='Path to the training data CSV file')
@click.option('--test-data', type=click.Path(exists=True), help='Path to the test data CSV file')
@click.option('--output-file-path', type=click.Path(), nargs=2, help='Paths to save the preprocessed data CSV files for training and test data')
@click.option('--output-preprocessor', type=click.Path(), help='Path to save the preprocessor model')

def preprocessing(train_data, test_data, output_file_path, output_preprocessor):
    """
    Preprocessing the training and test data

    Applies standard scaling to numerical features

    Parameters:
    ----------
        train_data : str
            Path to the training data CSV file
        test_data : str
            Path to the test data CSV file
        output_file_path : List[str]
            Paths to save the preprocessed data CSV files for training and test data
        output_preprocessor : str
            Path to save the preprocessor model

    Returns:
    -------
        None
    """
    # Load data
    train_data = pd.read_csv(train_data)
    test_data = pd.read_csv(test_data)

    cols_to_scale = train_data.select_dtypes(include=['float64', 'int64']).columns.to_list()

    preprocessor = make_column_transformer(
        (StandardScaler(), cols_to_scale),
        remainder='passthrough'
    )

    preprocessor.fit(train_data)
    preprocessor.set_output(transform='pandas')

    scaled_train_data = preprocessor.transform(train_data)
    scaled_test_data = preprocessor.transform(test_data)

    # Remove prefix added by column transformer
    scaled_train_data.columns = [col.split('__')[1] for col in scaled_train_data.columns]
    scaled_test_data.columns = [col.split('__')[1] for col in scaled_test_data.columns]

    # Save preprocessed data to separate files for training and test data
    scaled_train_data.to_csv(output_file_path[0], index=False)
    scaled_test_data.to_csv(output_file_path[1], index=False)

    # Save preprocessor model
    preprocessor_df = pd.DataFrame.from_dict({
        'columns': train_data.columns,
        'mean': preprocessor.named_transformers_['standardscaler'].mean_,
        'scale': preprocessor.named_transformers_['standardscaler'].scale_
    })
    preprocessor_df.to_csv(output_preprocessor, index=False)

    # Save preprocessor model using pickle
    with open(output_preprocessor, 'wb') as preprocessor_file:
        pickle.dump(preprocessor, preprocessor_file)

if __name__ == '__main__':
    preprocessing()