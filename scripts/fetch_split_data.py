import pandas as pd
import os
import click
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo


@click.command()
@click.option('--output-raw-path', type=str, help="Path to directory where raw data will be saved")
@click.option('--output-processed-path', type=str, help="Path to directory where train and test splits will be saved")

def main(output_raw_path, output_processed_path):
    """
    Usage: python scripts/fetch_split_data.py --output-raw-path="data/raw" --output-processed-path="data/processed"
    """
    # fetch dataset 
    data = fetch_ucirepo(id=109) 
    variables = data.variables

    #Split into train/test with equal distribution of target classes
    train, test = train_test_split(
        data.data.original, train_size=0.70,
        stratify=data.data.original['class'],
        random_state=123
    )

    #Save split data
    if not os.path.exists(output_raw_path):
        os.makedirs(output_raw_path)

    data.data.original.to_csv(os.path.join(output_raw_path, "raw.csv"), index=False)
    variables.to_csv(os.path.join(output_raw_path, "variables.csv"), index=False)

    if not os.path.exists(output_processed_path):
        os.makedirs(output_processed_path)
    train.to_csv(os.path.join(output_processed_path, "train.csv"), index=False)
    test.to_csv(os.path.join(output_processed_path,  "test.csv"), index=False)
   

if __name__ in '__main__':
    main()