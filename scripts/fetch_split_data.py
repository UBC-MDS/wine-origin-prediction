import pandas as pd
import os
import click
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo


@click.command()
@click.option('--output-path', type=str, help="Path to directory where train and test splits will be saved")

def main(output_path):
    """
    Usage: python scripts/fetch_split_data.py --output-path="data/processed" 
    """
    # fetch dataset 
    data = fetch_ucirepo(id=109) 

    #Split into train/test with equal distribution of target classes
    train, test = train_test_split(
        data.data.original, train_size=0.70, 
        stratify=data.data.original['class']
    )

    #Save split data
    train.to_csv(os.path.join(output_path, "train.csv"))
    test.to_csv(os.path.join(output_path,  "test.csv"))

if __name__ in '__main__':
    main()