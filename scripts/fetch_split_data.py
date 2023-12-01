import pandas as pd
import os
import click
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

"""
Usage: python scripts/fetch_split_data.py --output_path="data/processed" --uci_id=109
"""
@click.command()
@click.option('--output_path')
@click.option('--uci_id', type=int)
def main(output_path, uci_id):
    # fetch dataset 
    data = fetch_ucirepo(id=uci_id) 

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