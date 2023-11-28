import pandas as pd
import os
import click
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

@click.command()
@click.option('--output_path')
@click.option('--ucirepo_id', type=int)
def main(output_path, id):
    # fetch dataset 
    wine = fetch_ucirepo(id=id) 

    #Split into train/test with equal distribution of target classes
    train, test = train_test_split(
        wine.data.original, train_size=0.70, stratify=wine.data.original['class']
    )

    #Save split data
    train.to_csv(os.path.join(output_path, "train.csv"))
    test.to_csv(os.path.join(output_path,  "test.csv"))

if __name__ in '__main__':
    main()