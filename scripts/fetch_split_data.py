import pandas as pd
import os
import click
from sklearn.model_selection import train_test_split

@click.command()
@click.option('--output_path')
def main(output_path):
    # fetch dataset 
    wine = fetch_ucirepo(id=109) 

    #Split into train/test with equal distribution of target classes
    wine_train, wine_test = train_test_split(
        wine.data.original, train_size=0.70, stratify=wine.data.original['class']
    )

    #Save split data
    wine_train.to_csv(os.path.join(output_path, f"{wine_train}"))
    wine_test.to_csv(os.path.join(output_path, f"{wine_test}"))