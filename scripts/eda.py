import pandas as pd
import os
import click
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.eda import plot_density

"""
Usage: python scripts/eda.py --input_path="data/processed/scaled_train_data.csv" --output_path="data/processed"
"""
@click.command()
@click.option('--input_path', type=click.Path(exists=True), help='Path to the processed data for plotting')
@click.option('--output_figure_path', type=click.Path(), help="Path to save the EDA Density plot to")
@click.option('--plot_width', type=int, help="Width of each density plot", default=150)
@click.option('--plot_height', type=int, help="Height of each density plot", default=100)
def main(input_path, output_figure_path, plot_width, plot_height):
    TARGET = "class"
    N_COLS = 4

    #Fetch dataset
    if input_path is None:
        print("Input Data and Output figure path are required options")
        return 1
    
    data = pd.read_csv(input_path)

    plt = plot_density(data, data.columns, TARGET, N_COLS, plot_width, plot_height)

    plt.save(os.path.join(output_figure_path, "densities_plot_by_class.png"), scale_factor=2.0)

if __name__ in '__main__':
    main()
