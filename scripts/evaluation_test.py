import pandas as pd
import os
import click

"""

"""
@click.command()
@click.option('--input_path') # path to test.csv 
@click.option('--pipe') 
@click.option('--output_path') # output path to save results
def evaluate(input_path, pipe, output_path):
    test = pd.read_csv(input_path)
    X_test = test.drop(columns=['class'])
    y_test = test['class']

    test_score = pipe.score(X_test,y_test)
    res = f"Final LR model test accuracy: {round(test_score*100,2)}%"
    
    res_f = open(os.path.join(output_path, "test_results.txt"), "w")
    res_f.write(res)
    res_f.close()

    print(res_f)
    