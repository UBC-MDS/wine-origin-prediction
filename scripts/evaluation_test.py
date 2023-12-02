import pandas as pd
import os
import click
import pickle


@click.command()
@click.option('--input-test-path', help="Path to directory where test.csv is located")  
@click.option('--pipeline-from', help="Path to directory where pickled model lives") 
@click.option('--target-col', help="Name of target column") 
@click.option('--results-to', help="Path to directory where test scores will be saved") 

def evaluate(input_test_path, pipeline_from, target_col, results_to):
    """

    """

    # Derive X_test and y_test
    test = pd.read_csv(input_test_path)
    X_test = test.drop(columns=[target_col])
    y_test = test[target_col]

    # Load pickled model

    with open(pipeline_from, 'rb') as f:
        wine_model = pickle.load(f)


    accuracy_score = wine_model.score(X_test,y_test)
    res = f"Final LR model test accuracy: {round(accuracy_score*100,2)}%"
    
    with open(os.path.join(results_to, "test_results.txt"), "w") as res_f:
        res_f.write(res)

    print(res_f)
    