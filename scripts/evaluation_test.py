import pandas as pd
import os
import sys
import click
import pickle
from sklearn.metrics import f1_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

@click.command()
@click.option('--input-test-path', type=str, help="Path to directory where test.csv is located")  
@click.option('--pipeline-from', type=str, help="Path to directory where pickled model lives") 
@click.option('--target-col', type=str, help="Name of target column") 
@click.option('--results-to', type=str, help="Path to directory where test scores will be saved") 

def main(input_test_path, pipeline_from, target_col, results_to):
    """
    Usage: python --input-test-path=data/processed/test.csv --pipeline-from
    --target-col=class --results-to=results
    """

    # Derive X_test and y_test
    test = pd.read_csv(input_test_path)
    X_test = test.drop(columns=[target_col])
    y_test = test[target_col]

    # Load pickled model
    wine_model = pickle.load(open(pipeline_from, "rb"))

    accuracy_score = wine_model.score(X_test,y_test)

    wine_predictions = test.assign(prediction=wine_model.predict(test))

    f1 = f1_score(wine_predictions['class'], wine_predictions['prediction'],average='weighted')

    results = pd.DataFrame({'accuracy': [accuracy_score], 'F1 score': [f1]})
    results.to_csv(os.path.join(results_to, "test_results.csv"), index=False, float_format='%.3f')
    
if __name__ == '__main__':
    main()
    