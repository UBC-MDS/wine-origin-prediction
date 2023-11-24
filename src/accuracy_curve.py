# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
def accuracy_curve(train_scores, cv_scores, param_grid):
    """
    Generates a visualization of mean accuracy scores for the training and validation sets. 
    These are obtained during the optimization of the hyperparameter C in the Logistic Regression model.
    
    
    Parameters:
    - train_scores (list or array or pd.Series): Accuracy scores for the training set across the C values specified in the `param_grid`. 
    - cv_scores (list or array or pd.Series): Accuracy scores for the validation set across the C values specified in the `param_grid`.
    - param_grid (dict): Parameter grid containing the values of C used in the analysis.

    Returns:
    plot: matpotlib.pyplot.figure
        semilogx plot with C values on the logarithmic x-axis and accuracy
        scores on the linear y-axis. 

    Example:
    ```
    param_grid = {"logisticregression__C": [0.01,0.1,1,10,100,1000]}
    train_scores = C_search.cv_results_["mean_train_score"]
    cv_scores = C_search.cv_results_["mean_test_score"]
    accuracy_curve(train_scores, cv_scores, param_grid)
    ```
    Notes:
    The train and cv scores are obtained from the GridSearchCV performed
    for the LR model across the values specified in the `param_grid`. 
    In the provided usage example, these scores are accessed 
    using `C_search.cv_results_`.
    
    """
    # Plotting code
    fig, ax = plt.subplots()
    
    ax.semilogx(param_grid, train_scores, label="train")
    ax.semilogx(param_grid, cv_scores, label="valid")
    ax.legend()

    ax.set_xlabel("C")
    ax.set_ylabel("Accuracy")

    return fig, ax
    


