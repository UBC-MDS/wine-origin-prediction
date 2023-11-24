import pandas as pd
from sklearn.model_selection import GridSearchCV

def optimize_hyperparameter(X_train, y_train, pipe, param_grid, njobs=-1, cv=5,method=GridSearchCV):
    """
    Optimize the hyperparameter(s) of logistic regression model.

    Uses exhaustive grid search to optimize the hyperparameters of a 
    linear model using 5 fold cross validation by default.

    Parameters:
    ----------
    X_train : pandas.DataFrame
        The X training data used to fit the model.
    y_train : pandas.DataFrame
        The y training data used to fit the model.
    pipe : scikit-learn Pipeline object
        The pipeline containing a list of the estimator objects.
    param_grid : numpy.dict
        A numpy dictionary containing the hyperparameter(s) to optimize and the range to
        optimize on.
    njobs : int
        (optional, default=-1) The number of jobs to run in parallel.
    cv : int
        (optional, default=5) Cross-validation splitting strategy.

    Returns:
    -------
    sklearn.pipeline.Pipeline
        Fitted pipeline with optimal hyperparameters
        
    Examples:
    --------
    >>> import pandas as pd
    >>> data = pd.read_csv('wine.csv')  # Replace 'wine.csv' with your dataset file
    >>> result = plot_density(data, ['alcohol', 'proline'], "class", ncols=1)
    >>> chart #Display chart

    Notes:
    Code adapted from https://github.com/ttimbers/breast_cancer_predictor_py
    
    """
    grid_search = GridSearchCV(
        pipe, param_grid, n_jobs=njobs, cv=cv, return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    return grid_search