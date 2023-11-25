from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


def optimize_hyperparameter(X_train, y_train, pipe, 
                            param_grid, njobs=-1, cv=5, method=GridSearchCV):
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
        A numpy dictionary containing the hyperparameter(s) 
        to optimize and the range to optimize on.
    njobs : int
        (optional, default=-1) The number of jobs to run in parallel.
    cv : int
        (optional, default=5) Cross-validation splitting strategy.

    Returns:
    -------
    sklearn.model_selection._search.GridSearchCV
        Fitted GridSearchCV object
        
    Examples:
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.linear_model import LogisticRegression
    >>> param_grid = param_grid = {"logisticregression__C": 
    >>>                 np.array([0.01,0.1,1,10,100,1000])}
    >>> pipe_lr = make_pipeline(StandardScaler(), 
    >>>                 LogisticRegression(max_iter=2000))
    >>> X_train = pd.DataFrame({'a': [1, 2, 3],
    >>>                 'b': [400, 500, 600],
    >>>                 'c': [0.007, 0.008, 0.009]})
    >>> y_train = pd.DataFrame({'Label': ['class1', 'class2', 'class1']})
    >>> X_test = pd.DataFrame({'a': [4, 5],
    >>>                 'b': [700, 800],
    >>>                 'c': [0.001, 0.002],})
    >>> y_test = pd.DataFrame({'Label': ['class2', 'class1']})
    >>> results = optimize_hyperparameter(X_train, y_train, pipe, param_grid)
    
    """
    if (not isinstance(pipe, Pipeline)):
        raise TypeError("pipe must be type sklearn.pipeline.Pipeline")
    
    if (not isinstance(param_grid, dict)):
        raise TypeError("param_grid must be type dictionary")

    if (not param_grid):
        raise ValueError("param_grid must contain hyperparameters to optimize")

    grid_search = GridSearchCV(
        pipe, param_grid, n_jobs=njobs, cv=cv, return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    return grid_search