import pandas as pd
import pytest
import sys
import os

# Import the plot_density function
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.optimization import optimize_hyperparameter

# Test data
param_grid = param_grid = {
    "logisticregression__C": np.array([0.01,0.1,1,10,100,1000])
}

pipe_lr = make_pipeline(
    StandardScaler(), LogisticRegression(max_iter=2000)
)

one_feature_one_class = pd.DataFrame({"feature_1" : [1, 2, 1, 3, 2], "target" : ["1", "1", "1", "1", "1"]})
two_feature_two_classes = pd.DataFrame({"feature_1" : [1, 2, 1, 3, 2], "feature_2" : [1.1, 2.2, 1.3, 3.2, 2.2], "target" : ["1", "2", "1", "2", "1"]})
