import pandas as pd
import altair as alt
import pytest
import sys
import os

# Import the plot_density function
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.eda import plot_density

# Test data
empty_data = pd.DataFrame([])
one_feature_no_observations = pd.DataFrame({'feature_1': [], "target": []})
one_feature_one_class = pd.DataFrame({"feature_1" : [1, 2, 1, 3, 2], "target" : ["1", "1", "1", "1", "1"]})
one_feature_two_classes = pd.DataFrame({"feature_1" : [1, 2, 1, 3, 2], "target" : ["1", "1", "1", "2", "2"]})
two_feature_three_classes = pd.DataFrame({"feature_1" : [1, 2, 1, 3, 2], "feature_2" : [1.1, 2.2, 1.3, 3.2, 2.2], "target" : ["1", "1", "1", "2", "3"]})
five_features_three_classes = pd.DataFrame({"feature_1" : [1, 2, 1, 3, 2], \
                                            "feature_2" : [1.1, 2.2, 1.3, 3.2, 2.2], \
                                            "feature_3" : [1.1, 2.2, 1.3, 3.2, 2.2], \
                                            "feature_4" : [1.1, 2.2, 1.3, 3.2, 2.2], \
                                            "feature_5" : [1.1, 2.2, 1.3, 3.2, 2.2], \
                                            "target" : ["1", "1", "1", "2", "2"]})

# Test for correct return type
def test_plot_density_returns_facet_chart():
    chart = plot_density(one_feature_two_classes, ["feature_1"], "target")
    assert isinstance(chart, alt.vegalite.v5.api.FacetChart), "plot_density` should return a Facet Chart"


def test_plot_density_chart_has_correct_format():
    feats = ["feature_1", "feature_2"]
    target = "target"
    ncols, height, width = 1, 50, 50
    chart = plot_density(two_feature_three_classes, feats, target, ncols=ncols, height=height, width=width)
    chart_dict = chart.to_dict()
    assert chart_dict["facet"]["type"] == "nominal"
    assert chart_dict["columns"] == ncols
    assert chart_dict["spec"]["height"] == height
    assert chart_dict["spec"]["width"] == width
    assert chart_dict["spec"]["encoding"]["x"]["type"] == "quantitative"
    assert chart_dict["spec"]["encoding"]["y"]["field"] == "density"
    assert chart_dict["spec"]["encoding"]["y"]["type"] == "quantitative"
    assert chart_dict["spec"]["transform"][0]['density'] == 'value' 
    assert chart_dict["spec"]["transform"][0]['groupby'] == [target, "predictor"]

def test_plot_density_gets_subset_of_features():
    feats = ["feature_1", "feature_4"]
    target = "target"
    chart = plot_density(five_features_three_classes, feats, target)    
    assert set(chart['data']['predictor'].unique().tolist()) == set(feats)

def test_plot_density_with_wrong_col_names():
    feats = ["feature_1", "wrong_feature_name"]
    target = "target"
    with pytest.raises(KeyError):
        chart = plot_density(two_feature_three_classes, feats, target)

def test_plot_density_with_wrong_target_name():
    feats = ["feature_1", "feature_2"]
    target = "wrong_target"
    with pytest.raises(KeyError):
        chart = plot_density(two_feature_three_classes, feats, target)

def test_plot_density_with_empty_df():
    feats = ["feature_1", "feature_2"]
    target = "target"
    with pytest.raises(ValueError):
        chart = plot_density(one_feature_no_observations, feats, target)
    with pytest.raises(ValueError):
        chart = plot_density(empty_data, feats, target)