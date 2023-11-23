import pandas as pd
import altair as alt

def plot_density(data, cols, target, ncols=-1, width=150, height=100):
    """
    Plot a density plot for all columns in the dataset

    Creates an Altair faceted plot with, with each plot showing the distribution of values in each target class,
    for each of the variables in "cols"
    The plotting assumes the range of values each variable takes on is the same (has been scaled using StandardScaler or equivalent)

    Parameters:
    ----------
    data : pandas.DataFrame
        The input DataFrame containing the data to analyze.
    cols : list(str)
        The list of columns (features) to plot in the Facet chart
    target : str
        The name of the column in the DataFrame containing target labels
    ncols : int
        (optional, default=-1) The number of columns in the faceted plot
    width : int
        (optional, default=150) The width of each of the faceted plots
    height : int]
        (optional, default=100) The height of each of the faceted plots

    Returns:
    -------
    altair.vegalite.v5.api.FacetChart
        A FacetChart plotting the distribution density for each of the features in cols in separate graphs
        - x: The variable value (should be standardized for each variable before passing to the function)
        - y: The distribution density
        
    Examples:
    --------
    >>> import pandas as pd
    >>> data = pd.read_csv('wine.csv')  # Replace 'wine.csv' with your dataset file
    >>> result = plot_density(data, ['alcohol', 'proline'], "class", ncols=1)
    >>> chart #Display chart

    Notes:
    Code adapted from https://github.com/ttimbers/breast_cancer_predictor_py
    
    """
    if len(data) == 0:
        raise ValueError(f"Empty Dataframe, cannot plot data")

    unknown_cols = [col for col in cols if col not in data.columns]
    if len(unknown_cols) > 0:
        raise KeyError(f"Unknown columns: {unknown_cols}. Not found in dataframe columns")

    if target not in data.columns:
        raise KeyError(f"Target columns {target} not found in dataframe columns")

    # melt for plotting via facets 
    data_melted = data.melt(
        id_vars=[target],
        var_name='predictor',
        value_name='value'
    ).query("predictor in @cols")

    
    #Plot the distribution of each feature for each class
    chart = alt.Chart(data_melted, width=width, height=height).transform_density(
        'value',
        groupby=[target, 'predictor']
    ).mark_area(opacity=0.7).encode(
        x=alt.X("value:Q"),
        y=alt.Y('density:Q').stack(False),
        color=f'{target}:N'
    ).facet(
        'predictor:N',
        columns=ncols
    ).resolve_scale(
        y='independent'
    )
    return chart
    