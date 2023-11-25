import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer, make_column_selector


def preprocessing(train_data, test_data, output_train_path, output_test_path, variables):
    """
    Preprocessing the training and test data

    Applies standard scaling to numerical features

    Parameters:
    ----------
        train_data : pandas.DataFrame
            The training data to be preprocessed
        test_data : pandas.DataFrame 
            The test data to be preprocessed
        output_train_path : str
            File path to save the preprocessed training data
        output_test_path : str 
            File path to save the preprocessed test data
        variables : pandas.DataFrame
            Dataframe containing variable information

    Returns:
    -------
        pandas.DataFrame
            The preprocessed training data

    Examples:
    --------
    >>> wine = fetch_ucirepo(id=109) # fetch dataset
    >>> wine_train, wine_test = train_test_split(wine.data.original, train_size=0.70, stratify=wine.data.original['class'])
    >>> wine_train.to_csv("./data/processed/wine_train.csv")
    >>> wine_test.to_csv("./data/processed/wine_test.csv")
    >>> preprocessing(wine_train, wine_test, "./data/processed/scaled_wine_train.csv", "./data/processed/scaled_wine_test.csv", wine.variables)

    Notes:
    -----
    # reading the saved preprocessed training data for eda distribution plotting
    >>> scaled_wine_train = pd.read_csv('./data/processed/scaled_wine_train.csv')
    
    """
    
    cols_to_scale = variables.query('role == "Feature" and type in ["Continuous", "Integer"]')["name"].to_list()
    
    preprocessor = make_column_transformer(
        (StandardScaler(), cols_to_scale),
        remainder='passthrough'
    )

    preprocessor.fit(train_data)
    preprocessor.set_output(transform='pandas')
    
    scaled_train_data = preprocessor.transform(train_data)
    scaled_test_data = preprocessor.transform(test_data)

    # Remove prefix added by column transformer
    scaled_train_data.columns = [col.split('__')[1] for col in scaled_train_data.columns]
    scaled_test_data.columns = [col.split('__')[1] for col in scaled_test_data.columns]

    scaled_train_data.to_csv(output_train_path, index=False)
    scaled_test_data.to_csv(output_test_path, index=False)

    return scaled_train_data, scaled_test_data
