


def handle_missing_values(data, missing_threshold=50):
    """
    Handles missing values in a DataFrame by:
    - Dropping columns with missing values exceeding a specified threshold.
    - Imputing numerical columns with the median.
    - Imputing categorical columns with the mode.

    Parameters:
    - data (pd.DataFrame): Input DataFrame.
    - missing_threshold (float): Percentage threshold for dropping columns with missing values.

    Returns:
    - pd.DataFrame: Processed DataFrame with missing values handled.
    """
    # Check for missing values
    missing_values = data.isnull().sum()
    missing_percentage = (missing_values / len(data)) * 100
    
    # Drop columns with excessive missing values
    columns_to_drop = missing_percentage[missing_percentage > missing_threshold].index
    data = data.drop(columns=columns_to_drop)
    print(f"Dropped columns with more than {missing_threshold}% missing values: {columns_to_drop.tolist()}")

    # Impute missing values
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns

    # Impute numerical columns with the median
    for col in numerical_cols:
        if data[col].isnull().sum() > 0:
            data[col].fillna(data[col].median(), inplace=True)
            print(f"Imputed missing values in {col} with the median.")

    # Impute categorical columns with the mode
    for col in categorical_cols:
        if data[col].isnull().sum() > 0:
            data[col].fillna(data[col].mode()[0], inplace=True)
            print(f"Imputed missing values in {col} with the mode.")

    return data



