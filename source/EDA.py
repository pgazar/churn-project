
import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

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




def preprocess_data(data):
    """
    Preprocesses the dataset by:
    - Handling missing values.
    - Removing outliers in numerical columns using z-score.
    - Encoding categorical features with one-hot or label encoding.
    - Scaling numerical features using standard scaling.

    Parameters:
    - data (pd.DataFrame): The input dataset.

    Returns:
    - pd.DataFrame: The preprocessed dataset.
    """
    # Step 1: Handle Missing Values
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns

    # Impute missing values: median for numerical, mode for categorical
    for col in numerical_cols:
        if data[col].isnull().sum() > 0:
            data[col].fillna(data[col].median(), inplace=True)
            print(f"Imputed missing values in {col} with the median.")

    for col in categorical_cols:
        if data[col].isnull().sum() > 0:
            data[col].fillna(data[col].mode()[0], inplace=True)
            print(f"Imputed missing values in {col} with the mode.")

    # Step 2: Remove Outliers in Numerical Columns using Z-Score
    for col in numerical_cols:
        z_scores = zscore(data[col])
        data = data[(z_scores < 3) & (z_scores > -3)]  # Keep rows within ±3 standard deviations
        print(f"Outliers removed from {col} using z-score.")

    # Step 3: Encode Categorical Features
    encoder = OneHotEncoder(sparse=False, drop='first')
    encoded_df = pd.DataFrame(encoder.fit_transform(data[categorical_cols]),
                              columns=encoder.get_feature_names_out(categorical_cols),
                              index=data.index)
    data = data.drop(columns=categorical_cols)  # Drop original categorical columns
    data = pd.concat([data, encoded_df], axis=1)  # Add encoded columns
    print(f"Categorical features encoded using one-hot encoding.")

    # Step 4: Scale Numerical Features
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    print(f"Numerical features scaled using standard scaling.")

    return data



