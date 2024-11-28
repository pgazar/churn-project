
import pandas as pd

import sys
import os

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import correlation

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import util.visual



# Load the dataset
file_path = './data/online_retail_customer_churn.csv'
data = pd.read_csv(file_path)
print(data.columns)

# List of categorical and numerical columns
categorical_columns = ['Target_Churn', 'Promotion_Response', 'Email_Opt_In']
numerical_columns = ['Years_as_Customer', 'Annual_Income', 'Satisfaction_Score']

# Generate count plots for categorical features
for col in categorical_columns:
    util.visual.count_plot(
        data=data,
        column=col,
        title=f'Distribution of {col.replace("_", " ").title()}',
        save_name=f'{col}_distribution.png'
    )

# Generate distribution plots for numerical features
for col in numerical_columns:
     util.visual.distribution_plot(
        data=data,
        column=col,
        title=f'Distribution of {col.replace("_", " ").title()}',
        xlabel=col.replace("_", " ").title(),
        save_name=f'{col}_distribution.png'
    )

# Generate boxplots to compare numerical features with churn
for col in numerical_columns:
    util.visual.boxplot_with_churn(
        data=data,
        feature_col=col,
        churn_col='Target_Churn',
        save_name=f'{col}_vs_churn_boxplot.png'
    )




# Compute correlations
correlation_matrix = correlation.compute_correlations(data)

# Identify strong correlations
strong_correlations = correlation.identify_high_correlations(correlation_matrix, threshold=0.8)
if strong_correlations:
    print("Strong correlations found:", strong_correlations)
else:
    print("No strong correlations found.")

# Optional: Drop redundant features
for pair in strong_correlations:
    feature_to_drop = pair[0]  # Choose one feature from the pair
    print(f"Dropping feature: {feature_to_drop}")
    data = data.drop(columns=[feature_to_drop])







