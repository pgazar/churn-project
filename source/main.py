
import pandas as pd
import sys
import os


from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import correlation
import EDA

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import util.visual



# Load the dataset
file_path = './data/online_retail_customer_churn.csv'
data = pd.read_csv(file_path)
print(data.columns)

# List of categorical and numerical columns
categorical_columns = ['Target_Churn', 'Promotion_Response', 'Email_Opt_In']
numerical_columns = ['Years_as_Customer', 'Annual_Income', 'Satisfaction_Score','Num_of_Purchases','Average_Transaction_Amount','Num_of_Returns','Num_of_Support_Contacts']

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
strong_correlations = correlation.identify_high_correlations(correlation_matrix, threshold=0.9)
if strong_correlations:
    print("Strong correlations found:", strong_correlations)
else:
    print("No strong correlations found.")

# Optional: Drop redundant features
for pair in strong_correlations:
    feature_to_drop = pair[0]  # Choose one feature from the pair
    print(f"Dropping feature: {feature_to_drop}")
    data = data.drop(columns=[feature_to_drop])


data_cleaned = EDA.handle_missing_values(data)



preprocessed_data = EDA.preprocess_data(data_cleaned)
print(preprocessed_data.head())
preprocessed_data.to_csv("preprocessed_data.csv", index=False)


# Load the preprocessed data
data = pd.read_csv('./data/preprocessed_data.csv')  # Update the file path if needed

# Separate features and target
# Assuming the target column is named 'Target_Churn'
X = data.drop(columns=['Target_Churn'])
y = data['Target_Churn']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the XGBoost classifier
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

# Fit the model to the training data
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Display the results
print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)



