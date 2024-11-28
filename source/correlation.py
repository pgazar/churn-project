import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import util.visual

def compute_correlations(data):
    """
    Computes and visualizes pairwise correlations for numerical features.

    Parameters:
        data (pd.DataFrame): The dataset containing numerical columns.

    Returns:
        correlation_matrix (pd.DataFrame): Pairwise correlation matrix.
    """
    # Select numerical columns
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns

    # Compute the correlation matrix
    correlation_matrix = data[numerical_columns].corr()

    # Visualize the correlation matrix
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
    plt.title("Pairwise Correlation Matrix")
    util.visual.save_plot(fig,'correlation_matrix.png' )
    plt.show()

    return correlation_matrix

def identify_high_correlations(correlation_matrix, threshold=0.8):
    """
    Identifies pairs of features with strong correlations.

    Parameters:
        correlation_matrix (pd.DataFrame): Pairwise correlation matrix.
        threshold (float): Threshold for identifying strong correlations.

    Returns:
        high_corr_pairs (list): List of tuples with strongly correlated feature pairs.
    """
    high_corr_pairs = []

    for i in correlation_matrix.columns:
        for j in correlation_matrix.columns:
            if i != j and abs(correlation_matrix[i][j]) > threshold:
                high_corr_pairs.append((i, j, correlation_matrix[i][j]))

    # Deduplicate pairs (e.g., (A, B) and (B, A))
    high_corr_pairs = list(set([tuple(sorted(pair[:2])) + (pair[2],) for pair in high_corr_pairs]))
    
    return high_corr_pairs








