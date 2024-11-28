import seaborn as sns
import matplotlib.pyplot as plt
import os


def save_plot(fig, save_name):
    """
    Save the plot to a specified location.
    
    Parameters:
        fig (matplotlib.figure.Figure): The matplotlib figure to save.
        save_name (str): Path where the figure should be saved.
    """
    save_dir = './util/plots'
    os.makedirs(save_dir, exist_ok=True)  # Ensure the folder exists
    save_path = os.path.join(save_dir, save_name)
    fig.savefig(save_path, bbox_inches='tight')
    print(f"Plot successfully saved at: {save_path}")


def count_plot(data, column='Target_Churn', labels=None, title='Churn Distribution', save_name='count_churn.png'):
    """
    Creates a count plot for a specified column and saves it.
    
    Parameters:
        data (pd.DataFrame): The dataset containing the column.
        column (str): The column to plot (default is 'Target_Churn').
        labels (list): Custom labels for the x-axis (optional).
        title (str): Title of the plot.
        save_name (str): Name of the file to save the plot as (default is 'count_churn.png').
    """
    fig = plt.figure(figsize=(8, 6))
    sns.countplot(x=data[column], palette=['skyblue', 'salmon'])
    if labels:
        plt.xticks(ticks=[0, 1], labels=labels)
    plt.title(title)
    plt.xlabel('Category')
    plt.ylabel('Count')
    save_plot(fig, save_name)
  


def distribution_plot(data, column, bins=20, kde=True, color='blue', title='Distribution Plot', xlabel=None, save_name='distribution.png'):
    """
    Creates a distribution plot for a numerical column and saves it.
    
    Parameters:
        data (pd.DataFrame): The dataset containing the column.
        column (str): The numerical column to plot.
        bins (int): Number of bins for the histogram.
        kde (bool): Whether to plot the kernel density estimate.
        color (str): Color of the histogram.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis (optional).
        save_name (str): Name of the file to save the plot as.
    """
    fig = plt.figure(figsize=(8, 5))
    sns.histplot(data[column], bins=bins, kde=kde, color=color)
    plt.title(title)
    plt.xlabel(xlabel if xlabel else column.replace('_', ' ').title())
    plt.ylabel('Frequency')
    save_plot(fig, save_name)
    


def boxplot_with_churn(data, feature_col, churn_col='Target_Churn', save_name='boxplot_churn.png'):
    """
    Creates a boxplot to compare a numerical feature with churn status and saves it.
    
    Parameters:
        data (pd.DataFrame): The dataset containing the feature and churn columns.
        feature_col (str): The numerical column to compare.
        churn_col (str): The churn status column.
        save_name (str): Name of the file to save the plot as.
    """
    fig = plt.figure(figsize=(8, 5))
    sns.boxplot(x=data[churn_col], y=data[feature_col], palette='Set2')
    plt.title(f'{feature_col} by Churn Status')
    plt.xlabel('Churn (True/False)')
    plt.ylabel(feature_col.replace('_', ' ').title())
    save_plot(fig, save_name)
   


def compare_tenure_with_churn(data, tenure_col='Years_as_Customer', churn_col='Target_Churn'):
    """
    Visualizes the relationship between tenure and churn status using boxplot and histograms.
    
    Parameters:
        data (pd.DataFrame): The dataset containing the tenure and churn columns.
        tenure_col (str): Column name for customer tenure.
        churn_col (str): Column name for churn status.
    """
    # Boxplot
    boxplot_with_churn(data, feature_col=tenure_col, churn_col=churn_col, save_name='tenure_vs_churn_boxplot.png')

    # Histogram: Tenure distribution for churners and non-churners
    fig = plt.figure(figsize=(8, 5))
    sns.histplot(data[data[churn_col] == True][tenure_col], bins=20, kde=True, color='red', label='Churners')
    sns.histplot(data[data[churn_col] == False][tenure_col], bins=20, kde=True, color='green', label='Non-Churners')
    plt.title('Tenure Distribution for Churners vs. Non-Churners')
    plt.xlabel('Years as Customer')
    plt.ylabel('Frequency')
    plt.legend()
    save_plot(fig, 'tenure_churn_histogram.png')
   
