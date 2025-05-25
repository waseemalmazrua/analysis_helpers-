
# analysis_helpers.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#  1. Histogram Plot Function
def plot_histogram(df, column):
    """ Plots a histogram with KDE overlay for a numeric column. """
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x=column, kde=True, color='skyblue')
    plt.title(f'Distribution of {column}')
    plt.show()

#  2. Scatter Plot Function
def plot_scatter(df, x, y):
    """ Creates a scatter plot to analyze the relationship between two numeric variables. """
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=x, y=y)
    plt.title(f'{y} vs {x}')
    plt.show()

#  3. Line Plot Function
def plot_line(df, x, y):
    """ Creates a line plot to show trends over time or ordered data. """
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x=x, y=y)
    plt.title(f'Trend of {y} over {x}')
    plt.xticks(rotation=45)
    plt.show()

#  4. Missing Values Summary
def missing_data_summary(df):
    """ Prints the percentage of missing values for each column in the DataFrame. """
    missing = df.isnull().mean() * 100
    print("Missing Values (%):")
    print(missing[missing > 0].sort_values(ascending=False))

#  5. Top N by Group Column
def top_n_by_column(df, group_col, target_col, n=10):
    """ Returns top N values grouped by one column using the sum of another column. """
    return df.groupby(group_col)[target_col].sum().sort_values(ascending=False).head(n)

#  6. Summary Statistics
def summary_stats(df):
    """ Returns descriptive statistics for all numeric columns. """
    return df.describe().T

#  7. Bar Plot Function
def plot_bar(df, x, y, title=''):
    """ Creates a bar plot for comparison between categories. """
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x=x, y=y)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()

#  8. Convert Column to Datetime
def convert_to_datetime(df, col):
    """ Converts a column to datetime format. """
    df[col] = pd.to_datetime(df[col])

#  9. Drop Columns with High Missing Values
def drop_high_missing_columns(df, threshold=0.7):
    """ Drops columns with a missing ratio higher than the threshold. """
    missing_ratio = df.isnull().mean()
    to_drop = missing_ratio[missing_ratio > threshold].index
    return df.drop(columns=to_drop)

#  10. Value Counts Plot
def value_counts_plot(df, column):
    """ Plots value counts of a categorical column as a bar chart. """
    plt.figure(figsize=(8, 4))
    df[column].value_counts().plot(kind='bar')
    plt.title(f'Distribution of {column}')
    plt.show()
