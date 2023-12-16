import pandas as pd
import numpy as np
from IPython.display import Image
import seaborn as sns
import matplotlib.pyplot as plt


def plot_hist(df: pd.DataFrame, column: str, color: str) -> None:
    # plt.figure(figsize=(15, 10))
    # fig, ax = plt.subplots(1, figsize=(12, 7))
    sns.displot(data=df, x=column, color=color, kde=True, height=7, aspect=2)
    plt.title(f'Distribution of {column}', size=20, fontweight='bold')
    plt.show()


def plot_count(df: pd.DataFrame, column: str) -> None:
    plt.figure(figsize=(12, 7))
    sns.countplot(data=df, x=column)
    plt.title(f'Distribution of {column}', size=20, fontweight='bold')
    plt.show()


def plot_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(12, 7))
    sns.barplot(data=df, x=x_col, y=y_col)
    plt.title(title, size=20)
    plt.xticks(rotation=75, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.show()


def plot_heatmap(df: pd.DataFrame, title: str, cbar=False) -> None:
    plt.figure(figsize=(12, 7))
    sns.heatmap(df, annot=True, cmap='viridis', vmin=0,
                vmax=1, fmt='.2f', linewidths=.7, cbar=cbar)
    plt.title(title, size=18, fontweight='bold')
    plt.show()


def plot_box(df: pd.DataFrame, x_col: str, title: str) -> None:
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df, x=x_col)
    plt.title(title, size=20)
    plt.xticks(rotation=75, fontsize=14)
    plt.show()


def plot_box_multi(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> None:
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df, x=x_col, y=y_col)
    plt.title(title, size=20)
    plt.xticks(rotation=75, fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str, hue: str, style: str) -> None:
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, style=style)

    # Move the legend to the top right without covering data
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))

    plt.title(title, size=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()


def univariate_analysis(df):
    """
    Conducts graphical univariate analysis for each variable in the DataFrame.

    """
    # Set the style for the plots
    sns.set(style="whitegrid")

    # Get column names
    variables = df.columns

    for variable in variables:
        # Create a figure with a subplot
        plt.figure(figsize=(10, 6))

        if 'MB' in variable:
            # For data volume columns, use a histogram
            sns.histplot(df[variable], bins=30, kde=True)
            plt.title(f'Distribution of {variable}')
            plt.xlabel(variable)
            plt.ylabel('Frequency')
        else:
            # For other columns, use a boxplot
            sns.boxplot(x=df[variable])
            plt.title(f'Boxplot of {variable}')
            plt.xlabel(variable)

        plt.show()


def bivariate_analysis(df, app_columns, total_dl_col, total_ul_col):
    """
    Conducts bivariate analysis between each application and total DL+UL data.

    Parameters:
    - df: DataFrame
        The input DataFrame for analysis.
    - app_columns: list
        List of application columns to be analyzed.
    - total_dl_col: str
        Column name for total download (DL) data.
    - total_ul_col: str
        Column name for total upload (UL) data.
    """
    # Set the style for the plots
    sns.set(style="whitegrid")

    # Create subplots
    fig, axs = plt.subplots(nrows=len(app_columns),
                            ncols=2, figsize=(18, len(app_columns) * 8))

    # Iterate through each application column
    for i, app_column in enumerate(app_columns):
        # Scatter plot for App DL vs. Total DL
        axs[i, 0].scatter(df[app_column + ' DL (MB)'], df[total_dl_col],
                          c=df[total_dl_col], cmap='viridis', label=f'{app_column} DL')
        axs[i, 0].set_title(f'{app_column} DL vs. {total_dl_col}')
        axs[i, 0].set_xlabel(f'{app_column} DL')
        axs[i, 0].set_ylabel(total_dl_col)
        axs[i, 0].set_xlim([0, df[app_column + ' DL (MB)'].max()])
        axs[i, 0].set_ylim([0, df[total_dl_col].max()])

        # Scatter plot for App UL vs. Total UL
        axs[i, 1].scatter(df[app_column + ' UL (MB)'], df[total_ul_col],
                          c=df[total_ul_col], cmap='viridis', label=f'{app_column} UL')
        axs[i, 1].set_title(f'{app_column} UL vs. {total_ul_col}')
        axs[i, 1].set_xlabel(f'{app_column} UL')
        axs[i, 1].set_ylabel(total_ul_col)
        axs[i, 1].set_xlim([0, df[app_column + ' UL (MB)'].max()])
        axs[i, 1].set_ylim([0, df[total_ul_col].max()])

        # Add legend
        axs[i, 0].legend(loc='upper right')
        axs[i, 1].legend(loc='upper right')

    plt.tight_layout()
    plt.show()
