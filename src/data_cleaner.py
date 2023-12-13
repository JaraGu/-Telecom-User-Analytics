import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# how many missing values exist or better still what is the % of missing values in the dataset?


def percent_missing(df):

    # Calculate total number of cells in dataframe
    totalCells = np.product(df.shape)

    # Count number of missing values per column
    missingCount = df.isnull().sum()

    # Calculate total number of missing values
    totalMissing = missingCount.sum()

    # Calculate percentage of missing values
    print("The dataset contains", round(
        ((totalMissing/totalCells) * 100), 2), "%", "missing values.")


# Function to calculate missing values by column
def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # dtype of missing values
    mis_val_dtype = df.dtypes

    # Make a table with the results
    mis_val_table = pd.concat(
        [mis_val, mis_val_percent, mis_val_dtype], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values', 2: 'Dtype'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


def get_numeric_columns(df):
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=['float64']).columns
    return list(numeric_cols)


def fix_outliers(df, columns, percentile=95):
    # Create a copy to avoid the "A value is trying to be set on a copy" warning
    df = df.copy()

    # Iterate over specified columns and fix outliers
    for col in columns:
        # Check if the column needs outlier fixing
        if col not in ["Bearer Id", "IMSI", "MSISDN/Number", "IMEI", "Start ms", "End ms", "Dur. (ms)", "Dur. (s)"]:
            # Calculate the threshold for outliers
            threshold = df[col].quantile(percentile / 100)

            # Replace outliers with the mean
            df.loc[df[col] > threshold, col] = df[col].mean()

    return df

def aggregate_user_data(df, app_columns):
    # If the columns have spaces, replace them with underscores for easier access
    df.columns = df.columns.str.replace(' ', '_')

    # Filter columns that exist in the DataFrame
    existing_columns = [col for col in app_columns if col in df.columns]

    # Group by 'MSISDN/Number' and aggregate the required information
    user_aggregation = df.groupby('MSISDN/Number').agg(
        Num_of_xDR_sessions=('Bearer_Id', 'count'),  # Number of xDR sessions
        Session_duration=('Dur._(ms).1', 'sum'),  # Session duration
        Total_DL=('Total_DL_(Bytes)', 'sum'),  # Total download data
        Total_UL=('Total_UL_(Bytes)', 'sum'),  # Total upload data
        **{f'{col}_DL': (col.replace(' ', '_'), 'sum') for col in existing_columns},  # DL data for each application
        **{f'{col}_UL': (col.replace(' ', '_'), 'sum') for col in existing_columns}  # UL data for each application
    ).reset_index()

    # Add a column for the total data volume (DL + UL) for each application
    user_aggregation['Total_Data_Volume'] = user_aggregation[
        [f'{col}_DL' for col in existing_columns] + [f'{col}_UL' for col in existing_columns]
    ].sum(axis=1)

    # Merge the aggregated data back to the original DataFrame
    df = pd.merge(df, user_aggregation, on='MSISDN/Number', how='left')

    return df

