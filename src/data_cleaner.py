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


def drop_columns_by_missing_percentage(df, percentage):
    """
    Drop columns in the DataFrame based on the given percentage threshold of missing values.

    Parameters:
    - df: DataFrame
        The input DataFrame.
    - percentage: float
        The threshold percentage for dropping columns.

    Returns:
    - df_cleaned: DataFrame
        The DataFrame with columns dropped based on the threshold.
    """
    # Calculate missing values information
    missing_values_info = missing_values_table(df)

    # Identify columns to drop based on the threshold percentage
    columns_to_drop = missing_values_info[missing_values_info['% of Total Values']
                                          > percentage].index

    # Drop the identified columns from the DataFrame
    df_cleaned = df.drop(columns=columns_to_drop)

    # Print information about dropped columns
    print(
        f"\nDropped {len(columns_to_drop)} columns based on the {percentage}% missing value threshold.")
    print("Remaining columns:", df_cleaned.shape[1])

    return df_cleaned


def fill_missing_values(df, column_names, fill_method='mean'):
    """
    Fill missing values in specified columns of the DataFrame based on the provided fill method.

    Parameters:
    - df: DataFrame
        The input DataFrame.
    - column_names: list
        List of column names with missing values to fill.
    - fill_method: str
        The fill method to use (default is 'mean'). Options: 'mean', 'median', 'mode', 'bfill', 'ffill', or a specific value.

    Returns:
    - df_filled: DataFrame
        The DataFrame with missing values filled in the specified columns.
    """
    # Make a copy of the DataFrame to avoid modifying the original
    df_filled = df.copy()

    # Choose the fill method
    if fill_method == 'mean':
        fill_values = df_filled[column_names].mean()
    elif fill_method == 'median':
        fill_values = df_filled[column_names].median()
    elif fill_method == 'mode':
        fill_values = df_filled[column_names].mode().iloc[0]
    elif fill_method == 'bfill':
        fill_values = df_filled[column_names].bfill()
    elif fill_method == 'ffill':
        fill_values = df_filled[column_names].ffill()
    else:
        fill_values = fill_method

    # Fill missing values in the specified columns
    df_filled[column_names] = df_filled[column_names].fillna(fill_values)

    # Print information about the filling process
    print(
        f"\nFilled missing values in columns {column_names} using '{fill_method}' method.")

    return df_filled


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


def convert_bytes_to_megabytes(df, *bytes_columns):
    """
    Convert specified columns from bytes to megabytes and rename columns.

    Parameters:
    - df: DataFrame
        The input DataFrame.
    - *bytes_columns: str
        Variable-length list of column names with bytes values.

    """
    # Copy the DataFrame to avoid modifying the original
    df_result = df.copy()

    megabyte = 1 * 10e+5  # 1 megabyte = 10^5 bytes

    # Iterate over specified columns
    for col in bytes_columns:
        # Convert bytes to megabytes
        df_result[col] = df_result[col] / megabyte

        # Rename the column by replacing 'Bytes' with 'MB'
        new_col_name = col.replace('Bytes', 'MB')
        df_result.rename(columns={col: new_col_name}, inplace=True)

    return df_result


def convert_ms_to_s(df, *ms_columns):
    """
    Convert specified columns from milliseconds to seconds and rename columns.

    Parameters:
    - df: DataFrame
        The input DataFrame.
    - *ms_columns: str
        Variable-length list of column names with milliseconds values.

    """
    # Copy the DataFrame to avoid modifying the original
    df_result = df.copy()

    # Iterate over specified columns
    for col in ms_columns:
        # Convert milliseconds to seconds
        df_result[col] = df_result[col] / 1000

        # Rename the column by replacing 'ms' with 's'
        new_col_name = col.replace('ms', 's')
        df_result.rename(columns={col: new_col_name}, inplace=True)

    return df_result


def convert_kbps_to_mbps(df, *kbps_columns):
    """
    Convert specified columns from kilobits per second (kbps) to megabits per second (mbps)
    and rename columns.

    Parameters:
    - df: DataFrame
        The input DataFrame.
    - *kbps_columns: str
        Variable-length list of column names with kbps values.

    Returns:
    - DataFrame
        A new DataFrame with specified columns converted to mbps and renamed.
    """
    # Copy the DataFrame to avoid modifying the original
    df_result = df.copy()

    # Iterate over specified columns
    for col in kbps_columns:
        # Convert kbps to mbps
        df_result[col] = df_result[col] / 1000

        # Rename the column by replacing 'kbps' with 'mbps'
        new_col_name = col.replace('kbps', 'mbps')
        df_result.rename(columns={col: new_col_name}, inplace=True)

    return df_result


def convert_column_names_kbps_to_mbps(df, *kbps_columns):
    """
    Convert column names from Kbps to Mbps in a DataFrame.

    Parameters:
    - df: DataFrame
        The input DataFrame.
    - *kbps_columns: str
        Variable-length list of column names with "Kbps" in their name.

    Returns:
    - DataFrame
        A new DataFrame with updated column names.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Conversion factor from Kbps to Mbps
    conversion_factor = 1e-3

    # Iterate through each specified column
    for kbps_column in kbps_columns:
        # Extract numeric part and convert to Mbps
        numeric_part = "".join(filter(str.isdigit, kbps_column))
        numeric_value = int(numeric_part) * conversion_factor

        # Replace "Kbps" with "Mbps" in the column name
        new_column_name = kbps_column.replace('Kbps', 'Mbps')

        # Update the numeric value in the column name
        new_column_name = new_column_name.replace(
            str(int(numeric_part)), str(numeric_value))

        # Rename the column in the copy of the DataFrame
        df_copy.rename(columns={kbps_column: new_column_name}, inplace=True)

    return df_copy
