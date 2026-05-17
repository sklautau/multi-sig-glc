import pandas as pd
from typing import List
import os
import pandas as pd


def remove_missing_data(
    df: pd.DataFrame,
    drop_rows: bool = True,
    drop_cols: bool = False,
    inplace: bool = False
) -> pd.DataFrame:
    """
    Remove rows and/or columns containing any missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    drop_rows : bool
        If True, drop rows with any NaN
    drop_cols : bool
        If True, drop columns with any NaN
    inplace : bool
        If True, modify df in place

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame (or None if inplace=True)
    """
    target = df if inplace else df.copy()

    if drop_rows:
        # drop rows with inf or NaN values
        target.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
        target.dropna(axis=0, how="any", inplace=True)

    if drop_cols:
        # drop columns with inf or NaN values
        target.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
        target.dropna(axis=1, how="any", inplace=True)

    return target


def find_duplicate_columns(df: pd.DataFrame) -> List[str]:
    """
    Returns a list of column names that appear more than once in the DataFrame.
    """
    cols = pd.Series(df.columns)
    duplicates = cols[cols.duplicated()].unique().tolist()
    return duplicates


def duplicate_column_counts(df: pd.DataFrame) -> pd.Series:
    """
    Returns a Series with counts of duplicated column names (only those >1).
    """
    counts = pd.Series(df.columns).value_counts()
    return counts[counts > 1]


if __name__ == "__main__":
    input_file = "../generated_waveforms/multimodal_dataset_example/machinelearning/multimodal_features.csv"
    # read input file
    df = pd.read_csv(input_file)
    dups = find_duplicate_columns(df)

    if dups:
        print("Duplicate columns found:", dups)
    else:
        print("No duplicate columns.")

    print(df.head())
    # drop columns called has_ppg,ppg_error, etc.
    # Use errors='ignore' to safely drop columns that may not exist
    df = df.drop(columns=["has_ppg", "has_ecg",
                          "ppg_error", "has_bioimp",
                          "file_id", "modality", "segment_id"], errors='ignore')

    print(df.head())

    df_clean = remove_missing_data(df, drop_rows=False, drop_cols=True)
    # df_clean = remove_missing_data(df, drop_rows=True, drop_cols=False)
    print("Input DataFrame shape:", df.shape)
    print("Cleaned DataFrame shape:", df_clean.shape)

    # check whether NaN exists in cleaned DataFrame
    if df_clean.isna().any().any():
        print("Warning: Cleaned DataFrame still contains NaN values.")
    else:
        print("No NaN values in cleaned DataFrame.")

    # check whether inf exists in cleaned DataFrame
    if df_clean.isin([float('inf'), float('-inf')]).any().any():
        print("Warning: Cleaned DataFrame still contains inf values.")
    else:
        print("No inf values in cleaned DataFrame.")

    # save to file
    output_file = os.path.dirname(input_file) + \
        "/multimodal_features_clean.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    df_clean.to_csv(output_file, index=False)

    print(f"Cleaned feature dataset saved to: {output_file}")
