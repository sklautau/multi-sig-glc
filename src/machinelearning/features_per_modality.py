'''
Multimodal machine learning for physiological signals.
Modality-aware: ECG ≠ PPG ≠ Bioimp

New strategy to extract features from set:
For each segment in given dataframe, extract the feature based on its modality.
Then, create a new dataframe for each modality (three dataframes in current
case), with the extracted features,
such that they can have the same columns, and where each row has the
corresponding feature values. Include in each row, all important information,
such as patient_id, quality_indicator, segment ID, etc, so that we can later merge the dataframes based
on these columns, and have a concatenaded dataframe with all features for each segment,
which can then be used for machine learning.
'''

import os
import numpy as np
import pandas as pd
import neurokit2 as nk
from typing import Dict, Any, Tuple

from datasets_util.naming_conventions import DatasetConfig
from datasets_util.waveforms import load_signal
from datasets_util.segments import SegmentManager, segments_to_fixed_duration_windows
from signal_processing.ecg import extract_ecg_features
from signal_processing.ppg import extract_ppg_features
from signal_processing.bioimpedance import extract_bioimp_features

# Set to True to slice segments into fixed-duration windows (e.g., 5 seconds)
USE_FIXED_DURATION_WINDOWS = True

# Input data is from the following files:
WAVEFORM_ID = "filtered"

# Metadata columns to include in all modality-specific dataframes
METADATA_COLUMNS = [
    "participant_id",
    "session_id",
    "datetime",
    "file_id",
    "modality",
    "segment_id",
    "quality_indicator",
    "GLC"
]

# If False, errors will be logged but the pipeline will continue, returning NaN or empty values for features that failed to extract.
RAISE_EXCEPTION_ON_PPG_PROCESSING = False


def extract_features_for_segment(row: pd.Series, fs_dict: Dict[str, int], datasetConfig: DatasetConfig) -> Dict[str, Any]:
    """
    Extract features for a single segment based on its modality.

    Args:
        row: A row from segments dataframe with segment info
        fs_dict: Sampling rates dict {"ecg": 250, "ppg": 100, "bioimp": 50}
        datasetConfig: Dataset configuration object

    Returns:
        Dictionary with extracted features
    """
    modality = row["modality"]
    file_id = row["file_id"]

    # Load signal
    path = datasetConfig.get_gen_complete_path(file_id, WAVEFORM_ID)
    print(f"Processing {file_id} ({modality}): {path}")

    try:
        signal = load_signal(path)
    except Exception as e:
        print(f"Error loading signal: {e}")
        return None

    features = {}

    # Extract features based on modality
    try:
        if modality == "ecg":
            features = extract_ecg_features(signal, fs_dict["ecg"])
        elif modality == "ppg":
            features = extract_ppg_features(signal, fs_dict["ppg"])
        elif modality in {"bioimp", "bioimpedance_frequency", "bioimpedance_time"}:
            features = extract_bioimp_features(signal, fs_dict["bioimp"])
        else:
            print(f"Unknown modality: {modality}")
            return None
    except Exception as e:
        if RAISE_EXCEPTION_ON_PPG_PROCESSING:
            raise Exception(e)
        else:
            print(
                f"Error in extract_features_for_segment() for {modality}. Message is: {e}")
            return None

    return features


def build_modality_dataframes(segmentManager: SegmentManager, fs_dict: Dict[str, int], datasetConfig: DatasetConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract features for each segment and create separate dataframes per modality.

    Returns:
        Tuple of (df_ecg, df_ppg, df_bioimp)
    """
    segments_df = segmentManager.get_segments_dataframe()
    df_dataset_info = datasetConfig.get_dataset_info_dataframe()

    # Merge segments with dataset info to get all metadata
    df_merged = segments_df.merge(
        df_dataset_info[["file_id", "participant_id",
                         "session_id", "datetime", "GLC"]],
        on="file_id",
        how="left"
    )

    # Initialize lists for each modality
    ecg_features = []
    ppg_features = []
    bioimp_features = []

    num_processed = 0
    num_errors = 0

    # Process each segment
    for idx, row in df_merged.iterrows():
        modality = row["modality"]

        # Extract features
        features = extract_features_for_segment(row, fs_dict, datasetConfig)

        if features is None:
            num_errors += 1
            continue

        # Add metadata to features
        segment_metadata = {col: row[col]
                            for col in METADATA_COLUMNS if col in row}
        features.update(segment_metadata)

        # Append to appropriate modality list
        if modality == "ecg":
            ecg_features.append(features)
        elif modality == "ppg":
            ppg_features.append(features)
        elif modality in {"bioimp", "bioimpedance_frequency", "bioimpedance_time"}:
            bioimp_features.append(features)

        num_processed += 1
        if num_processed % 10 == 0:
            print(f"Processed {num_processed} segments...")

    print(f"\nTotal segments processed: {num_processed}")
    print(f"Total errors: {num_errors}")

    # Create dataframes
    df_ecg = pd.DataFrame(ecg_features) if ecg_features else pd.DataFrame()
    df_ppg = pd.DataFrame(ppg_features) if ppg_features else pd.DataFrame()
    df_bioimp = pd.DataFrame(
        bioimp_features) if bioimp_features else pd.DataFrame()

    print(f"\nECG dataframe shape: {df_ecg.shape}")
    print(f"PPG dataframe shape: {df_ppg.shape}")
    print(f"Bioimpedance dataframe shape: {df_bioimp.shape}")

    return df_ecg, df_ppg, df_bioimp


def save_modality_dataframes(df_ecg: pd.DataFrame, df_ppg: pd.DataFrame, df_bioimp: pd.DataFrame, datasetConfig: DatasetConfig):
    """Save modality-specific dataframes to CSV files."""
    output_dir = datasetConfig.machine_learning_path

    if not df_ecg.empty:
        output_file = os.path.join(output_dir, "features_ecg_segments.csv")
        df_ecg.to_csv(output_file, index=False)
        print(f"ECG features saved to: {output_file}")

    if not df_ppg.empty:
        output_file = os.path.join(output_dir, "features_ppg_segments.csv")
        df_ppg.to_csv(output_file, index=False)
        print(f"PPG features saved to: {output_file}")

    if not df_bioimp.empty:
        output_file = os.path.join(output_dir, "features_bioimp_segments.csv")
        df_bioimp.to_csv(output_file, index=False)
        print(f"Bioimpedance features saved to: {output_file}")


if __name__ == "__main__":
    fs_dict = {
        "ecg": 500,
        "ppg": 500,
        "bioimp": np.nan  # set to NaN if not available or not applicable
    }

    # Create a DatasetConfig instance
    dataset_config_file = "dataset_folders.json"
    datasetConfig = DatasetConfig(dataset_config_file)

    # Load segments information
    dataset_segments = "all_segments_with_quality.csv"
    segments_path = os.path.join(datasetConfig.segments_path, dataset_segments)
    segmentManager = SegmentManager(datasetConfig, segments_path)

    # Here one can slice the segments into windows of fixed
    # duration or use the original segments as they are
    if USE_FIXED_DURATION_WINDOWS:
        # Convert segments to fixed-duration windows (5 seconds windows with 2 second shift)
        window_size_seconds = 5.0
        window_shift_seconds = 1.0
        fs = 500  # sampling frequency in Hz
        print("Slicing segments into fixed-duration windows...")
        windows_df = segments_to_fixed_duration_windows(
            segmentManager.get_segments_dataframe(),
            window_size_seconds=window_size_seconds,
            window_shift_seconds=window_shift_seconds,
            fs=fs
        )
        # update the SegmentManager object
        segmentManager = SegmentManager(datasetConfig, windows_df)

    # Extract features for each modality
    df_ecg, df_ppg, df_bioimp = build_modality_dataframes(
        segmentManager, fs_dict, datasetConfig)

    # Save dataframes
    save_modality_dataframes(df_ecg, df_ppg, df_bioimp, datasetConfig)

    print("\nFeature extraction complete!")
